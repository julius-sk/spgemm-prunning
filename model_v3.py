#!/usr/bin/env python3
"""
Edge Weight Normalization for MaxK SAGE
Bakes degree normalization directly into edge weights for efficiency
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Function
from torch.nn import Linear
import torch.nn.init as init
import dgl
from dgl.utils import expand_as_pair

# Import MaxK SpGEMM components
try:
    from maxk_spgemm_function import MaxKSpmmWrapper, MAXK_KERNELS_AVAILABLE
    import maxk_cuda_kernels
except ImportError:
    MAXK_KERNELS_AVAILABLE = False

class EdgeWeightNormalizedMaxKSAGEConv(nn.Module):
    """
    MaxK SAGE Convolution with edge weight normalization
    Uses 1/in_degree for forward CSR and 1/out_degree for backward CSC
    """
    
    def __init__(self, in_feats, out_feats, aggregator_type='mean', 
                 feat_drop=0., bias=True, norm=None, activation=None, k_value=32):
        super().__init__()
        
        # Validate aggregator type
        valid_aggre_types = {"mean", "gcn", "pool", "lstm"}
        if aggregator_type not in valid_aggre_types:
            raise ValueError(f"Invalid aggregator_type: {aggregator_type}")
        
        # Store parameters
        self._in_src_feats, self._in_dst_feats = expand_as_pair(in_feats)
        self._out_feats = out_feats
        self._aggre_type = aggregator_type
        self.norm = norm
        self.feat_drop = nn.Dropout(feat_drop)
        self.activation = activation
        self.k_value = k_value
        
        # Linear layers
        self.fc_neigh = nn.Linear(self._in_src_feats, out_feats, bias=False)
        
        if aggregator_type != "gcn":
            self.fc_self = nn.Linear(self._in_dst_feats, out_feats, bias=bias)
        elif bias:
            self.bias = nn.parameter.Parameter(torch.zeros(self._out_feats))
        else:
            self.register_buffer("bias", None)
        
        # MaxK SpGEMM wrapper
        if MAXK_KERNELS_AVAILABLE:
            self.maxk_wrapper = MaxKSpmmWrapper()
        else:
            self.maxk_wrapper = None
        
        # Graph data storage
        self.graph_data_set = False
        self.use_maxk_kernel = False
        
        self.reset_parameters()
    
    def reset_parameters(self):
        """Initialize parameters"""
        gain = nn.init.calculate_gain("relu")
        if self._aggre_type != "gcn":
            nn.init.xavier_uniform_(self.fc_self.weight, gain=gain)
        nn.init.xavier_uniform_(self.fc_neigh.weight, gain=gain)
    
    def set_graph_data(self, graph, graph_name=""):
        """
        Set graph data with pre-computed normalized edge weights
        OPTIMIZED: Bakes degree normalization into edge weights
        """
        if self.graph_data_set:
            return
            
        # Extract graph information
        graph = graph.local_var()
        device = graph.device
        
        # Get CSR and CSC representations
        indptr, indices, _ = graph.adj_tensors('csr')
        csc_indptr, csc_indices, _ = graph.adj_tensors('csc')
        
        # Compute degrees
        in_degrees = graph.in_degrees().float().to(device)
        out_degrees = graph.out_degrees().float().to(device)
        
        # Avoid division by zero for isolated nodes
        in_degrees = torch.clamp(in_degrees, min=1.0)
        out_degrees = torch.clamp(out_degrees, min=1.0)
        
        # 🔥 KEY OPTIMIZATION: Pre-compute normalized edge weights
        
        # For forward pass (CSR): Use 1/in_degree normalization
        # Each edge (u,v) gets weight 1/in_degree[v]
        forward_edge_weights = torch.zeros(len(indices), device=device, dtype=torch.float32)
        for i in range(len(indptr) - 1):  # For each source node
            start, end = indptr[i], indptr[i + 1]
            for edge_idx in range(start, end):
                dst_node = indices[edge_idx]
                forward_edge_weights[edge_idx] = 1.0 / in_degrees[dst_node]
        
        # For backward pass (CSC): Use 1/out_degree normalization  
        # Each edge (u,v) gets weight 1/out_degree[u]
        backward_edge_weights = torch.zeros(len(csc_indices), device=device, dtype=torch.float32)
        for i in range(len(csc_indptr) - 1):  # For each destination node
            start, end = csc_indptr[i], csc_indptr[i + 1]
            for edge_idx in range(start, end):
                src_node = csc_indices[edge_idx]
                backward_edge_weights[edge_idx] = 1.0 / out_degrees[src_node]
        
        # Store normalized graph data
        self.graph_indices = indices.int()
        self.graph_indptr = indptr.int()
        self.graph_values_normalized = forward_edge_weights  # 🔥 Pre-normalized!
        
        self.graph_indices_T = csc_indices.int()
        self.graph_indptr_T = csc_indptr.int()
        self.graph_values_T_normalized = backward_edge_weights  # 🔥 Pre-normalized!
        
        # Store raw degrees for debugging/validation
        self.in_degrees = in_degrees
        self.out_degrees = out_degrees
        
        print(f"🔥 Pre-computed normalized edge weights:")
        print(f"   Forward weights range: [{forward_edge_weights.min():.6f}, {forward_edge_weights.max():.6f}]")
        print(f"   Backward weights range: [{backward_edge_weights.min():.6f}, {backward_edge_weights.max():.6f}]")
        print(f"   Forward weights mean: {forward_edge_weights.mean():.6f}")
        print(f"   Backward weights mean: {backward_edge_weights.mean():.6f}")
        
        # Load MaxK metadata
        if (MAXK_KERNELS_AVAILABLE and graph_name and self.maxk_wrapper and 
            self._aggre_type == "mean"):
            self.metadata_loaded = self.maxk_wrapper.load_metadata(graph_name)
            if self.metadata_loaded:
                self.use_maxk_kernel = True
                print(f"✅ MaxK kernel with edge weight normalization ready for {graph_name}")
            else:
                print(f"⚠️ MaxK metadata failed for {graph_name}")
        
        self.graph_data_set = True
    
    def forward(self, graph, feat, topk_values=None, topk_indices=None):
        """
        Forward pass with edge weight normalization
        NO post-processing normalization needed!
        """
        if not self.graph_data_set:
            raise RuntimeError("Graph data not set. Call set_graph_data() first")
        
        with graph.local_scope():
            # Handle input features
            feat_src = feat_dst = self.feat_drop(feat)
            if graph.is_block:
                feat_dst = feat_src[:graph.number_of_dst_nodes()]
            
            h_self = feat_dst
            
            # Handle empty graphs
            if graph.num_edges() == 0:
                h_neigh = torch.zeros(feat_dst.shape[0], self._in_src_feats).to(feat_dst)
            else:
                # Check if we can use MaxK kernel with pre-computed TopK
                if (self.use_maxk_kernel and topk_values is not None and 
                    topk_indices is not None and self._aggre_type == "mean"):
                    
                    try:
                        print(f"🚀 Using MaxK kernel with edge weight normalization")
                        
                        # Convert indices to uint8 for kernel
                        sparse_selector = topk_indices.to(torch.uint8)
                        
                        # Decide transformation order
                        lin_before_mp = self._in_src_feats > self._out_feats
                        
                        if lin_before_mp:
                            # Transform BEFORE aggregation
                            feat_to_aggregate = self.fc_neigh(topk_values)
                            h_neigh = self._maxk_spmm_normalized(
                                feat_to_aggregate, sparse_selector, forward=True
                            )
                        else:
                            # Aggregate THEN transform
                            h_neigh_aggregated = self._maxk_spmm_normalized(
                                topk_values, sparse_selector, forward=True
                            )
                            h_neigh = self.fc_neigh(h_neigh_aggregated)
                        
                        print(f"✅ MaxK kernel with edge weights successful")
                        
                    except Exception as e:
                        print(f"⚠️ MaxK kernel failed: {e}, falling back")
                        h_neigh = self._standard_aggregation(graph, feat_src)
                else:
                    # Standard DGL aggregation
                    h_neigh = self._standard_aggregation(graph, feat_src)
            
            # Combine self and neighbor features
            if self._aggre_type == "gcn":
                rst = h_neigh
                if self.bias is not None:
                    rst = rst + self.bias
            else:
                rst = self.fc_self(h_self) + h_neigh
            
            # Post-processing
            if self.activation is not None:
                rst = self.activation(rst)
            if self.norm is not None:
                rst = self.norm(rst)
            
            return rst
    
    def _maxk_spmm_normalized(self, data, sparse_selector, forward=True):
        """
        Run MaxK SpMM with pre-normalized edge weights
        NO additional normalization needed!
        """
        if forward:
            # Forward pass: use CSR with 1/in_degree weights
            output = maxk_cuda_kernels.spmm_maxk_forward(
                self.maxk_wrapper.warp4_metadata,
                self.graph_indices,
                self.graph_values_normalized,  # 🔥 Pre-normalized weights!
                data,
                sparse_selector,
                self.maxk_wrapper.num_warps,
                self.k_value
            )
        else:
            # Backward pass: use CSC with 1/out_degree weights
            output = maxk_cuda_kernels.spmm_maxk_backward(
                self.maxk_wrapper.warp4_metadata,
                self.graph_indices_T,
                self.graph_values_T_normalized,  # 🔥 Pre-normalized weights!
                data,
                sparse_selector,
                self.maxk_wrapper.num_warps,
                self.k_value
            )
        
        return output  # Already normalized - no post-processing needed!
    
    def _standard_aggregation(self, graph, feat_src):
        """Standard DGL aggregation fallback with manual normalization"""
        from dgl import function as fn
        
        lin_before_mp = self._in_src_feats > self._out_feats
        graph.srcdata["h"] = (
            self.fc_neigh(feat_src) if lin_before_mp else feat_src
        )
        graph.update_all(fn.copy_u("h", "m"), fn.mean("m", "neigh"))
        h_neigh = graph.dstdata["neigh"]
        if not lin_before_mp:
            h_neigh = self.fc_neigh(h_neigh)
        return h_neigh

class OptimizedMaxK(Function):
    """MaxK function that returns sparse output and TopK info"""
    @staticmethod
    def forward(ctx, input, k=1):
        topk_values, topk_indices = input.topk(k, dim=1)
        mask = torch.zeros_like(input)
        mask.scatter_(1, topk_indices, 1)
        output = input * mask
        ctx.save_for_backward(mask)
        return output, topk_values, topk_indices
    
    @staticmethod
    def backward(ctx, grad_output, grad_topk_values, grad_topk_indices):
        mask, = ctx.saved_tensors
        grad_input = grad_output * mask
        return grad_input, None

class EdgeWeightNormalizedMaxKSAGE(nn.Module):
    """
    Complete SAGE model with edge weight normalization
    """
    
    def __init__(self, in_size, hid_size, num_hid_layers, out_size, maxk=32, 
                 feat_drop=0.5, norm=False, nonlinear="maxk", graph_name=""):
        super().__init__()
        
        self.num_layers = num_hid_layers
        self.graph_name = graph_name
        self.nonlinear = nonlinear
        self.k_value = maxk
        
        # Build layers with edge weight normalization
        self.layers = nn.ModuleList()
        for i in range(self.num_layers):
            if norm:
                norm_layer = nn.LayerNorm(hid_size, elementwise_affine=True)
            else:
                norm_layer = None
            
            layer = EdgeWeightNormalizedMaxKSAGEConv(
                in_feats=hid_size,
                out_feats=hid_size,
                aggregator_type='mean',
                feat_drop=feat_drop,
                norm=norm_layer,
                k_value=maxk
            )
            self.layers.append(layer)
        
        # Input and output layers
        self.lin_in = Linear(in_size, hid_size)
        self.lin_out = Linear(hid_size, out_size)
        init.xavier_uniform_(self.lin_in.weight)
        init.xavier_uniform_(self.lin_out.weight)
        
        self.graph_set = False
    
    def set_graph(self, graph):
        """Set graph data for all layers"""
        if not self.graph_set:
            for layer in self.layers:
                if isinstance(layer, EdgeWeightNormalizedMaxKSAGEConv):
                    layer.set_graph_data(graph, self.graph_name)
            self.graph_set = True
            print(f"✅ Graph data set for edge weight normalized MaxK-SAGE")
    
    def forward(self, g, x):
        """Forward pass with edge weight normalization"""
        if not self.graph_set:
            self.set_graph(g)
        
        # Input transformation
        x = self.lin_in(x)
        
        # Hidden layers with optimized TopK and edge weight normalization
        for i in range(self.num_layers):
            if self.nonlinear == 'maxk':
                # Get TopK info and pass to SAGE layer
                x_sparse, topk_values, topk_indices = OptimizedMaxK.apply(x, self.k_value)
                x = self.layers[i](g, x_sparse, topk_values, topk_indices)
            elif self.nonlinear == 'relu':
                x = F.relu(x)
                x = self.layers[i](g, x)
            else:
                x = self.layers[i](g, x)
        
        # Output transformation
        x = self.lin_out(x)
        return x

def test_edge_weight_normalization():
    """Test edge weight normalization approach"""
    print("🧪 Testing Edge Weight Normalization")
    print("=" * 50)
    
    if not torch.cuda.is_available():
        print("❌ CUDA not available")
        return
    
    # Create test data
    num_nodes = 1000
    num_edges = 5000
    feat_dim = 128
    hidden_dim = 64
    output_dim = 10
    k_value = 32
    
    # Create test graph
    src = torch.randint(0, num_nodes, (num_edges,))
    dst = torch.randint(0, num_nodes, (num_edges,))
    g = dgl.graph((src, dst), num_nodes=num_nodes).to('cuda')
    
    # Create test features
    features = torch.randn(num_nodes, feat_dim).cuda()
    
    print(f"📊 Test: {num_nodes} nodes, {num_edges} edges, k={k_value}")
    
    # Test edge weight normalized model
    print(f"\n🔥 Testing Edge Weight Normalized MaxK-SAGE...")
    try:
        model = EdgeWeightNormalizedMaxKSAGE(
            feat_dim, hidden_dim, 2, output_dim, 
            maxk=k_value, graph_name="test_graph"
        ).cuda()
        
        # Forward pass
        output = model(g, features)
        print(f"✅ Forward pass successful: {output.shape}")
        
        # Backward pass
        loss = output.sum()
        loss.backward()
        print(f"✅ Backward pass successful")
        
        # Validate normalization by checking output statistics
        print(f"📊 Output statistics:")
        print(f"   Mean: {output.mean().item():.6f}")
        print(f"   Std: {output.std().item():.6f}")
        print(f"   Min: {output.min().item():.6f}")
        print(f"   Max: {output.max().item():.6f}")
        
        # Check if gradients are reasonable
        total_grad_norm = 0
        for param in model.parameters():
            if param.grad is not None:
                total_grad_norm += param.grad.norm().item()
        print(f"   Total gradient norm: {total_grad_norm:.6f}")
        
        print(f"🎉 Edge weight normalization test completed!")
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_edge_weight_normalization()
