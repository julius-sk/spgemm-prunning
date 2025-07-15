#!/usr/bin/env python3
"""
Optimized MaxK Integration - Pass TopK results directly to avoid double computation
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Function
from torch.nn import Linear
import torch.nn.init as init

# Import MaxK SpGEMM components
try:
    from maxk_spgemm_function import MaxKSpmmWrapper, MAXK_KERNELS_AVAILABLE
    import maxk_cuda_kernels
except ImportError:
    MAXK_KERNELS_AVAILABLE = False
    print("⚠️ MaxK CUDA kernels not available")

class OptimizedMaxK(Function):
    """
    Optimized MaxK that returns both the sparse output AND the TopK selection info
    This avoids recomputing TopK in the SpGEMM function
    """
    @staticmethod
    def forward(ctx, input, k=1):
        # Get TopK values and indices
        topk_values, topk_indices = input.topk(k, dim=1)
        
        # Create sparse output (standard MaxK behavior)
        mask = torch.zeros_like(input)
        mask.scatter_(1, topk_indices, 1)
        output = input * mask
        
        # Save for backward
        ctx.save_for_backward(mask)
        
        # Return both output and TopK info
        return output, topk_values, topk_indices
    
    @staticmethod
    def backward(ctx, grad_output, grad_topk_values, grad_topk_indices):
        mask, = ctx.saved_tensors
        grad_input = grad_output * mask
        return grad_input, None

class OptimizedMaxKSAGEConv(nn.Module):
    """
    SAGE Convolution that efficiently passes TopK results to SpGEMM kernel
    Eliminates double TopK computation
    """
    
    def __init__(self, in_feats, out_feats, aggregator_type='mean', 
                 feat_drop=0., bias=True, norm=None, activation=None, k_value=32):
        super().__init__()
        
        # Store parameters exactly like before
        self._in_src_feats = in_feats
        self._out_feats = out_feats
        self._aggre_type = aggregator_type
        self.norm = norm
        self.feat_drop = nn.Dropout(feat_drop)
        self.activation = activation
        self.k_value = k_value
        
        # Linear layers
        self.fc_neigh = nn.Linear(self._in_src_feats, out_feats, bias=False)
        if aggregator_type != "gcn":
            self.fc_self = nn.Linear(self._in_src_feats, out_feats, bias=bias)
        elif bias:
            self.bias = nn.parameter.Parameter(torch.zeros(self._out_feats))
        else:
            self.register_buffer("bias", None)
        
        # MaxK SpGEMM wrapper
        if MAXK_KERNELS_AVAILABLE:
            self.maxk_wrapper = MaxKSpmmWrapper()
        else:
            self.maxk_wrapper = None
        
        # Graph data (set during initialization)
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
        """Set graph data for MaxK kernel usage"""
        if self.graph_data_set:
            return  # Already set
            
        # Extract graph information
        graph = graph.local_var()
        device = graph.device
        
        # Get CSR and CSC representations
        indptr, indices, _ = graph.adj_tensors('csr')
        csc_indptr, csc_indices, _ = graph.adj_tensors('csc')
        
        # Store graph data
        self.graph_indices = indices.int()
        self.graph_indptr = indptr.int()
        self.graph_indices_T = csc_indices.int()
        self.graph_values = torch.ones(len(indices), device=device, dtype=torch.float32)
        self.graph_values_T = torch.ones(len(csc_indices), device=device, dtype=torch.float32)
        
        # Compute degrees for normalization
        self.in_degrees = graph.in_degrees().float().to(device)
        self.out_degrees = graph.out_degrees().float().to(device)
        self.in_degrees = torch.clamp(self.in_degrees, min=1.0)
        self.out_degrees = torch.clamp(self.out_degrees, min=1.0)
        
        # Load MaxK metadata
        if (MAXK_KERNELS_AVAILABLE and graph_name and self.maxk_wrapper and 
            self._aggre_type == "mean"):
            self.metadata_loaded = self.maxk_wrapper.load_metadata(graph_name)
            if self.metadata_loaded:
                self.use_maxk_kernel = True
                print(f"✅ Optimized MaxK kernel ready for {graph_name}")
            else:
                print(f"⚠️ MaxK metadata failed for {graph_name}")
        
        self.graph_data_set = True
    
    def forward(self, graph, feat, topk_values=None, topk_indices=None):
        """
        Forward pass with optimized TopK handling
        
        Args:
            graph: DGL graph
            feat: Input features OR sparse output from MaxK (if topk_values provided)
            topk_values: Pre-computed TopK values (from MaxK layer)
            topk_indices: Pre-computed TopK indices (from MaxK layer)
        """
        # Set graph data if not already done
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
                # Check if we can use optimized MaxK kernel
                if (self.use_maxk_kernel and topk_values is not None and 
                    topk_indices is not None and self._aggre_type == "mean"):
                    
                    try:
                        # Use pre-computed TopK results directly
                        print(f"🚀 Using optimized MaxK kernel with pre-computed TopK")
                        
                        # Convert indices to uint8 for kernel
                        sparse_selector = topk_indices.to(torch.uint8)
                        
                        # Decide transformation order
                        lin_before_mp = self._in_src_feats > self._out_feats
                        
                        if lin_before_mp:
                            # Transform BEFORE aggregation
                            feat_to_aggregate = self.fc_neigh(topk_values)
                            h_neigh_sum = self.maxk_wrapper.spmm(
                                self.graph_indices, self.graph_values,
                                feat_to_aggregate, self.k_value, self.graph_indptr,
                                self.in_degrees, self.out_degrees,
                                self.graph_indices_T, self.graph_values_T
                            )
                            h_neigh = h_neigh_sum  # Already normalized in kernel
                        else:
                            # Aggregate THEN transform
                            h_neigh_sum = self.maxk_wrapper.spmm(
                                self.graph_indices, self.graph_values,
                                topk_values, self.k_value, self.graph_indptr,
                                self.in_degrees, self.out_degrees,
                                self.graph_indices_T, self.graph_values_T
                            )
                            h_neigh = self.fc_neigh(h_neigh_sum)  # Already normalized
                        
                        print(f"✅ Optimized MaxK kernel successful")
                        
                    except Exception as e:
                        print(f"⚠️ Optimized MaxK kernel failed: {e}, falling back")
                        # Fall back to standard DGL
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
    
    def _standard_aggregation(self, graph, feat_src):
        """Standard DGL aggregation fallback"""
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

class OptimizedMaxKSAGE(nn.Module):
    """
    Optimized SAGE model that efficiently passes TopK results between layers
    Eliminates redundant TopK computations
    """
    
    def __init__(self, in_size, hid_size, num_hid_layers, out_size, maxk=32, 
                 feat_drop=0.5, norm=False, nonlinear="maxk", graph_name=""):
        super().__init__()
        
        self.num_layers = num_hid_layers
        self.graph_name = graph_name
        self.nonlinear = nonlinear
        self.k_value = maxk
        
        # Build optimized layers
        self.layers = nn.ModuleList()
        for i in range(self.num_layers):
            if norm:
                norm_layer = nn.LayerNorm(hid_size, elementwise_affine=True)
            else:
                norm_layer = None
            
            layer = OptimizedMaxKSAGEConv(
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
                if isinstance(layer, OptimizedMaxKSAGEConv):
                    layer.set_graph_data(graph, self.graph_name)
            self.graph_set = True
            print(f"✅ Graph data set for optimized MaxK-SAGE model")
    
    def forward(self, g, x):
        """
        Optimized forward pass that passes TopK results between layers
        """
        # Set graph on first forward pass
        if not self.graph_set:
            self.set_graph(g)
        
        # Input transformation
        x = self.lin_in(x)
        
        # Hidden layers with optimized TopK passing
        for i in range(self.num_layers):
            # Apply MaxK activation and get TopK info
            if self.nonlinear == 'maxk':
                # Use optimized MaxK that returns TopK info
                x_sparse, topk_values, topk_indices = OptimizedMaxK.apply(x, self.k_value)
                
                # Pass TopK info directly to SAGE layer (OPTIMIZATION!)
                x = self.layers[i](g, x_sparse, topk_values, topk_indices)
                
                print(f"📊 Layer {i}: Used optimized TopK passing")
            elif self.nonlinear == 'relu':
                x = F.relu(x)
                # Standard SAGE layer without TopK optimization
                x = self.layers[i](g, x)
            else:
                # No activation
                x = self.layers[i](g, x)
        
        # Output transformation
        x = self.lin_out(x)
        return x

def test_optimization():
    """Test the optimization to ensure it works correctly"""
    print("🧪 Testing Optimized MaxK Integration")
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
    import dgl
    src = torch.randint(0, num_nodes, (num_edges,))
    dst = torch.randint(0, num_nodes, (num_edges,))
    g = dgl.graph((src, dst), num_nodes=num_nodes).to('cuda')
    
    # Create test features
    features = torch.randn(num_nodes, feat_dim).cuda()
    
    print(f"📊 Test: {num_nodes} nodes, {num_edges} edges, k={k_value}")
    
    # Test optimized model
    print(f"\n🚀 Testing Optimized MaxK-SAGE...")
    try:
        model = OptimizedMaxKSAGE(
            feat_dim, hidden_dim, 2, output_dim, 
            maxk=k_value, graph_name="test_graph"
        ).cuda()
        
        # Forward pass
        output = model(g, features)
        print(f"✅ Optimized forward pass: {output.shape}")
        
        # Backward pass
        loss = output.sum()
        loss.backward()
        print(f"✅ Optimized backward pass successful")
        
        # Check output statistics
        print(f"📊 Output stats: mean={output.mean().item():.4f}, "
              f"std={output.std().item():.4f}")
        
    except Exception as e:
        print(f"❌ Optimized model failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_optimization()
