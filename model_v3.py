#!/usr/bin/env python3
"""
Edge Weight Normalized MaxK Models
MODEL FILE ONLY - imports and uses the SpGEMM function
Respects your modular structure
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Function
from torch.nn import Linear
import torch.nn.init as init
import dgl
from dgl.utils import expand_as_pair

# Import YOUR SpGEMM function from the other file
from edge_weight_spgemm_function import (
    EdgeWeightNormalizedMaxKSpmmWrapper,
    edge_weight_normalized_maxk_spgemm
)

class OptimizedMaxK(Function):
    """MaxK activation that returns TopK info for optimization"""
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

class EdgeWeightNormalizedMaxKSAGEConv(nn.Module):
    """
    MaxK SAGE Convolution with edge weight normalization
    Uses your SpGEMM function from the other file
    """
    
    def __init__(self, in_feats, out_feats, aggregator_type='mean', 
                 feat_drop=0., bias=True, norm=None, activation=None, k_value=32):
        super().__init__()
        
        assert aggregator_type == 'mean', f"Only 'mean' supported, got {aggregator_type}"
        
        self._in_src_feats, self._in_dst_feats = expand_as_pair(in_feats)
        self._out_feats = out_feats
        self._aggre_type = aggregator_type
        self.norm = norm
        self.feat_drop = nn.Dropout(feat_drop)
        self.activation = activation
        self.k_value = k_value
        
        # Linear layers
        self.fc_neigh = nn.Linear(self._in_src_feats, out_feats, bias=False)
        self.fc_self = nn.Linear(self._in_dst_feats, out_feats, bias=bias)
        
        # Use YOUR wrapper from the function file
        self.maxk_wrapper = EdgeWeightNormalizedMaxKSpmmWrapper()
        
        # Graph data storage
        self.graph_data_set = False
        
        self.reset_parameters()
    
    def reset_parameters(self):
        gain = nn.init.calculate_gain("relu")
        nn.init.xavier_uniform_(self.fc_self.weight, gain=gain)
        nn.init.xavier_uniform_(self.fc_neigh.weight, gain=gain)
    
    def set_graph_data(self, graph, graph_name=""):
        """
        Prepare graph data with pre-normalized edge weights
        """
        assert graph_name, "graph_name REQUIRED"
        
        if self.graph_data_set:
            return
            
        print(f"🔧 Setting graph data for {graph_name}")
        
        # Extract graph representations
        graph = graph.local_var()
        device = graph.device
        
        # Get CSR and CSC formats
        indptr_csr, indices_csr, _ = graph.adj_tensors('csr')
        indptr_csc, indices_csc, _ = graph.adj_tensors('csc')
        
        # Compute degrees
        in_degrees = graph.in_degrees().float().to(device)
        out_degrees = graph.out_degrees().float().to(device)
        in_degrees = torch.clamp(in_degrees, min=1.0)
        out_degrees = torch.clamp(out_degrees, min=1.0)
        
        print(f"📊 Graph: {len(indices_csr)} CSR edges, {len(indices_csc)} CSC edges")
        
        # 🔥 PRE-COMPUTE NORMALIZED EDGE WEIGHTS
        
        # CSR: 1/in_degree[dst] for forward pass
        csr_weights_normalized = torch.zeros(len(indices_csr), device=device, dtype=torch.float32)
        for i in range(len(indptr_csr) - 1):
            start, end = indptr_csr[i], indptr_csr[i + 1]
            for edge_idx in range(start, end):
                dst_node = indices_csr[edge_idx]
                csr_weights_normalized[edge_idx] = 1.0 / in_degrees[dst_node]
        
        # CSC: 1/out_degree[src] for backward pass
        csc_weights_normalized = torch.zeros(len(indices_csc), device=device, dtype=torch.float32)
        for i in range(len(indptr_csc) - 1):
            start, end = indptr_csc[i], indptr_csc[i + 1]
            for edge_idx in range(start, end):
                src_node = indices_csc[edge_idx]
                csc_weights_normalized[edge_idx] = 1.0 / out_degrees[src_node]
        
        print(f"🔥 Edge weights normalized:")
        print(f"   CSR: range [{csr_weights_normalized.min():.6f}, {csr_weights_normalized.max():.6f}]")
        print(f"   CSC: range [{csc_weights_normalized.min():.6f}, {csc_weights_normalized.max():.6f}]")
        
        # Store graph data
        self.graph_indices_csr = indices_csr.int()
        self.graph_indptr_csr = indptr_csr.int()
        self.graph_values_csr_normalized = csr_weights_normalized
        
        self.graph_indices_csc = indices_csc.int()
        self.graph_values_csc_normalized = csc_weights_normalized
        
        # Load metadata using YOUR wrapper
        metadata_loaded = self.maxk_wrapper.load_metadata(graph_name)
        assert metadata_loaded, f"Metadata loading failed for {graph_name}"
        
        print(f"✅ Metadata loaded: {self.maxk_wrapper.num_warps} warps")
        
        self.graph_data_set = True
        print(f"✅ Graph setup complete for {graph_name}")
    
    def forward(self, graph, feat, topk_values=None, topk_indices=None):
        """
        Forward pass using YOUR SpGEMM function with edge weight normalization
        """
        assert self.graph_data_set, "Must call set_graph_data() first"
        assert topk_values is not None, "topk_values REQUIRED"
        assert topk_indices is not None, "topk_indices REQUIRED"
        
        with graph.local_scope():
            feat_src = feat_dst = self.feat_drop(feat)
            if graph.is_block:
                feat_dst = feat_src[:graph.number_of_dst_nodes()]
            
            h_self = feat_dst
            
            assert graph.num_edges() > 0, "Empty graphs not supported"
            
            print(f"🚀 Forward with edge weight normalization")
            
            # Transformation order
            lin_before_mp = self._in_src_feats > self._out_feats
            
            if lin_before_mp:
                # Transform BEFORE aggregation
                feat_to_aggregate = self.fc_neigh(topk_values)
                
                # Use YOUR SpGEMM function
                h_neigh = self.maxk_wrapper.spmm(
                    self.graph_indices_csr, self.graph_values_csr_normalized,
                    self.graph_indices_csc, self.graph_values_csc_normalized,
                    feat_to_aggregate, topk_indices, self.graph_indptr_csr
                )
            else:
                # Aggregate THEN transform
                h_neigh_aggregated = self.maxk_wrapper.spmm(
                    self.graph_indices_csr, self.graph_values_csr_normalized,
                    self.graph_indices_csc, self.graph_values_csc_normalized,
                    topk_values, topk_indices, self.graph_indptr_csr
                )
                h_neigh = self.fc_neigh(h_neigh_aggregated)
            
            # Combine features
            rst = self.fc_self(h_self) + h_neigh
            
            # Post-processing
            if self.activation is not None:
                rst = self.activation(rst)
            if self.norm is not None:
                rst = self.norm(rst)
            
            print(f"✅ Forward complete: {rst.shape}")
            return rst

class EdgeWeightNormalizedMaxKSAGE(nn.Module):
    """
    Complete SAGE model using edge weight normalization
    Imports and uses YOUR SpGEMM function
    """
    
    def __init__(self, in_size, hid_size, num_hid_layers, out_size, maxk=32, 
                 feat_drop=0.5, norm=False, nonlinear="maxk", graph_name=""):
        super().__init__()
        
        assert graph_name, "graph_name REQUIRED"
        assert nonlinear == "maxk", f"Only 'maxk' supported, got {nonlinear}"
        
        self.num_layers = num_hid_layers
        self.graph_name = graph_name
        self.nonlinear = nonlinear
        self.k_value = maxk
        
        print(f"🏗️ Building EdgeWeightNormalizedMaxKSAGE:")
        print(f"   Graph: {graph_name}, Layers: {num_hid_layers}, MaxK: {maxk}")
        
        # Build layers
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
        
        # Input/output transformations
        self.lin_in = Linear(in_size, hid_size)
        self.lin_out = Linear(hid_size, out_size)
        init.xavier_uniform_(self.lin_in.weight)
        init.xavier_uniform_(self.lin_out.weight)
        
        self.graph_set = False
    
    def set_graph(self, graph):
        """Configure all layers for the graph"""
        if not self.graph_set:
            print(f"🔧 Configuring {self.num_layers} layers for {self.graph_name}")
            for i, layer in enumerate(self.layers):
                print(f"   Layer {i}")
                layer.set_graph_data(graph, self.graph_name)
            self.graph_set = True
            print(f"✅ All layers configured")
    
    def forward(self, g, x):
        """
        Forward pass using edge weight normalization
        """
        if not self.graph_set:
            self.set_graph(g)
        
        print(f"🚀 EdgeWeightNormalizedMaxKSAGE forward: {x.shape}")
        
        # Input transformation
        x = self.lin_in(x)
        
        # Hidden layers with edge weight optimization
        for i in range(self.num_layers):
            print(f"   Layer {i}")
            
            # Apply MaxK and get TopK info
            x_sparse, topk_values, topk_indices = OptimizedMaxK.apply(x, self.k_value)
            
            # Pass TopK info to layer (uses YOUR SpGEMM function)
            x = self.layers[i](g, x_sparse, topk_values, topk_indices)
            
            print(f"     Output: {x.shape}")
        
        # Output transformation
        x = self.lin_out(x)
        print(f"✅ Model forward complete: {x.shape}")
        
        return x

def test_edge_weight_models():
    """Test the edge weight normalized models"""
    print("🧪 Testing Edge Weight Normalized Models")
    print("=" * 50)
    
    assert torch.cuda.is_available(), "CUDA REQUIRED"
    
    # Test setup
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
    
    features = torch.randn(num_nodes, feat_dim, device='cuda', requires_grad=True)
    
    print(f"📊 Test: {num_nodes} nodes, {num_edges} edges, k={k_value}")
    
    try:
        # Test model
        model = EdgeWeightNormalizedMaxKSAGE(
            feat_dim, hidden_dim, 2, output_dim, 
            maxk=k_value, graph_name="test_graph"
        ).cuda()
        
        print(f"🏗️ Model created")
        
        # Forward pass
        output = model(g, features)
        print(f"✅ Forward: {output.shape}")
        
        # Backward pass  
        loss = output.sum()
        loss.backward()
        print(f"✅ Backward: gradients computed")
        
        # Validate
        assert not torch.isnan(output).any(), "NaN in output"
        assert features.grad is not None, "No input gradients"
        
        print(f"📊 Results:")
        print(f"   Output mean: {output.mean().item():.6f}")
        print(f"   Output std: {output.std().item():.6f}")
        
        print(f"🎉 Edge Weight Model Test PASSED!")
        
    except Exception as e:
        print(f"💀 FATAL ERROR in model: {e}")
        raise

if __name__ == "__main__":
    test_edge_weight_models()
