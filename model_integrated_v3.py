#!/usr/bin/env python3
"""
Complete MaxK Model Integration with GCN and GIN
Integrates optimized MaxK SAGE, GCN, and GIN implementations
Uses simplified SpGEMM function for undirected graphs
"""

import dgl
import dgl.nn as dglnn
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Linear
import torch.nn.init as init
from torch.autograd import Function
import math
from dgl import function as fn
from dgl.utils import check_eq_shape, expand_as_pair

# Import our optimized SpGEMM function
try:
    from maxk_spgemm_function import MaxKSpmmWrapper, maxk_spgemm, MAXK_KERNELS_AVAILABLE
    print("✅ MaxK CUDA kernels loaded for training integration")
except ImportError:
    MAXK_KERNELS_AVAILABLE = False
    print("⚠️ MaxK CUDA kernels not available, falling back to DGL")
    
class OPTMaxK(Function):
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
    
class MaxK(Function):
    """Standard MaxK activation function"""
    @staticmethod
    def forward(ctx, input, k=1):
        topk_values, topk_indices = input.topk(k, dim=1)
        mask = torch.zeros_like(input)
        mask.scatter_(1, topk_indices, 1)
        output = input * mask
        ctx.save_for_backward(mask)
        return output
    
    @staticmethod
    def backward(ctx, grad_output):
        mask, = ctx.saved_tensors
        grad_input = grad_output * mask
        return grad_input, None

class MaxKSAGEConv(nn.Module):
    """
    MaxK SAGE Convolution with optimized SpGEMM for undirected graphs
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
        
        # MaxK SpGEMM wrapper
        self.maxk_wrapper = MaxKSpmmWrapper()
        
        # Graph data storage
        self.graph_data_set = False
        
        self.reset_parameters()
    
    def reset_parameters(self):
        gain = nn.init.calculate_gain("relu")
        nn.init.xavier_uniform_(self.fc_self.weight, gain=gain)
        nn.init.xavier_uniform_(self.fc_neigh.weight, gain=gain)
    
    def set_graph_data(self, graph, graph_name=""):
        """
        Prepare graph data for undirected graphs (simplified)
        """
        assert graph_name, "graph_name REQUIRED"
        
        if self.graph_data_set:
            return
            
        print(f"🔧 Setting SAGE graph data for {graph_name}")
        
        # Extract graph representations
        graph = graph.local_var()
        device = graph.device
        
        # Get CSR format (only need one for undirected graphs)
        indptr_csr, indices_csr, edge_values = graph.adj_tensors('csr')
        
        # Compute degrees (for undirected graphs: in_degree = out_degree)
        degrees = graph.in_degrees().float().to(device)
        degrees = torch.clamp(degrees, min=1.0)  # Avoid division by zero
        
        # Count self-edges for debugging
        src, dst = graph.edges()
        self_edges = (src == dst).sum().item()
        print(f"   Self-edges: {self_edges}")
        print(f"   Degrees: min={degrees.min():.0f}, max={degrees.max():.0f}, mean={degrees.mean():.2f}")
        
        # Create uniform edge weights
        csr_values = torch.ones(len(indices_csr), device=device, dtype=torch.float32)
        
        # Store simplified graph data
        self.graph_indices = indices_csr.int()
        self.graph_indptr = indptr_csr.int()
        self.graph_values = csr_values
        self.degrees = degrees
        
        # Load metadata
        metadata_loaded = self.maxk_wrapper.load_metadata(graph_name)
        assert metadata_loaded, f"Metadata loading failed for {graph_name}"
        
        print(f"   ✅ Metadata loaded: {self.maxk_wrapper.num_warps} warps")
        
        self.graph_data_set = True
        print(f"✅ SAGE graph setup complete for {graph_name}")
    
    def forward(self, graph, feat, topk_values=None, topk_indices=None):
        """
        Forward pass using optimized SpGEMM for undirected graphs
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
            
            # Transformation order
            lin_before_mp = self._in_src_feats > self._out_feats
            
            if lin_before_mp:
                # Transform BEFORE aggregation
                feat_to_aggregate = self.fc_neigh(topk_values)
                
                # Use optimized SpGEMM (simplified for undirected graphs)
                h_neigh = self.maxk_wrapper.spmm(
                    self.graph_indices, self.graph_values,
                    feat_to_aggregate, topk_indices,
                    self.graph_indptr, self.degrees
                )
            else:
                # Aggregate THEN transform
                h_neigh_aggregated = self.maxk_wrapper.spmm(
                    self.graph_indices, self.graph_values,
                    topk_values, topk_indices,
                    self.graph_indptr, self.degrees
                )
                h_neigh = self.fc_neigh(h_neigh_aggregated)
            
            # Combine features
            rst = self.fc_self(h_self) + h_neigh
            
            # Post-processing
            if self.activation is not None:
                rst = self.activation(rst)
            if self.norm is not None:
                rst = self.norm(rst)
            
            return rst

class MaxKGraphConv(nn.Module):
    """
    MaxK-accelerated GraphConv (GCN) for undirected graphs
    """
    
    def __init__(self, in_feats, out_feats, norm="both", weight=True, bias=True, 
                 activation=None, allow_zero_in_degree=False, k_value=32):
        super(MaxKGraphConv, self).__init__()
        
        # Validate norm parameter
        if norm not in ("none", "both", "right", "left"):
            raise ValueError(f'Invalid norm value. Must be either "none", "both", "right" or "left". But got "{norm}".')
        
        self._in_feats = in_feats
        self._out_feats = out_feats
        self._norm = norm
        self._allow_zero_in_degree = allow_zero_in_degree
        self.k_value = k_value
        
        # Weight and bias
        if weight:
            self.weight = nn.Parameter(torch.Tensor(in_feats, out_feats))
        else:
            self.register_parameter("weight", None)
        
        if bias:
            self.bias = nn.Parameter(torch.Tensor(out_feats))
        else:
            self.register_parameter("bias", None)
        
        self.reset_parameters()
        self._activation = activation
        
        # MaxK SpGEMM wrapper
        if MAXK_KERNELS_AVAILABLE:
            self.maxk_wrapper = MaxKSpmmWrapper()
        else:
            self.maxk_wrapper = None
        
        self.graph_data_set = False
    
    def reset_parameters(self):
        if self.weight is not None:
            init.xavier_uniform_(self.weight)
        if self.bias is not None:
            init.zeros_(self.bias)
    
    def set_graph_data(self, graph, graph_name=""):
        """Set graph data for undirected graphs"""
        if self.graph_data_set:
            return
            
        print(f"🔧 Setting GCN graph data for {graph_name}")
        
        graph = graph.local_var()
        
        # Get CSR representation
        indptr, indices, _ = graph.adj_tensors('csr')
        
        # Store graph data
        self.graph_indices = indices.int()
        self.graph_indptr = indptr.int()
        
        # Create uniform edge weights
        num_edges = indices.size(0)
        self.graph_values = torch.ones(num_edges, device=indices.device, dtype=torch.float32)
        
        # Compute degrees (undirected: in_deg = out_deg)
        self.degrees = graph.in_degrees().float().to(indices.device)
        self.degrees = torch.clamp(self.degrees, min=1.0)
        
        # Load MaxK metadata
        if MAXK_KERNELS_AVAILABLE and graph_name and self.maxk_wrapper:
            self.graph_data_set = self.maxk_wrapper.load_metadata(graph_name)
            if self.graph_data_set:
                print(f"   ✅ GCN metadata loaded: {self.maxk_wrapper.num_warps} warps")
            else:
                print(f"⚠️ MaxK metadata failed, using DGL fallback for {graph_name}")
        else:
            self.graph_data_set = False
    
    def forward(self, graph, feat, topk_values=None, topk_indices=None, weight=None, edge_weight=None):
        """
        Forward pass with MaxK acceleration for GraphConv (GCN)
        """
        with graph.local_scope():
            # Zero in-degree check
            if not self._allow_zero_in_degree:
                if (graph.in_degrees() == 0).any():
                    raise ValueError(
                        "There are 0-in-degree nodes in the graph, "
                        "output for those nodes will be invalid. "
                        "Adding self-loop on the input graph by "
                        "calling `g = dgl.add_self_loop(g)` will resolve "
                        "the issue."
                    )
            
            # Setup aggregation function
            aggregate_fn = fn.copy_u("h", "m")
            if edge_weight is not None:
                assert edge_weight.shape[0] == graph.num_edges()
                graph.edata["_edge_weight"] = edge_weight
                aggregate_fn = fn.u_mul_e("h", "_edge_weight", "m")
            
            # Feature processing
            feat_src, feat_dst = expand_as_pair(feat, graph)
            
            # Left normalization
            if self._norm in ["left", "both"]:
                degs = graph.out_degrees().to(feat_src).clamp(min=1)
                if self._norm == "both":
                    norm = torch.pow(degs, -0.5)
                else:
                    norm = 1.0 / degs
                shp = norm.shape + (1,) * (feat_src.dim() - 1)
                norm = torch.reshape(norm, shp)
                feat_src = feat_src * norm
            
            # Weight handling
            if weight is not None:
                if self.weight is not None:
                    raise ValueError("External weight provided while module has defined weight parameter")
            else:
                weight = self.weight
            
            # Core computation with MaxK acceleration
            lin_before_mp = self._in_feats > self._out_feats
            
            if (self.graph_data_set and self.maxk_wrapper and 
                edge_weight is None and topk_values is not None and topk_indices is not None):
                
                try:
                    # MaxK-accelerated computation
                    if lin_before_mp:
                        # Transform BEFORE aggregation
                        if weight is not None:
                            feat_to_aggregate = torch.matmul(topk_values, weight)
                        else:
                            feat_to_aggregate = topk_values
                        
                        rst = self.maxk_wrapper.spmm(
                            self.graph_indices, self.graph_values,
                            feat_to_aggregate, topk_indices,
                            self.graph_indptr, self.degrees
                        )
                    else:
                        # Aggregate THEN transform
                        rst = self.maxk_wrapper.spmm(
                            self.graph_indices, self.graph_values,
                            topk_values, topk_indices,
                            self.graph_indptr, self.degrees
                        )
                        
                        if weight is not None:
                            rst = torch.matmul(rst, weight)
                    
                except Exception as e:
                    print(f"⚠️ MaxK kernel failed: {e}, falling back to DGL")
                    # Fall back to DGL implementation
                    if lin_before_mp:
                        if weight is not None:
                            feat_src = torch.matmul(feat_src, weight)
                        graph.srcdata["h"] = feat_src
                        graph.update_all(aggregate_fn, fn.sum(msg="m", out="h"))
                        rst = graph.dstdata["h"]
                    else:
                        graph.srcdata["h"] = feat_src
                        graph.update_all(aggregate_fn, fn.sum(msg="m", out="h"))
                        rst = graph.dstdata["h"]
                        if weight is not None:
                            rst = torch.matmul(rst, weight)
            else:
                # Standard DGL computation
                if lin_before_mp:
                    if weight is not None:
                        feat_src = torch.matmul(feat_src, weight)
                    graph.srcdata["h"] = feat_src
                    graph.update_all(aggregate_fn, fn.sum(msg="m", out="h"))
                    rst = graph.dstdata["h"]
                else:
                    graph.srcdata["h"] = feat_src
                    graph.update_all(aggregate_fn, fn.sum(msg="m", out="h"))
                    rst = graph.dstdata["h"]
                    if weight is not None:
                        rst = torch.matmul(rst, weight)
            
            # Right normalization
            if self._norm in ["right", "both"]:
                degs = graph.in_degrees().to(feat_dst).clamp(min=1)
                if self._norm == "both":
                    norm = torch.pow(degs, -0.5)
                else:
                    norm = 1.0 / degs
                shp = norm.shape + (1,) * (feat_dst.dim() - 1)
                norm = torch.reshape(norm, shp)
                rst = rst * norm
            
            # Bias and activation
            if self.bias is not None:
                rst = rst + self.bias
            
            if self._activation is not None:
                rst = self._activation(rst)
            
            return rst

class MaxKGINConv(nn.Module):
    """
    MaxK-accelerated GINConv for undirected graphs
    """
    
    def __init__(self, apply_func=None, aggregator_type="sum", init_eps=0, 
                 learn_eps=False, activation=None, k_value=32):
        super(MaxKGINConv, self).__init__()
        
        self.apply_func = apply_func
        self._aggregator_type = aggregator_type
        self.activation = activation
        self.k_value = k_value
        
        # Validate aggregator type
        if aggregator_type not in ("sum", "max", "mean"):
            raise KeyError(f"Aggregator type {aggregator_type} not recognized.")
        
        # Epsilon parameter
        if learn_eps:
            self.eps = torch.nn.Parameter(torch.FloatTensor([init_eps]))
        else:
            self.register_buffer("eps", torch.FloatTensor([init_eps]))
        
        # MaxK SpGEMM wrapper
        if MAXK_KERNELS_AVAILABLE:
            self.maxk_wrapper = MaxKSpmmWrapper()
        else:
            self.maxk_wrapper = None
        
        self.graph_data_set = False
    
    def set_graph_data(self, graph, graph_name=""):
        """Set graph data for undirected graphs"""
        if self.graph_data_set:
            return
            
        print(f"🔧 Setting GIN graph data for {graph_name}")
        
        graph = graph.local_var()
        
        # Get CSR representation
        indptr, indices, _ = graph.adj_tensors('csr')
        
        # Store graph data
        self.graph_indices = indices.int()
        self.graph_indptr = indptr.int()
        
        # Create uniform edge weights
        num_edges = indices.size(0)
        self.graph_values = torch.ones(num_edges, device=indices.device, dtype=torch.float32)
        
        # Compute degrees
        self.degrees = graph.in_degrees().float().to(indices.device)
        self.degrees = torch.clamp(self.degrees, min=1.0)
        
        # Load MaxK metadata (only for sum aggregation)
        if (MAXK_KERNELS_AVAILABLE and graph_name and self.maxk_wrapper and 
            self._aggregator_type == "sum"):
            self.graph_data_set = self.maxk_wrapper.load_metadata(graph_name)
            if self.graph_data_set:
                print(f"   ✅ GIN metadata loaded: {self.maxk_wrapper.num_warps} warps")
            else:
                print(f"⚠️ MaxK metadata failed, using DGL fallback for {graph_name}")
        else:
            self.graph_data_set = False
    
    def forward(self, graph, feat, topk_values=None, topk_indices=None, edge_weight=None):
        """
        Forward pass with MaxK acceleration for GINConv
        """
        _reducer = getattr(fn, self._aggregator_type)
        
        with graph.local_scope():
            # Setup aggregation function
            aggregate_fn = fn.copy_u("h", "m")
            if edge_weight is not None:
                assert edge_weight.shape[0] == graph.num_edges()
                graph.edata["_edge_weight"] = edge_weight
                aggregate_fn = fn.u_mul_e("h", "_edge_weight", "m")
            
            # Feature processing
            feat_src, feat_dst = expand_as_pair(feat, graph)
            
            # Neighbor aggregation with MaxK acceleration
            if (self.graph_data_set and self.maxk_wrapper and 
                self._aggregator_type == "sum" and edge_weight is None and
                topk_values is not None and topk_indices is not None):
                
                try:
                    # MaxK-accelerated sum aggregation
                    neigh = self.maxk_wrapper.spmm(
                        self.graph_indices, self.graph_values,
                        topk_values, topk_indices,
                        self.graph_indptr, self.degrees
                    )
                    
                except Exception as e:
                    print(f"⚠️ MaxK kernel failed: {e}, falling back to DGL")
                    # Fall back to DGL implementation
                    graph.srcdata["h"] = feat_src
                    graph.update_all(aggregate_fn, _reducer("m", "neigh"))
                    neigh = graph.dstdata["neigh"]
            else:
                # Standard DGL aggregation
                graph.srcdata["h"] = feat_src
                graph.update_all(aggregate_fn, _reducer("m", "neigh"))
                neigh = graph.dstdata["neigh"]
            
            # GIN formula: (1 + eps) * h + aggregate(neighbors)
            rst = (1 + self.eps) * feat_dst + neigh
            
            # Apply function
            if self.apply_func is not None:
                rst = self.apply_func(rst)
            
            # Activation
            if self.activation is not None:
                rst = self.activation(rst)
            
            return rst

class MaxKSAGE(nn.Module):
    """Complete SAGE model using optimized SpGEMM for undirected graphs"""
    
    def __init__(self, in_size, hid_size, num_hid_layers, out_size, maxk=32, 
                 feat_drop=0.5, norm=False, nonlinear="maxk", graph_name=""):
        super().__init__()
        
        assert graph_name, "graph_name REQUIRED"
        assert nonlinear == "maxk", f"Only 'maxk' supported, got {nonlinear}"
        
        self.num_layers = num_hid_layers
        self.graph_name = graph_name
        self.nonlinear = nonlinear
        self.k_value = maxk
        
        # Build layers
        self.layers = nn.ModuleList()
        for i in range(self.num_layers):
            if norm:
                norm_layer = nn.LayerNorm(hid_size, elementwise_affine=True)
            else:
                norm_layer = None
            
            layer = MaxKSAGEConv(
                in_feats=hid_size, out_feats=hid_size,
                aggregator_type='mean', feat_drop=feat_drop,
                norm=norm_layer, k_value=maxk
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
            print(f"🔧 Configuring {self.num_layers} SAGE layers for {self.graph_name}")
            for i, layer in enumerate(self.layers):
                layer.set_graph_data(graph, self.graph_name)
            self.graph_set = True
            print(f"✅ All SAGE layers configured")
    
    def forward(self, g, x):
        """Forward pass using optimized SpGEMM"""
        if not self.graph_set:
            self.set_graph(g)
        
        # Input transformation
        x = self.lin_in(x)
        
        # Hidden layers with optimization
        for i in range(self.num_layers):
            # Apply MaxK and get TopK info
            x_sparse, topk_values, topk_indices = OPTMaxK.apply(x, self.k_value)
            
            # Pass TopK info to layer
            x = self.layers[i](g, x_sparse, topk_values, topk_indices)
        
        # Output transformation
        x = self.lin_out(x)
        
        return x

class MaxKGCN(nn.Module):
    """Complete GCN model with MaxK SpGEMM acceleration for undirected graphs"""
    
    def __init__(self, in_size, hid_size, num_hid_layers, out_size, maxk=32, 
                 feat_drop=0.5, norm=False, nonlinear="maxk", graph_name=""):
        super().__init__()
        self.dropoutlayers = nn.ModuleList()
        self.gcnlayers = nn.ModuleList()
        self.num_layers = num_hid_layers
        self.norm_flag = norm
        self.graph_name = graph_name
        self.k_value = maxk
        self.nonlinear = nonlinear
        
        # Normalization layers
        self.normlayers = nn.ModuleList()
        for i in range(self.num_layers):
            self.dropoutlayers.append(nn.Dropout(feat_drop))
            
            # Use our MaxK-accelerated GraphConv
            layer = MaxKGraphConv(
                in_feats=hid_size, out_feats=hid_size,
                norm="both", weight=False, bias=False,
                k_value=maxk
            )
            self.gcnlayers.append(layer)
            
            if self.norm_flag:
                self.normlayers.append(nn.LayerNorm(hid_size, elementwise_affine=True))
        
        # Linear layers
        self.linlayers = nn.ModuleList()
        for i in range(self.num_layers):
            self.linlayers.append(Linear(hid_size, hid_size))
            init.xavier_uniform_(self.linlayers[i].weight)
        
        # Input and output layers
        self.lin_in = Linear(in_size, hid_size)
        self.lin_out = Linear(hid_size, out_size)
        init.xavier_uniform_(self.lin_in.weight)
        init.xavier_uniform_(self.lin_out.weight)
        
        self.graph_set = False
    
    def set_graph(self, graph):
        """Set graph data for all MaxK layers"""
        if not self.graph_set:
            print(f"🔧 Configuring {self.num_layers} GCN layers for {self.graph_name}")
            for layer in self.gcnlayers:
                if isinstance(layer, MaxKGraphConv):
                    layer.set_graph_data(graph, self.graph_name)
            self.graph_set = True
            print(f"✅ All GCN layers configured")
    
    def forward(self, g, x):
        """Forward pass with MaxK acceleration"""
        if not self.graph_set:
            self.set_graph(g)
        
        x = self.lin_in(x).relu()
        
        for i in range(self.num_layers):
            x = self.linlayers[i](x)
            
            # Apply MaxK activation and get TopK info
            if self.nonlinear == 'maxk':
                x_sparse, topk_values, topk_indices = OPTMaxK.apply(x, self.k_value)
            elif self.nonlinear == 'relu':
                x_sparse = F.relu(x)
                topk_values, topk_indices = None, None
            
            x_sparse = self.dropoutlayers[i](x_sparse)
            
            # Pass TopK info to GCN layer
            x = self.gcnlayers[i](g, x_sparse, topk_values, topk_indices)
            
            if self.norm_flag:
                x = self.normlayers[i](x)
        
        x = self.lin_out(x)
        return x

class MaxKGIN(nn.Module):
    """Complete GIN model with MaxK SpGEMM acceleration for undirected graphs"""
    
    def __init__(self, in_size, hid_size, num_hid_layers, out_size, maxk=32, 
                 feat_drop=0.5, norm=False, nonlinear="maxk", graph_name=""):
        super().__init__()
        self.dropoutlayers = nn.ModuleList()
        self.ginlayers = nn.ModuleList()
        self.num_layers = num_hid_layers
        self.norm_flag = norm
        self.graph_name = graph_name
        self.k_value = maxk
        self.nonlinear = nonlinear
        
        # Normalization layers
        self.normlayers = nn.ModuleList()
        for i in range(self.num_layers):
            self.dropoutlayers.append(nn.Dropout(feat_drop))
            
            # Use our MaxK-accelerated GINConv
            layer = MaxKGINConv(
                apply_func=None, aggregator_type="sum",
                init_eps=0, learn_eps=True, activation=None,
                k_value=maxk
            )
            self.ginlayers.append(layer)
            
            if self.norm_flag:
                self.normlayers.append(nn.LayerNorm(hid_size, elementwise_affine=True))
        
        # Linear layers
        self.linlayers = nn.ModuleList()
        for i in range(self.num_layers):
            self.linlayers.append(Linear(hid_size, hid_size))
            init.xavier_uniform_(self.linlayers[i].weight)
        
        # Input and output layers
        self.lin_in = Linear(in_size, hid_size)
        self.lin_out = Linear(hid_size, out_size)
        init.xavier_uniform_(self.lin_in.weight)
        init.xavier_uniform_(self.lin_out.weight)
        
        self.graph_set = False
    
    def set_graph(self, graph):
        """Set graph data for all MaxK layers"""
        if not self.graph_set:
            print(f"🔧 Configuring {self.num_layers} GIN layers for {self.graph_name}")
            for layer in self.ginlayers:
                if isinstance(layer, MaxKGINConv):
                    layer.set_graph_data(graph, self.graph_name)
            self.graph_set = True
            print(f"✅ All GIN layers configured")
    
    def forward(self, g, x):
        """Forward pass with MaxK acceleration"""
        if not self.graph_set:
            self.set_graph(g)
        
        x = self.lin_in(x).relu()
        
        for i in range(self.num_layers):
            x = self.linlayers[i](x)
            
            # Apply MaxK activation and get TopK info
            if self.nonlinear == 'maxk':
                x_sparse, topk_values, topk_indices = OPTMaxK.apply(x, self.k_value)
            elif self.nonlinear == 'relu':
                x_sparse = F.relu(x)
                topk_values, topk_indices = None, None
            
            x_sparse = self.dropoutlayers[i](x_sparse)
            
            # Pass TopK info to GIN layer
            x = self.ginlayers[i](g, x_sparse, topk_values, topk_indices)
            
            if self.norm_flag:
                x = self.normlayers[i](x)
        
        x = self.lin_out(x)
        return x

# Keep original models for compatibility
class SAGE(nn.Module):
    """Original SAGE model (unchanged for compatibility)"""
    def __init__(self, in_size, hid_size, num_hid_layers, out_size, maxk=32, feat_drop=0.5, norm=False, nonlinear="maxk"):
        super().__init__()
        self.layers = nn.ModuleList()
        self.num_layers = num_hid_layers
        self.maxk = maxk
        
        for i in range(self.num_layers):
            if norm:
                norm_layer = nn.LayerNorm(hid_size, elementwise_affine=True)
            else:
                norm_layer = None
            self.layers.append(dglnn.SAGEConv(hid_size, hid_size, "mean", feat_drop=feat_drop, norm=norm_layer))
        
        self.lin_in = Linear(in_size, hid_size)
        self.lin_out = Linear(hid_size, out_size)
        init.xavier_uniform_(self.lin_in.weight)
        init.xavier_uniform_(self.lin_out.weight)
        
        self.nonlinear = nonlinear
    
    def forward(self, g, x):
        x = self.lin_in(x)
        
        for i in range(self.num_layers):
            if self.nonlinear == 'maxk':
                x = MaxK.apply(x, self.maxk)
            elif self.nonlinear == 'relu':
                x = F.relu(x)
            x = self.layers[i](g, x)
        
        x = self.lin_out(x)
        return x

class GCN(nn.Module):
    """Original GCN model (unchanged for compatibility)"""
    def __init__(self, in_size, hid_size, num_hid_layers, out_size, maxk=32, feat_drop=0.5, norm=False, nonlinear="maxk"):
        super().__init__()
        self.dropoutlayers = nn.ModuleList()
        self.gcnlayers = nn.ModuleList()
        self.maxk = maxk
        self.num_layers = num_hid_layers
        self.norm = norm
        self.normlayers = nn.ModuleList()
        
        for i in range(self.num_layers):
            self.dropoutlayers.append(nn.Dropout(feat_drop))
            self.gcnlayers.append(dglnn.GraphConv(hid_size, hid_size, activation=None, weight=None))
            if self.norm:
                self.normlayers.append(nn.LayerNorm(hid_size, elementwise_affine=True))
        
        self.linlayers = nn.ModuleList()
        for i in range(self.num_layers):
            self.linlayers.append(Linear(hid_size, hid_size))
            init.xavier_uniform_(self.linlayers[i].weight)
        
        self.lin_in = Linear(in_size, hid_size)
        self.lin_out = Linear(hid_size, out_size)
        init.xavier_uniform_(self.lin_in.weight)
        init.xavier_uniform_(self.lin_out.weight)
        
        self.nonlinear = nonlinear
    
    def forward(self, g, x):
        x = self.lin_in(x).relu()
        
        for i in range(self.num_layers):
            x = self.linlayers[i](x)
            if self.nonlinear == 'maxk':
                x = MaxK.apply(x, self.maxk)
            elif self.nonlinear == 'relu':
                x = F.relu(x)
            x = self.dropoutlayers[i](x)
            x = self.gcnlayers[i](g, x)
            if self.norm:
                x = self.normlayers[i](x)
        
        x = self.lin_out(x)
        return x

class GIN(nn.Module):
    """Original GIN model (unchanged for compatibility)"""
    def __init__(self, in_size, hid_size, num_hid_layers, out_size, maxk=32, feat_drop=0.5, norm=False, nonlinear="maxk"):
        super().__init__()
        self.dropoutlayers = nn.ModuleList()
        self.gcnlayers = nn.ModuleList()
        self.maxk = maxk
        self.num_layers = num_hid_layers
        self.norm = norm
        self.normlayers = nn.ModuleList()
        
        for i in range(self.num_layers):
            self.dropoutlayers.append(nn.Dropout(feat_drop))
            self.gcnlayers.append(dglnn.pytorch.conv.GINConv(learn_eps=True, activation=None))
            if self.norm:
                self.normlayers.append(nn.LayerNorm(hid_size, elementwise_affine=True))
        
        self.linlayers = nn.ModuleList()
        for i in range(self.num_layers):
            self.linlayers.append(Linear(hid_size, hid_size))
            init.xavier_uniform_(self.linlayers[i].weight)
        
        self.lin_in = Linear(in_size, hid_size)
        self.lin_out = Linear(hid_size, out_size)
        init.xavier_uniform_(self.lin_in.weight)
        init.xavier_uniform_(self.lin_out.weight)
        
        self.nonlinear = nonlinear
    
    def forward(self, g, x):
        x = self.lin_in(x).relu()
        
        for i in range(self.num_layers):
            x = self.linlayers[i](x)
            if self.nonlinear == 'maxk':
                x = MaxK.apply(x, self.maxk)
            elif self.nonlinear == 'relu':
                x = F.relu(x)
            x = self.dropoutlayers[i](x)
            x = self.gcnlayers[i](g, x)
            if self.norm:
                x = self.normlayers[i](x)
        
        x = self.lin_out(x)
        return x

class GNN_res(nn.Module):
    """Original GNN_res model (unchanged for compatibility)"""
    def __init__(self, in_size, hid_size, num_hid_layers, out_size, maxk=32, feat_drop=0.5, norm=False, nonlinear="maxk"):
        super().__init__()
        self.dropoutlayers1 = nn.ModuleList()
        self.dropoutlayers2 = nn.ModuleList()
        self.gcnlayers = nn.ModuleList()
        self.num_layers = num_hid_layers
        self.norm = norm
        self.normlayers = nn.ModuleList()
        
        for i in range(self.num_layers):
            self.dropoutlayers1.append(nn.Dropout(feat_drop))
            self.dropoutlayers2.append(nn.Dropout(feat_drop))
            self.gcnlayers.append(dglnn.GraphConv(hid_size, hid_size, activation=None, weight=None))
            if self.norm:
                self.normlayers.append(nn.BatchNorm1d(hid_size))
        
        self.linlayers1 = nn.ModuleList()
        self.linlayers2 = nn.ModuleList()
        self.reslayers = nn.ModuleList()
        for i in range(self.num_layers):
            self.linlayers1.append(Linear(hid_size, hid_size))
            self.linlayers2.append(Linear(hid_size, hid_size))
            self.reslayers.append(Linear(hid_size, hid_size))
            init.xavier_uniform_(self.linlayers1[i].weight)
            init.xavier_uniform_(self.linlayers2[i].weight)
            init.xavier_uniform_(self.reslayers[i].weight)
        
        self.lin_in = Linear(in_size, hid_size)
        self.lin_out = Linear(hid_size, out_size)
        init.xavier_uniform_(self.lin_in.weight)
        init.xavier_uniform_(self.lin_out.weight)
    
    def forward(self, g, x):
        x = self.lin_in(x).relu()
        
        for i in range(self.num_layers):
            x_res = self.reslayers[i](x)
            x = self.gcnlayers[i](g, x)
            if self.norm:
                x = self.normlayers[i](x)
            
            x = self.linlayers1[i](x)
            x = F.relu(x)
            x = self.dropoutlayers1[i](x)
            x = self.linlayers2[i](x)
            
            x = x_res + x
            x = F.relu(x)
            x = self.dropoutlayers2[i](x)
        
        x = self.lin_out(x)
        return x

def test_integrated_models():
    """Test all integrated MaxK models"""
    print("🧪 Testing Integrated MaxK Models")
    print("=" * 40)
    
    if not torch.cuda.is_available():
        print("❌ CUDA not available")
        return
    
    # Create test graph and features
    num_nodes = 1000
    num_edges = 5000
    feat_dim = 128
    hidden_dim = 64
    output_dim = 10
    
    # Create random graph
    src = torch.randint(0, num_nodes, (num_edges,))
    dst = torch.randint(0, num_nodes, (num_edges,))
    g = dgl.graph((src, dst), num_nodes=num_nodes).to('cuda')
    
    # Create features
    features = torch.randn(num_nodes, feat_dim).cuda()
    
    print(f"📊 Test graph: {num_nodes} nodes, {num_edges} edges")
    
    # Test different models
    models = {
        "Original SAGE": SAGE(feat_dim, hidden_dim, 2, output_dim, maxk=32),
        "Original GCN": GCN(feat_dim, hidden_dim, 2, output_dim, maxk=32),
        "Original GIN": GIN(feat_dim, hidden_dim, 2, output_dim, maxk=32),
    }
    
    if MAXK_KERNELS_AVAILABLE:
        models.update({
            "MaxK SAGE": MaxKSAGE(feat_dim, hidden_dim, 2, output_dim, maxk=32, graph_name="test"),
            "MaxK GCN": MaxKGCN(feat_dim, hidden_dim, 2, output_dim, maxk=32, graph_name="test"),
            "MaxK GIN": MaxKGIN(feat_dim, hidden_dim, 2, output_dim, maxk=32, graph_name="test"),
        })
    
    results = {}
    
    for name, model in models.items():
        print(f"\n🔄 Testing {name}...")
        model = model.cuda()
        
        try:
            # Forward pass
            output = model(g, features)
            
            # Check output
            output_min, output_max = output.min().item(), output.max().item()
            output_mean = output.mean().item()
            
            print(f"✅ {name} forward pass successful")
            print(f"   Output shape: {output.shape}")
            print(f"   Output range: [{output_min:.4f}, {output_max:.4f}]")
            
            # Backward pass
            loss = output.sum()
            loss.backward()
            print(f"✅ {name} backward pass successful")
            
            results[name] = {
                'output_range': (output_min, output_max),
                'output_mean': output_mean,
                'success': True
            }
            
        except Exception as e:
            print(f"❌ {name} failed: {e}")
            results[name] = {'success': False, 'error': str(e)}
    
    print(f"\n📊 Test Summary:")
    print("=" * 40)
    for name, result in results.items():
        if result['success']:
            print(f"✅ {name}: PASSED")
        else:
            print(f"❌ {name}: FAILED - {result['error']}")

if __name__ == "__main__":
    test_integrated_models()
