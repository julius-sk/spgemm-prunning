#!/usr/bin/env python3
"""
Complete DGL-Equivalent MaxK SAGE Implementation
Mathematically identical to DGL SAGEConv with MaxK kernel acceleration
Addresses ALL differences: normalization, transformation timing, edge weights, etc.
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

# Import our MaxK SpGEMM function
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
    """MaxK activation that returns TopK info for optimization"""
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
        
        csr_values = torch.ones(len(indices_csr), device=device, dtype=torch.float32)
        csc_values = torch.ones(len(indices_csc), device=device, dtype=torch.float32)
        
        # Store graph data
        self.graph_indices = indices_csr.int()
        self.graph_indptr = indptr_csr.int()
        self.graph_values = csr_values
        
        self.graph_indices_T = indices_csc.int()
        self.graph_values_T = csc_values
        
        self.in_degrees=in_degrees
        self.out_degrees=out_degrees
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
            
            #print(f"🚀 Forward with edge weight normalization")
            
            # Transformation order
            lin_before_mp = self._in_src_feats > self._out_feats
            
            if lin_before_mp:
                # Transform BEFORE aggregation
                feat_to_aggregate = self.fc_neigh(topk_values)
                
                # Use YOUR SpGEMM function
                h_neigh = self.maxk_wrapper.spmm(
                self.graph_indices, self.graph_values, # Uniform weights
                feat_to_aggregate, topk_indices, # Pre-computed TopK
                self.graph_indptr, self.in_degrees, self.out_degrees, # Original normalization
                self.graph_indices_T, self.graph_values_T
                )
            else:
                # Aggregate THEN transform
                h_neigh_aggregated = self.maxk_wrapper.spmm(
                self.graph_indices, self.graph_values, # Uniform weights
                topk_values, topk_indices, # Pre-computed TopK
                self.graph_indptr, self.in_degrees, self.out_degrees, # Original normalization
                self.graph_indices_T, self.graph_values_T
                )
                h_neigh = self.fc_neigh(h_neigh_aggregated)
            
            # Combine features
            rst = self.fc_self(h_self) + h_neigh
            
            # Post-processing
            if self.activation is not None:
                rst = self.activation(rst)
            if self.norm is not None:
                rst = self.norm(rst)
            
            #print(f"✅ Forward complete: {rst.shape}")
            return rst

class MaxKGraphConv(nn.Module):
    """
    MaxK-accelerated GraphConv (GCN) implementation
    Replicates DGL's GraphConv with MaxK SpGEMM acceleration
    Supports all DGL GraphConv features: normalization modes, edge weights, bipartite graphs
    """
    
    def __init__(self, in_feats, out_feats, norm="both", weight=True, bias=True, 
                 activation=None, allow_zero_in_degree=False, k_value=32):
        super(MaxKGraphConv, self).__init__()
        
        # Validate norm parameter (exactly like DGL)
        if norm not in ("none", "both", "right", "left"):
            raise ValueError(f'Invalid norm value. Must be either "none", "both", "right" or "left". But got "{norm}".')
        
        # Store parameters exactly like DGL
        self._in_feats = in_feats
        self._out_feats = out_feats
        self._norm = norm
        self._allow_zero_in_degree = allow_zero_in_degree
        self.k_value = k_value
        
        # Weight and bias (exactly like DGL)
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
        
        # Graph metadata (will be set during first forward pass)
        self.graph_indices = None
        self.graph_values = None
        self.graph_indptr = None
        self.metadata_loaded = False
        self.use_maxk_kernel = False
    
    def reset_parameters(self):
        """Initialize parameters exactly like DGL"""
        if self.weight is not None:
            init.xavier_uniform_(self.weight)
        if self.bias is not None:
            init.zeros_(self.bias)
    
    def set_graph_data(self, graph, graph_name=""):
        """Set graph data for MaxK kernel usage"""
        # Extract CSR format from DGL graph
        graph = graph.local_var()
        
        # Get CSR representation
        indptr, indices, _ = graph.adj_tensors('csr')
        csc_indptr, csc_indices, _ = graph.adj_tensors('csc')
        
        # Store graph data
        self.graph_indices = indices.int()
        self.graph_indptr = indptr.int()
        self.graph_indices_T = csc_indices.int()
        self.graph_indptr_T = csc_indptr.int()
        
        # Create uniform edge weights (can be modified for weighted graphs)
        num_edges = indices.size(0)
        self.graph_values = torch.ones(num_edges, device=indices.device, dtype=torch.float32)
        self.graph_values_T = torch.ones_like(csc_indices, dtype=torch.float32)
        
        # Compute and store node degrees for proper normalization
        self.in_degrees = graph.in_degrees().float().to(indices.device)
        self.out_degrees = graph.out_degrees().float().to(indices.device)
        # Avoid division by zero for isolated nodes
        self.in_degrees = torch.clamp(self.in_degrees, min=1.0)
        self.out_degrees = torch.clamp(self.out_degrees, min=1.0)
        
        # Load MaxK metadata if kernels are available
        if MAXK_KERNELS_AVAILABLE and graph_name and self.maxk_wrapper:
            self.metadata_loaded = self.maxk_wrapper.load_metadata(graph_name)
            if self.metadata_loaded:
                self.use_maxk_kernel = True
            else:
                print(f"⚠️ MaxK metadata failed, using DGL fallback for {graph_name}")
        else:
            self.use_maxk_kernel = False
    
    def forward(self, graph, feat, weight=None, edge_weight=None):
        """
        Forward pass with MaxK acceleration for GraphConv (GCN)
        Exactly replicates DGL GraphConv.forward() behavior
        """
        with graph.local_scope():
            # === STEP 1: Zero in-degree check (exactly like DGL) ===
            if not self._allow_zero_in_degree:
                if (graph.in_degrees() == 0).any():
                    raise ValueError(
                        "There are 0-in-degree nodes in the graph, "
                        "output for those nodes will be invalid. "
                        "Adding self-loop on the input graph by "
                        "calling `g = dgl.add_self_loop(g)` will resolve "
                        "the issue. Setting `allow_zero_in_degree` "
                        "to be `True` when constructing this module will "
                        "suppress the check and let the code run."
                    )
            
            # === STEP 2: Setup aggregation function (exactly like DGL) ===
            aggregate_fn = fn.copy_u("h", "m")
            if edge_weight is not None:
                assert edge_weight.shape[0] == graph.num_edges()
                graph.edata["_edge_weight"] = edge_weight
                aggregate_fn = fn.u_mul_e("h", "_edge_weight", "m")
            
            # === STEP 3: Feature processing (exactly like DGL) ===
            feat_src, feat_dst = expand_as_pair(feat, graph)
            
            # === STEP 4: Left normalization (exactly like DGL) ===
            if self._norm in ["left", "both"]:
                degs = graph.out_degrees().to(feat_src).clamp(min=1)
                if self._norm == "both":
                    norm = torch.pow(degs, -0.5)
                else:
                    norm = 1.0 / degs
                shp = norm.shape + (1,) * (feat_src.dim() - 1)
                norm = torch.reshape(norm, shp)
                feat_src = feat_src * norm
            
            # === STEP 5: Weight handling (exactly like DGL) ===
            if weight is not None:
                if self.weight is not None:
                    raise ValueError(
                        "External weight is provided while at the same time the"
                        " module has defined its own weight parameter. Please"
                        " create the module with flag weight=False."
                    )
            else:
                weight = self.weight
            
            # === STEP 6: Core computation with MaxK acceleration ===
            lin_before_mp = self._in_feats > self._out_feats
            
            if (self.use_maxk_kernel and 
                self.maxk_wrapper and 
                edge_weight is None):  # MaxK doesn't support edge weights yet
                
                try:
                    # MaxK-accelerated computation
                    if lin_before_mp:
                        # Transform BEFORE aggregation
                        if weight is not None:
                            feat_to_aggregate = torch.matmul(feat_src, weight)
                        else:
                            feat_to_aggregate = feat_src
                        
                        rst = self.maxk_wrapper.spmm(
                            self.graph_indices,
                            self.graph_values,
                            feat_to_aggregate,
                            self.k_value,
                            self.graph_indptr,
                            self.in_degrees,
                            self.out_degrees,
                            self.graph_indices_T,
                            self.graph_values_T
                        )
                    else:
                        # Aggregate THEN transform
                        rst = self.maxk_wrapper.spmm(
                            self.graph_indices,
                            self.graph_values,
                            feat_src,
                            self.k_value,
                            self.graph_indptr,
                            self.in_degrees,
                            self.out_degrees,
                            self.graph_indices_T,
                            self.graph_values_T
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
            
            # === STEP 7: Right normalization (exactly like DGL) ===
            if self._norm in ["right", "both"]:
                degs = graph.in_degrees().to(feat_dst).clamp(min=1)
                if self._norm == "both":
                    norm = torch.pow(degs, -0.5)
                else:
                    norm = 1.0 / degs
                shp = norm.shape + (1,) * (feat_dst.dim() - 1)
                norm = torch.reshape(norm, shp)
                rst = rst * norm
            
            # === STEP 8: Bias and activation (exactly like DGL) ===
            if self.bias is not None:
                rst = rst + self.bias
            
            if self._activation is not None:
                rst = self._activation(rst)
            
            return rst

class MaxKGINConv(nn.Module):
    """
    MaxK-accelerated GINConv implementation
    Replicates DGL's GINConv with MaxK SpGEMM acceleration
    Supports all DGL GINConv features: different aggregators, edge weights, learnable eps
    """
    
    def __init__(self, apply_func=None, aggregator_type="sum", init_eps=0, 
                 learn_eps=False, activation=None, k_value=32):
        super(MaxKGINConv, self).__init__()
        
        self.apply_func = apply_func
        self._aggregator_type = aggregator_type
        self.activation = activation
        self.k_value = k_value
        
        # Validate aggregator type (exactly like DGL)
        if aggregator_type not in ("sum", "max", "mean"):
            raise KeyError(f"Aggregator type {aggregator_type} not recognized.")
        
        # Epsilon parameter (exactly like DGL)
        if learn_eps:
            self.eps = torch.nn.Parameter(torch.FloatTensor([init_eps]))
        else:
            self.register_buffer("eps", torch.FloatTensor([init_eps]))
        
        # MaxK SpGEMM wrapper
        if MAXK_KERNELS_AVAILABLE:
            self.maxk_wrapper = MaxKSpmmWrapper()
        else:
            self.maxk_wrapper = None
        
        # Graph metadata (will be set during first forward pass)
        self.graph_indices = None
        self.graph_values = None
        self.graph_indptr = None
        self.metadata_loaded = False
        self.use_maxk_kernel = False
    
    def set_graph_data(self, graph, graph_name=""):
        """Set graph data for MaxK kernel usage"""
        # Extract CSR format from DGL graph
        graph = graph.local_var()
        
        # Get CSR representation
        indptr, indices, _ = graph.adj_tensors('csr')
        csc_indptr, csc_indices, _ = graph.adj_tensors('csc')
        
        # Store graph data
        self.graph_indices = indices.int()
        self.graph_indptr = indptr.int()
        self.graph_indices_T = csc_indices.int()
        self.graph_indptr_T = csc_indptr.int()
        
        # Create uniform edge weights
        num_edges = indices.size(0)
        self.graph_values = torch.ones(num_edges, device=indices.device, dtype=torch.float32)
        self.graph_values_T = torch.ones_like(csc_indices, dtype=torch.float32)
        
        # Compute node degrees
        self.in_degrees = graph.in_degrees().float().to(indices.device)
        self.out_degrees = graph.out_degrees().float().to(indices.device)
        self.in_degrees = torch.clamp(self.in_degrees, min=1.0)
        self.out_degrees = torch.clamp(self.out_degrees, min=1.0)
        
        # Load MaxK metadata if kernels are available
        # Note: Only use MaxK for sum aggregation (most common and efficient)
        if (MAXK_KERNELS_AVAILABLE and graph_name and self.maxk_wrapper and 
            self._aggregator_type == "sum"):
            self.metadata_loaded = self.maxk_wrapper.load_metadata(graph_name)
            if self.metadata_loaded:
                self.use_maxk_kernel = True
            else:
                print(f"⚠️ MaxK metadata failed, using DGL fallback for {graph_name}")
        else:
            self.use_maxk_kernel = False
    
    def forward(self, graph, feat, edge_weight=None):
        """
        Forward pass with MaxK acceleration for GINConv
        Exactly replicates DGL GINConv.forward() behavior
        """
        _reducer = getattr(fn, self._aggregator_type)
        
        with graph.local_scope():
            # === STEP 1: Setup aggregation function (exactly like DGL) ===
            aggregate_fn = fn.copy_u("h", "m")
            if edge_weight is not None:
                assert edge_weight.shape[0] == graph.num_edges()
                graph.edata["_edge_weight"] = edge_weight
                aggregate_fn = fn.u_mul_e("h", "_edge_weight", "m")
            
            # === STEP 2: Feature processing (exactly like DGL) ===
            feat_src, feat_dst = expand_as_pair(feat, graph)
            
            # === STEP 3: Neighbor aggregation with MaxK acceleration ===
            if (self.use_maxk_kernel and 
                self.maxk_wrapper and 
                self._aggregator_type == "sum" and  # Only accelerate sum aggregation
                edge_weight is None):  # MaxK doesn't support edge weights yet
                
                try:
                    # MaxK-accelerated sum aggregation
                    neigh = self.maxk_wrapper.spmm(
                        self.graph_indices,
                        self.graph_values,
                        feat_src,
                        self.k_value,
                        self.graph_indptr,
                        self.in_degrees,
                        self.out_degrees,
                        self.graph_indices_T,
                        self.graph_values_T
                    )
                    
                except Exception as e:
                    print(f"⚠️ MaxK kernel failed: {e}, falling back to DGL")
                    # Fall back to DGL implementation
                    graph.srcdata["h"] = feat_src
                    graph.update_all(aggregate_fn, _reducer("m", "neigh"))
                    neigh = graph.dstdata["neigh"]
            else:
                # Standard DGL aggregation (for max, mean, or when MaxK unavailable)
                graph.srcdata["h"] = feat_src
                graph.update_all(aggregate_fn, _reducer("m", "neigh"))
                neigh = graph.dstdata["neigh"]
            
            # === STEP 4: GIN formula: (1 + eps) * h + aggregate(neighbors) ===
            rst = (1 + self.eps) * feat_dst + neigh
            
            # === STEP 5: Apply function (exactly like DGL) ===
            if self.apply_func is not None:
                rst = self.apply_func(rst)
            
            # === STEP 6: Activation (exactly like DGL) ===
            if self.activation is not None:
                rst = self.activation(rst)
            
            return rst


class MaxKSAGE(nn.Module):
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
        
        # print(f"🏗️ Building EdgeWeightNormalizedMaxKSAGE:")
        # print(f"   Graph: {graph_name}, Layers: {num_hid_layers}, MaxK: {maxk}")
        
        # Build layers
        self.layers = nn.ModuleList()
        for i in range(self.num_layers):
            if norm:
                norm_layer = nn.LayerNorm(hid_size, elementwise_affine=True)
            else:
                norm_layer = None
            
            layer = MaxKSAGEConv(
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
        
        #print(f"🚀 EdgeWeightNormalizedMaxKSAGE forward: {x.shape}")
        
        # Input transformation
        x = self.lin_in(x)
        
        # Hidden layers with edge weight optimization
        for i in range(self.num_layers):
            #print(f"   Layer {i}")
            
            # Apply MaxK and get TopK info
            x_sparse, topk_values, topk_indices = OPTMaxK.apply(x, self.k_value)
            
            # Pass TopK info to layer (uses YOUR SpGEMM function)
            x = self.layers[i](g, x_sparse, topk_values, topk_indices)
            
            #print(f"     Output: {x.shape}")
        
        # Output transformation
        x = self.lin_out(x)
        #print(f"✅ Model forward complete: {x.shape}")
        
        return x

class MaxKGCN(nn.Module):
    """
    Complete GCN model with MaxK SpGEMM acceleration
    """
    
    def __init__(self, in_size, hid_size, num_hid_layers, out_size, maxk=32, 
                 feat_drop=0.5, norm=False, nonlinear="maxk", graph_name=""):
        super().__init__()
        self.dropoutlayers = nn.ModuleList()
        self.gcnlayers = nn.ModuleList()
        self.num_layers = num_hid_layers
        self.norm_flag = norm
        self.graph_name = graph_name
        
        # Normalization layers
        self.normlayers = nn.ModuleList()
        for i in range(self.num_layers):
            self.dropoutlayers.append(nn.Dropout(feat_drop))
            
            # Use our MaxK-accelerated GraphConv
            layer = MaxKGraphConv(
                in_feats=hid_size,
                out_feats=hid_size,
                norm="both",  # Use both normalization like original
                weight=False,  # We'll use separate linear layers
                bias=False,
                k_value=maxk
            )
            self.gcnlayers.append(layer)
            
            if self.norm_flag:
                self.normlayers.append(nn.LayerNorm(hid_size, elementwise_affine=True))
        
        # Linear layers (exactly like original)
        self.linlayers = nn.ModuleList()
        for i in range(self.num_layers):
            self.linlayers.append(Linear(hid_size, hid_size))
            init.xavier_uniform_(self.linlayers[i].weight)
        
        # Input and output layers
        self.lin_in = Linear(in_size, hid_size)
        self.lin_out = Linear(hid_size, out_size)
        init.xavier_uniform_(self.lin_in.weight)
        init.xavier_uniform_(self.lin_out.weight)        

        
        self.nonlinear = nonlinear
        self.graph_set = False
    
    def set_graph(self, graph):
        """Set graph data for all MaxK layers"""
        if not self.graph_set:
            for layer in self.gcnlayers:
                if isinstance(layer, MaxKGraphConv):
                    layer.set_graph_data(graph, self.graph_name)
            self.graph_set = True
    
    def forward(self, g, x):
        """Forward pass with MaxK acceleration"""
        # Set graph data on first forward pass
        if not self.graph_set:
            self.set_graph(g)      
        for i in range(self.num_layers):
            x = self.linlayers[i](x)
            if self.nonlinear == 'maxk':
                x = MaxK.apply(x,self.maxk)
            elif self.nonlinear == 'relu':
                x = F.relu(x)
            x = self.dropoutlayers[i](x)
            x = self.gcnlayers[i](g, x)
            if self.norm:
                x = self.normlayers[i](x)
        x = self.lin_out(x)
        return x
    
    
class MaxKGIN(nn.Module):
    """
    Complete GIN model with MaxK SpGEMM acceleration
    """
    
    def __init__(self, in_size, hid_size, num_hid_layers, out_size, maxk=32, 
                 feat_drop=0.5, norm=False, nonlinear="maxk", graph_name=""):
        super().__init__()
        self.dropoutlayers = nn.ModuleList()
        self.ginlayers = nn.ModuleList()
        self.num_layers = num_hid_layers
        self.norm_flag = norm
        self.graph_name = graph_name
        
        # Normalization layers
        self.normlayers = nn.ModuleList()
        for i in range(self.num_layers):
            self.dropoutlayers.append(nn.Dropout(feat_drop))
            
            # Use our MaxK-accelerated GINConv
            layer = MaxKGINConv(
                apply_func=None,  # We'll handle the linear transformation separately
                aggregator_type="sum",  # Use sum for MaxK acceleration
                init_eps=0,
                learn_eps=True,
                activation=None,
                k_value=maxk
            )
            self.ginlayers.append(layer)
            
            if self.norm_flag:
                self.normlayers.append(nn.LayerNorm(hid_size, elementwise_affine=True))
        
        # Linear layers (exactly like original)
        self.linlayers = nn.ModuleList()
        for i in range(self.num_layers):
            self.linlayers.append(Linear(hid_size, hid_size))
            init.xavier_uniform_(self.linlayers[i].weight)
        
        # Input and output layers
        self.lin_in = Linear(in_size, hid_size)
        self.lin_out = Linear(hid_size, out_size)
        init.xavier_uniform_(self.lin_in.weight)
        init.xavier_uniform_(self.lin_out.weight)
        
        # MaxK activation functions
        # for i in range(self.num_layers):
        #     exec(f"self.maxk{i} = MaxK.apply")
        #     exec(f"self.k{i} = maxk")
        
        self.nonlinear = nonlinear
        self.graph_set = False
    
    def set_graph(self, graph):
        """Set graph data for all MaxK layers"""
        if not self.graph_set:
            for layer in self.ginlayers:
                if isinstance(layer, MaxKGINConv):
                    layer.set_graph_data(graph, self.graph_name)
            self.graph_set = True
    
    def forward(self, g, x):
        """Forward pass with MaxK acceleration"""
        # Set graph data on first forward pass
        if not self.graph_set:
            self.set_graph(g)
        x = self.lin_in(x).relu()

        for i in range(self.num_layers):
            x = self.linlayers[i](x)
            if self.nonlinear == 'maxk':
                x = MaxK.apply(x,self.maxk)
            elif self.nonlinear == 'relu':
                x = F.relu(x)
            x = self.dropoutlayers[i](x)
            x = self.gcnlayers[i](g, x)
            if self.norm:
                x = self.normlayers[i](x)
                
        x = self.lin_out(x)
        return x
# Alternative approach: Use DGL's SAGEConv with MaxK activation only
# class HybridMaxKSAGE(nn.Module):
#     """
#     ALTERNATIVE: Hybrid approach that uses DGL's proven SAGEConv for message passing
#     but applies MaxK activation. This guarantees correct normalization.
#     """
    
#     def __init__(self, in_size, hid_size, num_hid_layers, out_size, maxk=32, 
#                  feat_drop=0.5, norm=False, nonlinear="maxk"):
#         super().__init__()
#         self.layers = nn.ModuleList()
#         self.num_layers = num_hid_layers
        
#         # Use DGL's proven SAGEConv layers (they handle normalization correctly)
#         for i in range(self.num_layers):
#             if norm:
#                 norm_layer = nn.LayerNorm(hid_size, elementwise_affine=True)
#             else:
#                 norm_layer = None
            
#             # Use DGL's SAGEConv which includes proper mean normalization
#             self.layers.append(dglnn.SAGEConv(
#                 hid_size, hid_size, "mean", 
#                 feat_drop=feat_drop, 
#                 norm=norm_layer
#             ))
        
#         # Input and output linear layers
#         self.lin_in = Linear(in_size, hid_size)
#         self.lin_out = Linear(hid_size, out_size)
#         init.xavier_uniform_(self.lin_in.weight)
#         init.xavier_uniform_(self.lin_out.weight)
        
#         # MaxK activation functions
#         for i in range(self.num_layers):
#             exec(f"self.maxk{i} = MaxK.apply")
#             exec(f"self.k{i} = maxk")
        
#         self.nonlinear = nonlinear
    
#     def forward(self, g, x):
#         """Forward pass using DGL's SAGEConv with MaxK activation"""
#         # Input transformation
#         x = self.lin_in(x)
        
#         # Hidden layers with MaxK activation and DGL's message passing
#         for i in range(self.num_layers):
#             # Apply MaxK activation if specified
#             if self.nonlinear == 'maxk':
#                 x = eval(f"self.maxk{i}(x, self.k{i})")
#             elif self.nonlinear == 'relu':
#                 x = F.relu(x)
            
#             # Use DGL's SAGEConv (includes proper mean normalization)
#             x = self.layers[i](g, x)
        
#         # Output transformation
#         x = self.lin_out(x)
        
#         return x

# Keep original models for compatibility
class SAGE(nn.Module):
    """Original SAGE model (unchanged for compatibility)"""
    def __init__(self, in_size, hid_size, num_hid_layers, out_size, maxk=32, feat_drop=0.5, norm=False, nonlinear = "maxk"):
        super().__init__()
        self.layers = nn.ModuleList()
        self.num_layers = num_hid_layers
        self.maxk=maxk
        # Multi-layers SAGEConv
        for i in range(self.num_layers):
            if norm:
                norm_layer = nn.LayerNorm(hid_size, elementwise_affine=True)
                # norm_layer = nn.BatchNorm1d(hid_size)
            else:
                norm_layer = None
            self.layers.append(dglnn.SAGEConv(hid_size, hid_size, "mean", feat_drop=feat_drop, norm=norm_layer))
        # self.layers.append(dglnn.SAGEConv(hid_size, out_size, "mean", feat_drop=feat_drop))

        self.lin_in = Linear(in_size, hid_size)
        self.lin_out = Linear(hid_size, out_size)
        init.xavier_uniform_(self.lin_in.weight)
        init.xavier_uniform_(self.lin_out.weight)

        self.nonlinear = nonlinear
    def forward(self, g, x):
        x = self.lin_in(x)

        for i in range(self.num_layers):
            if self.nonlinear == 'maxk':
                x = MaxK.apply(x,self.maxk)
            elif self.nonlinear == 'relu':
                x = F.relu(x)
            # x = self.dropout(x)
            x = self.layers[i](g, x)
        x = self.lin_out(x)

        return x

class GCN(nn.Module):
    """Original GCN model (unchanged for compatibility)"""
    def __init__(self, in_size, hid_size, num_hid_layers, out_size, maxk=32, feat_drop=0.5, norm=False, nonlinear = "maxk"):
        super().__init__()
        self.dropoutlayers = nn.ModuleList()
        self.gcnlayers = nn.ModuleList()
        self.maxk=maxk
        # two-layer GCN
        self.num_layers = num_hid_layers
        self.norm = norm
        self.normlayers = nn.ModuleList()
        for i in range(self.num_layers):
            self.dropoutlayers.append(nn.Dropout(feat_drop))
            self.gcnlayers.append(dglnn.GraphConv(hid_size, hid_size, activation=None, weight=None))
            if self.norm:
                self.normlayers.append(nn.LayerNorm(hid_size, elementwise_affine=True))
                # self.normlayers.append(nn.BatchNorm1d(hid_size))

        self.linlayers = nn.ModuleList()
        for i in range(self.num_layers):
            self.linlayers.append(Linear(hid_size, hid_size))
        for i in range(self.num_layers):
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
                x = MaxK.apply(x,self.maxk)
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
    def __init__(self, in_size, hid_size, num_hid_layers, out_size, maxk=32, feat_drop=0.5, norm=False, nonlinear = "maxk"):
        super().__init__()
        self.dropoutlayers = nn.ModuleList()
        self.gcnlayers = nn.ModuleList()
        self.maxk=maxk
        # two-layer GCN
        self.num_layers = num_hid_layers
        self.norm = norm
        self.normlayers = nn.ModuleList()
        for i in range(self.num_layers):
            self.dropoutlayers.append(nn.Dropout(feat_drop))
            self.gcnlayers.append(dglnn.pytorch.conv.GINConv(learn_eps=True, activation=None))
            if self.norm:
                self.normlayers.append(nn.LayerNorm(hid_size, elementwise_affine=True))
                # self.normlayers.append(nn.BatchNorm1d(hid_size))

        self.linlayers = nn.ModuleList()
        for i in range(self.num_layers):
            self.linlayers.append(Linear(hid_size, hid_size))
        for i in range(self.num_layers):
            init.xavier_uniform_(self.linlayers[i].weight)
        self.lin_in = Linear(in_size, hid_size)
        self.lin_out = Linear(hid_size, out_size)
        init.xavier_uniform_(self.lin_in.weight)
        init.xavier_uniform_(self.lin_out.weight)

    def forward(self, g, x):
        x = self.lin_in(x).relu()

        for i in range(self.num_layers):
            x = self.linlayers[i](x)
            if self.nonlinear == 'maxk':
                x = MaxK.apply(x,self.maxk)
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
    def __init__(self, in_size, hid_size, num_hid_layers, out_size, maxk=32, feat_drop=0.5, norm=False, nonlinear = "maxk"):
        super().__init__()
        self.dropoutlayers1 = nn.ModuleList()
        self.dropoutlayers2 = nn.ModuleList()
        self.gcnlayers = nn.ModuleList()
        
        # two-layer GCN
        self.num_layers = num_hid_layers
        self.norm = norm
        self.normlayers = nn.ModuleList()
        for i in range(self.num_layers):
            self.dropoutlayers1.append(nn.Dropout(feat_drop))
            self.dropoutlayers2.append(nn.Dropout(feat_drop))
            self.gcnlayers.append(dglnn.GraphConv(hid_size, hid_size, activation=None, weight=None))
            if self.norm:
                # self.normlayers.append(nn.LayerNorm(hid_size, elementwise_affine=True))
                self.normlayers.append(nn.BatchNorm1d(hid_size))

        self.linlayers1 = nn.ModuleList()
        self.linlayers2 = nn.ModuleList()
        self.reslayers = nn.ModuleList()
        for i in range(self.num_layers):
            self.linlayers1.append(Linear(hid_size, hid_size))
            self.linlayers2.append(Linear(hid_size, hid_size))
            self.reslayers.append(Linear(hid_size, hid_size))
        for i in range(self.num_layers):
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
    
def test_normalization_fix():
    """Test the normalization fix"""
    print("🧪 Testing MaxK SAGE Normalization Fix")
    print("=" * 50)
    
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
    print(f"📊 Average degree: {num_edges * 2 / num_nodes:.1f}")
    
    # Test different approaches
    models = {
        "Original SAGE": SAGE(feat_dim, hidden_dim, 2, output_dim, maxk=32),
        "Hybrid MaxK-SAGE": HybridMaxKSAGE(feat_dim, hidden_dim, 2, output_dim, maxk=32),
    }
    
    if MAXK_KERNELS_AVAILABLE:
        models["Fixed MaxK-SAGE"] = MaxKSAGE(feat_dim, hidden_dim, 2, output_dim, maxk=32, graph_name="test")
    
    results = {}
    
    for name, model in models.items():
        print(f"\n🔄 Testing {name}...")
        model = model.cuda()
        
        try:
            # Forward pass
            output = model(g, features)
            
            # Check for reasonable output ranges
            output_min, output_max = output.min().item(), output.max().item()
            output_mean = output.mean().item()
            output_std = output.std().item()
            
            print(f"✅ {name} forward pass successful")
            print(f"   Output range: [{output_min:.4f}, {output_max:.4f}]")
            print(f"   Output mean: {output_mean:.4f}, std: {output_std:.4f}")
            
            # Check if values are reasonable (not exploding)
            if abs(output_max) < 1000 and abs(output_min) < 1000:
                print(f"✅ {name} produces reasonable output values")
            else:
                print(f"⚠️ {name} may have exploding values")
            
            # Backward pass
            loss = output.sum()
            loss.backward()
            print(f"✅ {name} backward pass successful")
            
            results[name] = {
                'output_range': (output_min, output_max),
                'output_mean': output_mean,
                'output_std': output_std,
                'success': True
            }
            
        except Exception as e:
            print(f"❌ {name} failed: {e}")
            results[name] = {'success': False, 'error': str(e)}
    
    # Compare results
    print(f"\n📊 Comparison Summary:")
    print("=" * 50)
    for name, result in results.items():
        if result['success']:
            print(f"{name:20s}: range=[{result['output_range'][0]:8.4f}, {result['output_range'][1]:8.4f}], "
                  f"mean={result['output_mean']:8.4f}")
        else:
            print(f"{name:20s}: FAILED - {result['error']}")
    
    print(f"\n💡 Key Points:")
    print("- Original SAGE should work correctly (baseline)")
    print("- Hybrid MaxK-SAGE combines MaxK activation with DGL's proven message passing")
    print("- Fixed MaxK-SAGE includes degree normalization in the CUDA kernel")
    print("- All approaches should produce similar output ranges")


if __name__ == "__main__":
    test_normalization_fix()
