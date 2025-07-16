#!/usr/bin/env python3
"""
Edge Weight Normalized MaxK SpGEMM Function
FUNCTION FILE ONLY - respects your modular structure
"""

import torch
from torch.autograd import Function
import numpy as np

# Import your existing kernels - NO FALLBACKS
try:
    import maxk_cuda_kernels
    MAXK_KERNELS_AVAILABLE = True
    assert MAXK_KERNELS_AVAILABLE, "MaxK kernels REQUIRED"
except ImportError as e:
    raise ImportError(f"MaxK CUDA kernels REQUIRED: {e}")

class EdgeWeightNormalizedMaxKSpGEMMFunction(Function):
    """
    SpGEMM Function with pre-normalized edge weights
    Forward: CSR with 1/in_degree weights
    Backward: CSC with 1/out_degree weights
    NO FALLBACKS - DEBUG UNTIL DEATH
    """
    
    @staticmethod
    def forward(ctx, graph_indices_csr, graph_values_csr_normalized, 
                graph_indices_csc, graph_values_csc_normalized,
                topk_values, topk_indices, warp4_metadata, num_warps, graph_indptr):
        """
        Forward pass using CSR with pre-normalized edge weights
        
        Args:
            graph_indices_csr: CSR indices for forward pass
            graph_values_csr_normalized: CSR weights = 1/in_degree[dst]
            graph_indices_csc: CSC indices for backward pass  
            graph_values_csc_normalized: CSC weights = 1/out_degree[src]
            topk_values: Pre-computed TopK values (V x k)
            topk_indices: Pre-computed TopK indices (V x k)
            warp4_metadata: Required warp metadata
            num_warps: Number of warps
            graph_indptr: CSR row pointers
        
        Returns:
            output: Already normalized output (no post-processing needed)
        """
        assert warp4_metadata is not None, "warp4_metadata REQUIRED"
        assert topk_values is not None, "topk_values REQUIRED"
        assert topk_indices is not None, "topk_indices REQUIRED"
        
        k_value = topk_values.size(1)
        sparse_selector = topk_indices.to(torch.uint8)
        
        # Save for backward pass
        ctx.save_for_backward(
            graph_indices_csr, graph_values_csr_normalized,
            graph_indices_csc, graph_values_csc_normalized, 
            sparse_selector
        )
        ctx.k_value = k_value
        ctx.warp4_metadata = warp4_metadata
        ctx.num_warps = num_warps
        ctx.graph_indptr = graph_indptr
        
        # Forward pass with pre-normalized CSR weights
        output = maxk_cuda_kernels.spmm_maxk_forward(
            warp4_metadata,
            graph_indices_csr,
            graph_values_csr_normalized,  # 🔥 Pre-normalized (1/in_degree)
            topk_values,
            sparse_selector,
            num_warps,
            k_value
        )
        
        # Output is already normalized - no post-processing needed!
        return output
    
    @staticmethod
    def backward(ctx, grad_output):
        """
        Backward pass using CSC with pre-normalized edge weights
        NO FALLBACKS - MUST WORK
        """
        (graph_indices_csr, graph_values_csr_normalized,
         graph_indices_csc, graph_values_csc_normalized,
         sparse_selector) = ctx.saved_tensors
        
        k_value = ctx.k_value
        warp4_metadata = ctx.warp4_metadata
        num_warps = ctx.num_warps
        
        # Backward pass with pre-normalized CSC weights
        grad_sparse = maxk_cuda_kernels.spmm_maxk_backward(
            warp4_metadata,
            graph_indices_csc,           # CSC indices (transpose)
            graph_values_csc_normalized, # 🔥 Pre-normalized (1/out_degree)
            grad_output,
            sparse_selector,
            num_warps,
            k_value
        )
        
        # Return gradients: (csr_indices, csr_weights, csc_indices, csc_weights, topk_values, topk_indices, ...)
        return (None, None, None, None, grad_sparse, None, None, None, None)

# Convenience function for your models to use
def edge_weight_normalized_maxk_spgemm(graph_indices_csr, graph_values_csr_normalized,
                                      graph_indices_csc, graph_values_csc_normalized,
                                      topk_values, topk_indices, 
                                      warp4_metadata, num_warps, graph_indptr):
    """
    Edge weight normalized MaxK SpGEMM operation
    
    Args:
        graph_indices_csr: CSR indices 
        graph_values_csr_normalized: CSR weights normalized by 1/in_degree
        graph_indices_csc: CSC indices for backward transpose
        graph_values_csc_normalized: CSC weights normalized by 1/out_degree
        topk_values: Pre-computed TopK values
        topk_indices: Pre-computed TopK indices
        warp4_metadata: Warp metadata
        num_warps: Number of warps
        graph_indptr: CSR row pointers
    
    Returns:
        output: Already normalized output
    """
    return EdgeWeightNormalizedMaxKSpGEMMFunction.apply(
        graph_indices_csr, graph_values_csr_normalized,
        graph_indices_csc, graph_values_csc_normalized,
        topk_values, topk_indices, warp4_metadata, num_warps, graph_indptr
    )

class EdgeWeightNormalizedMaxKSpmmWrapper:
    """
    Wrapper class for edge weight normalized SpMM operations
    Manages metadata and provides clean interface for models
    """
    
    def __init__(self, graph_name="", num_warps=12, warp_max_nz=64):
        self.graph_name = graph_name
        self.warp4_metadata = None
        self.num_warps = 0
        self.num_warps_config = num_warps
        self.warp_max_nz = warp_max_nz
        
    def load_metadata(self, graph_name=None):
        """Load warp4 metadata"""
        if graph_name is None:
            graph_name = self.graph_name
            
        assert graph_name, "graph_name REQUIRED"
        
        self.warp4_metadata = maxk_cuda_kernels.load_warp4_metadata(
            graph_name, self.num_warps_config, self.warp_max_nz
        )
        self.num_warps = self.warp4_metadata.size(0) // 4
        
        assert self.warp4_metadata is not None, f"Failed to load metadata for {graph_name}"
        return True
    
    def spmm(self, graph_indices_csr, graph_values_csr_normalized,
             graph_indices_csc, graph_values_csc_normalized,
             topk_values, topk_indices, graph_indptr):
        """
        Perform SpMM with pre-normalized edge weights
        
        Args:
            graph_indices_csr: CSR indices
            graph_values_csr_normalized: Pre-normalized CSR weights
            graph_indices_csc: CSC indices  
            graph_values_csc_normalized: Pre-normalized CSC weights
            topk_values: TopK values
            topk_indices: TopK indices
            graph_indptr: CSR row pointers
        
        Returns:
            Already normalized output
        """
        assert self.warp4_metadata is not None, "Metadata not loaded"
        
        return edge_weight_normalized_maxk_spgemm(
            graph_indices_csr, graph_values_csr_normalized,
            graph_indices_csc, graph_values_csc_normalized,
            topk_values, topk_indices,
            self.warp4_metadata, self.num_warps, graph_indptr
        )

def test_edge_weight_spgemm_function():
    """Test the edge weight SpGEMM function"""
    print("🧪 Testing Edge Weight SpGEMM Function")
    print("=" * 50)
    
    assert torch.cuda.is_available(), "CUDA REQUIRED"
    
    # Create test data
    V, E, k = 1000, 5000, 32
    device = 'cuda'
    
    # Generate test graph data
    torch.manual_seed(42)
    csr_indices = torch.randint(0, V, (E,), device=device, dtype=torch.int32)
    csc_indices = torch.randint(0, V, (E,), device=device, dtype=torch.int32)
    
    # Pre-normalized weights (simulated)
    csr_weights_norm = torch.rand(E, device=device, dtype=torch.float32) * 0.1 + 0.01
    csc_weights_norm = torch.rand(E, device=device, dtype=torch.float32) * 0.1 + 0.01
    
    # TopK data
    topk_values = torch.rand(V, k, device=device, dtype=torch.float32, requires_grad=True)
    topk_indices = torch.randint(0, 256, (V, k), device=device, dtype=torch.int64)
    
    # Create dummy metadata (you'd load real metadata)
    warp4_metadata = torch.randint(0, 1000, (100,), device=device, dtype=torch.int32)
    indptr = torch.arange(0, E+1, E//V, device=device, dtype=torch.int32)
    
    print(f"📊 Test data: {V} nodes, {E} edges, k={k}")
    
    try:
        # Test forward pass
        output = edge_weight_normalized_maxk_spgemm(
            csr_indices, csr_weights_norm,
            csc_indices, csc_weights_norm,
            topk_values, topk_indices,
            warp4_metadata, 25, indptr
        )
        
        print(f"✅ Forward pass: {output.shape}")
        
        # Test backward pass
        loss = output.sum()
        loss.backward()
        
        assert topk_values.grad is not None, "No gradients computed"
        print(f"✅ Backward pass: {topk_values.grad.shape}")
        
        print(f"🎉 Edge Weight SpGEMM Function Test PASSED!")
        
    except Exception as e:
        print(f"💀 FATAL ERROR in function: {e}")
        raise

if __name__ == "__main__":
    test_edge_weight_spgemm_function()
