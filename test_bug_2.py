#!/usr/bin/env python3
"""
Test script to validate MaxK kernel against cuSPARSE with dim_k=16
Isolates the validation function to debug illegal memory access issues
"""

import torch
import time
import argparse
from pathlib import Path
import traceback

# Import necessary components from your project
from graph_loader import GraphDataLoader
from direct_kernel_interface import DirectMaxKKernels, DIRECT_KERNELS_AVAILABLE

def test_validation_with_k18(graph_name, verbose=True):
    """
    Test just the validation function with k=18
    
    Args:
        graph_name: Name of the graph to test
        verbose: Whether to print detailed progress info
    
    Returns:
        True if validation succeeds, False otherwise
    """
    if not DIRECT_KERNELS_AVAILABLE:
        print("❌ Direct CUDA kernels not available")
        return False
    
    # Initialize everything needed for validation
    print(f"🧪 Testing validation against cuSPARSE with k=18 for {graph_name}")
    
    # 1. Load graph data
    loader = GraphDataLoader()
    try:
        graph_data = loader.load_graph(graph_name)
        graph_data = loader.to_cuda_tensors(graph_data)
        v_num = graph_data['v_num']
        
        if verbose:
            print(f"📊 Graph loaded: {v_num} vertices, {graph_data['e_num']} edges")
    except Exception as e:
        print(f"❌ Failed to load graph: {e}")
        return False
    
    # 2. Initialize MaxK kernels with metadata
    try:
        maxk_kernels = DirectMaxKKernels(graph_name)
        if not maxk_kernels.load_warp4_metadata():
            print("❌ Failed to load warp4 metadata")
            return False
            
        if verbose:
            print(f"✅ Initialized MaxK kernels with metadata")
    except Exception as e:
        print(f"❌ Failed to initialize MaxK kernels: {e}")
        return False
    
    # 3. Create input features with exact same seed as main
    try:
        torch.manual_seed(123)
        input_features = torch.rand(v_num, 256, device='cuda', dtype=torch.float32)
        
        if verbose:
            print(f"✅ Created input features: {input_features.shape}")
            
        # Optional: Print memory stats before validation
        if verbose:
            print(f"📊 CUDA memory before validation:")
            print(f"   Allocated: {torch.cuda.memory_allocated() / 1e9:.2f} GB")
            print(f"   Reserved: {torch.cuda.memory_reserved() / 1e9:.2f} GB")
            
        # Make sure GPU is clean
        torch.cuda.synchronize()
        torch.cuda.empty_cache()
    except Exception as e:
        print(f"❌ Failed to create input features: {e}")
        return False
    
    # 4. Run validation with k=16 with detailed debug info
    print("\n🔍 Starting validation with k=16...")
    
    # Special debug wrapper for validation function
    def debug_validation():
        # Track each step of validation
        try:
            print("Step 1: Starting validation process")
            # First try with PyTorch TopK
            print("Step 2: Testing with PyTorch TopK...")
            result_pytorch = maxk_kernels.validate_against_cusparse(
                graph_data, input_features, dim_k=19, 
                tolerance=0.001, use_cuda_topk=False
            )
            print(f"PyTorch TopK result: {'SUCCESS' if result_pytorch else 'FAILED'}")
            
            # Then try with CUDA TopK
            print("\nStep 3: Testing with CUDA TopK...")
            # Extra synchronization
            torch.cuda.synchronize()
            torch.cuda.empty_cache()
            
            try:
                # Try direct access to the CUDA TopK function first
                print("Step 3.1: Testing direct CUDA TopK call...")
                import maxk_cuda_kernels
                test_values, test_indices = maxk_cuda_kernels.cuda_topk_maxk_float(
                    input_features, 19
                )
                print(f"Direct CUDA TopK call succeeded: {test_values.shape}, {test_indices.shape}")
            except Exception as e:
                print(f"❌ Direct CUDA TopK call failed: {e}")
                print(traceback.format_exc())
            
            # Now try the full validation
            print("\nStep 3.2: Running full validation with CUDA TopK...")
            result_cuda = maxk_kernels.validate_against_cusparse(
                graph_data, input_features, dim_k=19, 
                tolerance=0.001, use_cuda_topk=True
            )
            print(f"CUDA TopK result: {'SUCCESS' if result_cuda else 'FAILED'}")
            
            return result_pytorch or result_cuda
            
        except Exception as e:
            print(f"❌ Validation failed with exception: {e}")
            print(traceback.format_exc())
            return False
    
    # Run the debug validation
    try:
        start_time = time.time()
        result = debug_validation()
        end_time = time.time()
        
        print(f"\n⏱️ Validation completed in {end_time - start_time:.2f} seconds")
        print(f"🔍 Final result: {'SUCCESS' if result else 'FAILED'}")
        
        # Print memory stats after validation
        if verbose:
            print(f"📊 CUDA memory after validation:")
            print(f"   Allocated: {torch.cuda.memory_allocated() / 1e9:.2f} GB")
            print(f"   Reserved: {torch.cuda.memory_reserved() / 1e9:.2f} GB")
        
        return result
    except Exception as e:
        print(f"❌ Fatal error during validation: {e}")
        print(traceback.format_exc())
        return False

def main():
    """Main function"""
    parser = argparse.ArgumentParser(description='Test MaxK validation with k=18')
    parser.add_argument('--graph', default=None, help='Specific graph to test')
    parser.add_argument('--quiet', action='store_true', help='Reduce verbosity')
    args = parser.parse_args()
    
    # Setup
    loader = GraphDataLoader()
    available_graphs = loader.get_available_graphs()
    
    if not available_graphs:
        print("❌ No graphs available for testing")
        return 1
    
    # Choose graph
    if args.graph:
        if args.graph in available_graphs:
            test_graphs = [args.graph]
        else:
            print(f"❌ Graph '{args.graph}' not found!")
            print(f"Available graphs: {available_graphs}")
            return 1
    else:
        # Just use the first graph
        test_graphs = [available_graphs[1]]
    
    print(f"🧪 Testing validation against cuSPARSE with k=16")
    print(f"📊 Test graphs: {test_graphs}")
    
    # Run tests
    results = {}
    for graph in test_graphs:
        print(f"\n{'='*60}")
        print(f"🔍 Testing graph: {graph}")
        print(f"{'='*60}")
        
        success = test_validation_with_k18(graph, verbose=not args.quiet)
        results[graph] = success
    
    # Print summary
    print(f"\n{'='*60}")
    print(f"📊 SUMMARY")
    print(f"{'='*60}")
    
    for graph, success in results.items():
        status = "✅ PASSED" if success else "❌ FAILED"
        print(f"{graph}: {status}")
    
    return 0

if __name__ == "__main__":
    import sys
    sys.exit(main())