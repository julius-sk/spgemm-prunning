#!/usr/bin/env python3
"""
Enhanced Metadata Generator for MaxK-GNN
Generates both CSR (forward) and CSC (backward) warp4 metadata files
Fixes the backward kernel transpose issue by providing proper CSC metadata
"""

from pathlib import Path
import numpy as np
from scipy.sparse import csr_matrix
import os
import time

def generate_warp4_metadata(indices, indptr, v_num, e_num, format_name="CSR"):
    """
    Generate warp4 metadata for given sparse matrix format
    
    Args:
        indices: Column indices (CSR) or row indices (CSC)
        indptr: Row pointers (CSR) or column pointers (CSC) 
        v_num: Number of vertices
        e_num: Number of edges
        format_name: "CSR" or "CSC" for logging
        
    Returns:
        warp_4: Flattened array of warp metadata
    """
    num_warps = 12
    warp_max_nz = 64
    
    print(f"  Generating {format_name} metadata...")
    print(f"    Format: {format_name}")
    print(f"    Vertices: {v_num:,}")
    print(f"    Edges: {e_num:,}")
    print(f"    Warps per block: {num_warps}")
    print(f"    Max NZ per warp: {warp_max_nz}")
    
    warp_row = []
    warp_loc = []
    warp_len = []
    cur_loc = 0
    
    # Track statistics
    total_warps = 0
    nodes_processed = 0
    max_degree = 0
    degrees = []
    
    for i in range(v_num):
        cur_degree = indptr[i+1] - indptr[i]
        if cur_degree == 0:
            continue
            
        degrees.append(cur_degree)
        max_degree = max(max_degree, cur_degree)
        nodes_processed += 1
        
        tmp_loc = 0
        node_warps = 0
        
        while True:
            warp_row.append(i)
            warp_loc.append(cur_loc)
            
            if cur_degree - tmp_loc <= warp_max_nz:
                # Last warp for this node
                warp_len.append(cur_degree - tmp_loc)
                cur_loc += cur_degree - tmp_loc
                node_warps += 1
                break
            else:
                # Full warp
                warp_len.append(warp_max_nz)
                cur_loc += warp_max_nz
                tmp_loc += warp_max_nz
                node_warps += 1
        
        total_warps += node_warps
    
    # Create padded metadata (4 integers per warp)
    pad = np.zeros_like(warp_row)
    warp_4 = np.dstack([warp_row, warp_loc, warp_len, pad]).flatten()
    
    # Print statistics
    avg_degree = np.mean(degrees) if degrees else 0
    print(f"    Statistics:")
    print(f"      Nodes with edges: {nodes_processed:,}")
    print(f"      Total warps generated: {total_warps:,}")
    print(f"      Average degree: {avg_degree:.2f}")
    print(f"      Max degree: {max_degree:,}")
    print(f"      Avg warps per node: {total_warps/nodes_processed:.2f}")
    
    return warp_4.astype(np.int32)

def process_graph_file(file_path, base_path, meta_dir_csr, meta_dir_csc):
    """
    Process a single graph file and generate both CSR and CSC metadata
    
    Args:
        file_path: Path to the .indptr file
        base_path: Base directory containing graph files
        meta_dir_csr: Output directory for CSR metadata  
        meta_dir_csc: Output directory for CSC metadata
        
    Returns:
        Tuple of (success, graph_name, csr_warps, csc_warps)
    """
    graph_name = file_path.stem
    
    try:
        print(f"\n📊 Processing graph: {graph_name}")
        print("=" * 50)
        
        # Load graph data
        indptr_file = base_path + graph_name + ".indptr"
        indices_file = base_path + graph_name + ".indices"
        
        print(f"  Loading files:")
        print(f"    indptr: {indptr_file}")
        print(f"    indices: {indices_file}")
        
        indptr = np.fromfile(indptr_file, dtype=np.int32)
        indices = np.fromfile(indices_file, dtype=np.int32)
        
        v_num = len(indptr) - 1
        e_num = len(indices)
        
        print(f"  Graph loaded:")
        print(f"    Vertices: {v_num:,}")
        print(f"    Edges: {e_num:,}")
        print(f"    Avg degree: {e_num/v_num:.2f}")
        
        # Create sparse matrix and convert to CSC
        print(f"\n🔄 Converting CSR to CSC...")
        vals = np.ones(e_num, dtype=np.float32)
        csr = csr_matrix((vals, indices, indptr), shape=(v_num, v_num))
        
        # Convert to CSC format  
        csc = csr.tocsc()
        csc_indices = csc.indices.astype(np.int32)
        csc_indptr = csc.indptr.astype(np.int32)
        
        print(f"  CSC conversion complete:")
        print(f"    CSC indices shape: {csc_indices.shape}")
        print(f"    CSC indptr shape: {csc_indptr.shape}")
        print(f"    CSC nnz: {csc.nnz:,}")
        
        # Verify conversion
        if csc.nnz != e_num:
            print(f"  ⚠️ Warning: CSC nnz ({csc.nnz}) != original edges ({e_num})")
        
        # Generate CSR metadata (for forward pass)
        print(f"\n🔧 Generating CSR metadata (forward pass)...")
        start_time = time.time()
        warp_4_csr = generate_warp4_metadata(indices, indptr, v_num, e_num, "CSR")
        csr_time = time.time() - start_time
        
        # Generate CSC metadata (for backward pass) 
        print(f"\n🔧 Generating CSC metadata (backward pass)...")
        start_time = time.time()
        warp_4_csc = generate_warp4_metadata(csc_indices, csc_indptr, v_num, e_num, "CSC")
        csc_time = time.time() - start_time
        
        # Save metadata files
        csr_file = meta_dir_csr + graph_name + '.warp4'
        csc_file = meta_dir_csc + graph_name + '.warp4_csc'
        
        print(f"\n💾 Saving metadata files...")
        print(f"  CSR: {csr_file}")
        print(f"  CSC: {csc_file}")
        
        warp_4_csr.tofile(csr_file)
        warp_4_csc.tofile(csc_file)
        
        # Verify file sizes
        csr_size = os.path.getsize(csr_file)
        csc_size = os.path.getsize(csc_file)
        
        print(f"  File sizes:")
        print(f"    CSR: {csr_size:,} bytes ({len(warp_4_csr)} int32s)")
        print(f"    CSC: {csc_size:,} bytes ({len(warp_4_csc)} int32s)")
        
        print(f"  Generation time:")
        print(f"    CSR: {csr_time:.3f}s")
        print(f"    CSC: {csc_time:.3f}s")
        
        csr_warps = len(warp_4_csr) // 4
        csc_warps = len(warp_4_csc) // 4
        
        print(f"✅ {graph_name} completed successfully!")
        print(f"   CSR warps: {csr_warps:,}")
        print(f"   CSC warps: {csc_warps:,}")
        
        return True, graph_name, csr_warps, csc_warps
        
    except Exception as e:
        print(f"❌ Error processing {graph_name}: {e}")
        import traceback
        traceback.print_exc()
        return False, graph_name, 0, 0

def verify_metadata_symmetry(base_path, meta_dir_csr, meta_dir_csc, graph_name):
    """
    Verify that CSR and CSC metadata are consistent for symmetric graphs
    
    Args:
        base_path: Base directory containing graph files
        meta_dir_csr: CSR metadata directory
        meta_dir_csc: CSC metadata directory  
        graph_name: Name of graph to verify
    """
    try:
        print(f"\n🔍 Verifying metadata consistency for {graph_name}...")
        
        # Load original graph
        indptr = np.fromfile(base_path + graph_name + ".indptr", dtype=np.int32)
        indices = np.fromfile(base_path + graph_name + ".indices", dtype=np.int32)
        v_num = len(indptr) - 1
        e_num = len(indices)
        
        # Create CSR and CSC matrices
        vals = np.ones(e_num)
        csr = csr_matrix((vals, indices, indptr), shape=(v_num, v_num))
        csc = csr.tocsc()
        
        # Check if graph is symmetric
        diff = csr - csc.T
        is_symmetric = diff.nnz == 0
        
        print(f"  Graph symmetry: {'Symmetric' if is_symmetric else 'Asymmetric'}")
        print(f"  Original edges: {e_num:,}")
        print(f"  CSR nnz: {csr.nnz:,}")
        print(f"  CSC nnz: {csc.nnz:,}")
        
        if not is_symmetric:
            print(f"  Asymmetric entries: {diff.nnz:,}")
            print(f"  ✅ CSC metadata is ESSENTIAL for correct backward pass!")
        else:
            print(f"  ✅ Graph is symmetric, but CSC metadata still recommended")
        
        # Load and compare metadata
        csr_meta = np.fromfile(meta_dir_csr + graph_name + '.warp4', dtype=np.int32)
        csc_meta = np.fromfile(meta_dir_csc + graph_name + '.warp4_csc', dtype=np.int32)
        
        csr_warps = len(csr_meta) // 4
        csc_warps = len(csc_meta) // 4
        
        print(f"  Metadata comparison:")
        print(f"    CSR warps: {csr_warps:,}")
        print(f"    CSC warps: {csc_warps:,}")
        print(f"    Warp count {'matches' if csr_warps == csc_warps else 'differs'}")
        
        return is_symmetric
        
    except Exception as e:
        print(f"⚠️ Verification failed: {e}")
        return None

def main():
    """Main function to generate both CSR and CSC metadata"""
    print("🚀 MaxK-GNN Enhanced Metadata Generator")
    print("=" * 60)
    print("Generates both CSR (forward) and CSC (backward) warp4 metadata")
    print("Fixes backward kernel transpose issue!")
    print()
    
    # Configuration
    base_path = "./graphs/"
    
    # Create output directories
    meta_dir_csr = './w12_nz64_warp_4/'
    meta_dir_csc = './w12_nz64_warp_4_csc/'
    
    os.makedirs(meta_dir_csr, exist_ok=True)
    os.makedirs(meta_dir_csc, exist_ok=True)
    
    print(f"📁 Directories:")
    print(f"  Input graphs: {base_path}")
    print(f"  CSR metadata: {meta_dir_csr}")
    print(f"  CSC metadata: {meta_dir_csc}")
    
    # Find graph files
    fileset = list(Path(base_path).glob('*.indptr'))
    total_files = len(fileset)
    
    if total_files == 0:
        print(f"❌ No .indptr files found in {base_path}")
        print("   Ensure graph files are in the correct directory")
        return
    
    print(f"\n📊 Found {total_files} graph files:")
    for i, file in enumerate(fileset[:5], 1):
        print(f"  {i}. {file.stem}")
    if total_files > 5:
        print(f"  ... and {total_files - 5} more")
    
    # Process all graphs
    print(f"\n🔄 Processing {total_files} graphs...")
    
    successful = 0
    failed = 0
    total_csr_warps = 0
    total_csc_warps = 0
    symmetric_graphs = []
    asymmetric_graphs = []
    
    start_time = time.time()
    
    for i, file in enumerate(fileset, 1):
        print(f"\n[{i}/{total_files}] " + "="*50)
        
        success, graph_name, csr_warps, csc_warps = process_graph_file(
            file, base_path, meta_dir_csr, meta_dir_csc
        )
        
        if success:
            successful += 1
            total_csr_warps += csr_warps
            total_csc_warps += csc_warps
            
            # Verify symmetry
            is_symmetric = verify_metadata_symmetry(
                base_path, meta_dir_csr, meta_dir_csc, graph_name
            )
            
            if is_symmetric is True:
                symmetric_graphs.append(graph_name)
            elif is_symmetric is False:
                asymmetric_graphs.append(graph_name)
        else:
            failed += 1
    
    end_time = time.time()
    total_time = end_time - start_time
    
    # Final summary
    print(f"\n" + "="*60)
    print(f"📊 METADATA GENERATION COMPLETE")
    print(f"=" * 60)
    print(f"✅ Successful: {successful}/{total_files} graphs")
    print(f"❌ Failed: {failed}/{total_files} graphs")
    print(f"⏱️ Total time: {total_time:.2f}s")
    print(f"📈 Average time per graph: {total_time/total_files:.2f}s")
    
    print(f"\n📊 Warp Statistics:")
    print(f"  Total CSR warps: {total_csr_warps:,}")
    print(f"  Total CSC warps: {total_csc_warps:,}")
    print(f"  Average CSR warps per graph: {total_csr_warps/successful:.0f}")
    print(f"  Average CSC warps per graph: {total_csc_warps/successful:.0f}")
    
    print(f"\n🔍 Graph Symmetry Analysis:")
    print(f"  Symmetric graphs: {len(symmetric_graphs)}")
    if symmetric_graphs:
        print(f"    {symmetric_graphs[:3]}{'...' if len(symmetric_graphs) > 3 else ''}")
    print(f"  Asymmetric graphs: {len(asymmetric_graphs)}")  
    if asymmetric_graphs:
        print(f"    {asymmetric_graphs[:3]}{'...' if len(asymmetric_graphs) > 3 else ''}")
    
    print(f"\n💡 Usage Instructions:")
    print(f"  1. Forward kernel (spmm_maxk.cu): Use CSR metadata")
    print(f"     File pattern: {meta_dir_csr}{{graph}}.warp4")
    print(f"  2. Backward kernel (spmm_maxk_backward.cu): Use CSC metadata")
    print(f"     File pattern: {meta_dir_csc}{{graph}}.warp4_csc")
    print(f"  3. Update kernel loading code to use appropriate metadata")
    
    print(f"\n🎯 Benefits:")
    print(f"  ✅ Mathematically correct backward pass (A^T operation)")
    print(f"  ✅ Better memory coalescing for transpose operations")
    print(f"  ✅ Improved training convergence and accuracy")
    print(f"  ✅ Works correctly for both symmetric and asymmetric graphs")
    
    if asymmetric_graphs:
        print(f"\n⚠️  CRITICAL: {len(asymmetric_graphs)} asymmetric graphs detected!")
        print(f"   These graphs REQUIRE CSC metadata for correct backward pass")
        print(f"   Using CSR metadata for backward pass would be mathematically wrong")
    
    print(f"\n🔧 Next Steps:")
    print(f"  1. Update spmm_maxk_backward.cu to load .warp4_csc files")
    print(f"  2. Modify kernel loading logic to use CSC metadata for backward")
    print(f"  3. Test training with new metadata on asymmetric graphs")
    print(f"  4. Benchmark performance improvements")

if __name__ == "__main__":
    main()
