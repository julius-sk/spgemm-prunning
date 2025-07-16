#!/usr/bin/env python3
"""
Complete graph processing: Make undirected, add self-loops, remove multi-edges, save
"""

import torch
import dgl
import numpy as np
from dgl import AddSelfLoop
from dgl.data import RedditDataset, FlickrDataset, YelpDataset
from ogb.nodeproppred import DglNodePropPredDataset
from utils.proteins_loader import load_proteins
import os

def process_single_dataset(dataset_name, data_path="./data/"):
    """Process single dataset with all cleaning steps"""
    
    print(f"\n🔄 PROCESSING {dataset_name.upper()}")
    print("=" * 50)
    
    # Step 1: Load dataset (EXACT training code)
    print("1️⃣ Loading dataset...")
    if "ogb" not in dataset_name:
        # NON-OGB datasets
        if dataset_name == 'reddit':
            data = RedditDataset(raw_dir=data_path)  # NO transform yet
        elif dataset_name == 'flickr':
            data = FlickrDataset(raw_dir=data_path)
        elif dataset_name == 'yelp':
            data = YelpDataset(raw_dir=data_path)
        g = data[0].int()
    elif "proteins" not in dataset_name:
        # OGB non-proteins
        data = DglNodePropPredDataset(name=dataset_name, root=data_path)
        g, labels = data[0]
        g = g.int()
    else:
        # OGB proteins
        data, g, labels, train_idx, val_idx, test_idx = load_proteins(data_path)
        g = g.int()
    
    print(f"   Original: {g.num_nodes():,} nodes, {g.num_edges():,} edges")
    
    # Step 2: Make undirected
    print("2️⃣ Making undirected...")
    src, dst = g.edges()
    
    # Add reverse edges
    all_src = torch.cat([src, dst])
    all_dst = torch.cat([dst, src])
    
    # Create undirected graph
    g_undirected = dgl.graph((all_src, all_dst), num_nodes=g.num_nodes())
    
    # Copy node features
    for key, val in g.ndata.items():
        g_undirected.ndata[key] = val
    
    print(f"   Undirected: {g_undirected.num_nodes():,} nodes, {g_undirected.num_edges():,} edges")
    
    # Step 3: Add self-loops
    print("3️⃣ Adding self-loops...")
    g_with_selfloop = dgl.add_self_loop(g_undirected)
    
    src_self, dst_self = g_with_selfloop.edges()
    selfloop_count = (src_self == dst_self).sum().item()
    print(f"   With self-loops: {g_with_selfloop.num_nodes():,} nodes, {g_with_selfloop.num_edges():,} edges")
    print(f"   Self-loops added: {selfloop_count:,}")
    
    # Step 4: Remove multi-edges
    print("4️⃣ Removing multi-edges...")
    src_clean, dst_clean = g_with_selfloop.edges()
    
    # Find unique edges
    edge_set = set()
    keep_indices = []
    
    for i, (s, d) in enumerate(zip(src_clean.cpu().numpy(), dst_clean.cpu().numpy())):
        edge_key = (int(s), int(d))
        if edge_key not in edge_set:
            edge_set.add(edge_key)
            keep_indices.append(i)
    
    print(f"   Multi-edges removed: {len(src_clean) - len(keep_indices):,}")
    
    # Create final clean graph
    if len(keep_indices) < len(src_clean):
        unique_src = src_clean[keep_indices]
        unique_dst = dst_clean[keep_indices]
        g_final = dgl.graph((unique_src, unique_dst), num_nodes=g_with_selfloop.num_nodes())
        
        # Copy node features
        for key, val in g_with_selfloop.ndata.items():
            g_final.ndata[key] = val
    else:
        g_final = g_with_selfloop
    
    print(f"   Final: {g_final.num_nodes():,} nodes, {g_final.num_edges():,} edges")
    
    # Step 5: Save as .indptr and .indices
    print("5️⃣ Saving to files...")
    
    # Get CSR format using correct method
    indptr, indices, edge_ids = g_final.adj_tensors('csr')
    
    # Create output directory
    output_dir = "./processed_graphs/"
    os.makedirs(output_dir, exist_ok=True)
    
    # Save files
    indptr_file = f"{output_dir}{dataset_name}.indptr"
    indices_file = f"{output_dir}{dataset_name}.indices"
    
    indptr.cpu().numpy().astype(np.int32).tofile(indptr_file)
    indices.cpu().numpy().astype(np.int32).tofile(indices_file)
    
    print(f"   ✅ Saved: {indptr_file}")
    print(f"   ✅ Saved: {indices_file}")
    
    # Verify self-loops in final graph
    final_src, final_dst = g_final.edges()
    final_selfloops = (final_src == final_dst).sum().item()
    print(f"   ✅ Final self-loops: {final_selfloops:,} (should be {g_final.num_nodes():,})")
    
    return g_final

def process_all_datasets():
    """Process all 6 datasets"""
    
    print("🔄 PROCESSING ALL DATASETS")
    print("=" * 60)
    print("Steps: Load → Undirected → Self-loops → Remove multi-edges → Save")
    
    datasets = ['reddit', 'flickr', 'yelp', 'ogbn-arxiv', 'ogbn-products', 'ogbn-proteins']
    results = []
    
    for dataset in datasets:
        try:
            g_final = process_single_dataset(dataset)
            
            # Record results
            src, dst = g_final.edges()
            selfloops = (src == dst).sum().item()
            
            results.append({
                'dataset': dataset,
                'nodes': g_final.num_nodes(),
                'edges': g_final.num_edges(),
                'selfloops': selfloops,
                'status': 'SUCCESS'
            })
            
        except Exception as e:
            print(f"❌ {dataset} failed: {e}")
            results.append({
                'dataset': dataset,
                'nodes': 0,
                'edges': 0,
                'selfloops': 0,
                'status': f'FAILED: {e}'
            })
    
    # Summary table
    print(f"\n📊 PROCESSING SUMMARY")
    print("=" * 80)
    print(f"{'Dataset':<15} {'Nodes':<10} {'Edges':<12} {'Self-loops':<12} {'Status':<15}")
    print("-" * 80)
    
    for r in results:
        if r['status'] == 'SUCCESS':
            print(f"{r['dataset']:<15} {r['nodes']:<10,} {r['edges']:<12,} "
                  f"{r['selfloops']:<12,} {r['status']:<15}")
        else:
            print(f"{r['dataset']:<15} {'N/A':<10} {'N/A':<12} {'N/A':<12} {r['status']:<15}")
    
    # Final instructions
    print(f"\n🎯 NEXT STEPS:")
    print(f"1. ✅ Processed graphs saved to ./processed_graphs/")
    print(f"2. 🔧 Generate metadata using these files:")
    print(f"   python generate_meta.py --input_dir ./processed_graphs/")
    print(f"3. 🚀 Use same graphs for training and kernels!")
    
    return results

def verify_processed_graphs():
    """Verify the processed graphs are correct"""
    
    print(f"\n🔍 VERIFYING PROCESSED GRAPHS")
    print("=" * 40)
    
    graphs_dir = "./processed_graphs/"
    
    if not os.path.exists(graphs_dir):
        print("❌ No processed graphs found!")
        return
    
    # Find all .indptr files
    indptr_files = [f for f in os.listdir(graphs_dir) if f.endswith('.indptr')]
    
    for indptr_file in indptr_files:
        dataset_name = indptr_file.replace('.indptr', '')
        
        indptr_path = os.path.join(graphs_dir, indptr_file)
        indices_path = os.path.join(graphs_dir, f"{dataset_name}.indices")
        
        if not os.path.exists(indices_path):
            print(f"❌ {dataset_name}: Missing .indices file")
            continue
        
        # Load and verify
        indptr = np.fromfile(indptr_path, dtype=np.int32)
        indices = np.fromfile(indices_path, dtype=np.int32)
        
        v_num = len(indptr) - 1
        e_num = len(indices)
        
        # Count self-loops
        selfloops = 0
        for i in range(v_num):
            start, end = indptr[i], indptr[i+1]
            neighbors = indices[start:end]
            if i in neighbors:
                selfloops += 1
        
        print(f"✅ {dataset_name}: {v_num:,} nodes, {e_num:,} edges, {selfloops:,} self-loops")

if __name__ == "__main__":
    # Process all datasets
    results = process_all_datasets()
    
    # Verify results
    verify_processed_graphs()
