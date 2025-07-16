#!/usr/bin/env python3
"""
Check graphs using EXACT same loading code as training
"""

import torch
import dgl
import dgl.nn as dglnn
from dgl import AddSelfLoop
from dgl.data import RedditDataset, FlickrDataset, YelpDataset
from ogb.nodeproppred import DglNodePropPredDataset, Evaluator
from utils.proteins_loader import load_proteins

def check_dataset_exact_training(dataset_name, data_path="./data/"):
    """Load dataset using EXACT same code as maxk_gnn_integrated.py"""
    
    print(f"\n🔍 {dataset_name.upper()} - EXACT TRAINING CODE")
    print("=" * 50)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # EXACT COPY from maxk_gnn_integrated.py lines 150-210
    if "ogb" not in dataset_name:
        # load and preprocess dataset
        transform = (
            AddSelfLoop()
        )  # by default, it will first remove self-loops to prevent duplication
        if dataset_name == 'reddit':
            data = RedditDataset(transform=transform, raw_dir=data_path)
        elif dataset_name == 'flickr':
            data = FlickrDataset(transform=transform, raw_dir=data_path)
        elif dataset_name == 'yelp':
            data = YelpDataset(transform=transform, raw_dir=data_path)
        g = data[0]
        g = g.int().to(device)
        features = g.ndata["feat"]
        if dataset_name == 'yelp':
            labels = g.ndata["label"].float()#.float()
        else:
            labels = g.ndata["label"]
        masks = g.ndata["train_mask"].bool(), g.ndata["val_mask"].bool(), g.ndata["test_mask"].bool()
    elif "proteins" not in dataset_name:
        data = DglNodePropPredDataset(name=dataset_name, root=data_path)
        split_idx = data.get_idx_split()

        # there is only one graph in Node Property Prediction datasets
        g, labels = data[0]
        labels = torch.squeeze(labels, dim=1)
        g = g.int().to(device)
        features = g.ndata["feat"]
        
        labels = labels.to(device)
        
        train_mask = split_idx["train"]
        valid_mask = split_idx["valid"]
        test_mask = split_idx["test"]
        total_nodes = train_mask.shape[0] + valid_mask.shape[0] + test_mask.shape[0]
        train_mask_bin = torch.zeros(total_nodes, dtype=torch.bool)
        train_mask_bin[train_mask] = 1
        valid_mask_bin = torch.zeros(total_nodes, dtype=torch.bool)
        valid_mask_bin[valid_mask] = 1
        test_mask_bin = torch.zeros(total_nodes, dtype=torch.bool)
        test_mask_bin[test_mask] = 1
        masks = train_mask_bin.to(device), valid_mask_bin.to(device), test_mask_bin.to(device)
    ##### ogbn_proteins loader
    else:
        data, g, labels, train_idx, val_idx, test_idx = load_proteins(data_path)
        g = g.int().to(device)
        features = g.ndata["feat"]
        labels = labels.float().to(device)
        ### Get the train, validation, and test mask
        total_nodes = train_idx.shape[0] + val_idx.shape[0] + test_idx.shape[0]
        train_mask_bin = torch.zeros(total_nodes, dtype=torch.bool)
        train_mask_bin[train_idx] = 1
        valid_mask_bin = torch.zeros(total_nodes, dtype=torch.bool)
        valid_mask_bin[val_idx] = 1
        test_mask_bin = torch.zeros(total_nodes, dtype=torch.bool)
        test_mask_bin[test_idx] = 1
        masks = train_mask_bin.to(device), valid_mask_bin.to(device), test_mask_bin.to(device)
        
    in_size = features.shape[1]
    out_size = data.num_classes
    if dataset_name == 'ogbn-proteins':
        out_size = 112
    
    # EXACT COPY: Add self-loop check (line 220)
    selfloop = False  # Default value
    if selfloop:
        g = dgl.add_self_loop(g)

    # Check properties
    src, dst = g.edges()
    selfloop_count = (src == dst).sum().item()
    
    print(f"📊 Graph loaded with EXACT training code:")
    print(f"   Nodes: {g.num_nodes():,}")
    print(f"   Edges: {g.num_edges():,}")
    print(f"   Self-loops: {selfloop_count:,}")
    print(f"   Features: {features.shape}")
    print(f"   Labels: {labels.shape}")
    print(f"   Classes: {out_size}")
    
    # Check multi-edges
    edge_pairs = set(zip(src.cpu().numpy(), dst.cpu().numpy()))
    multi_edges = g.num_edges() - len(edge_pairs)
    print(f"   Multi-edges: {multi_edges:,}")
    
    # Check if self-loop transform worked
    if "ogb" not in dataset_name:
        print(f"   Transform: AddSelfLoop() applied")
        print(f"   Expected self-loops: {g.num_nodes():,}")
        if selfloop_count == g.num_nodes():
            print("   ✅ Every node has self-loop")
        else:
            print("   ❌ Missing self-loops!")
    
    return g, data

def check_all_datasets_training():
    """Check all 6 datasets using exact training code"""
    
    datasets = ['reddit', 'flickr', 'yelp', 'ogbn-arxiv', 'ogbn-products', 'ogbn-proteins']
    
    print("🔍 ALL DATASETS - EXACT TRAINING LOADING")
    print("=" * 60)
    
    results = []
    
    for dataset in datasets:
        try:
            g, data = check_dataset_exact_training(dataset)
            
            src, dst = g.edges()
            selfloops = (src == dst).sum().item()
            edge_pairs = set(zip(src.cpu().numpy(), dst.cpu().numpy()))
            multi_edges = g.num_edges() - len(edge_pairs)
            
            results.append({
                'dataset': dataset,
                'nodes': g.num_nodes(),
                'edges': g.num_edges(),
                'selfloops': selfloops,
                'multiedges': multi_edges
            })
            
        except Exception as e:
            print(f"❌ {dataset} failed: {e}")
            results.append({
                'dataset': dataset,
                'nodes': 0,
                'edges': 0,
                'selfloops': 0,
                'multiedges': 0,
                'error': str(e)
            })
    
    # Summary table
    print(f"\n📊 TRAINING DATASETS SUMMARY")
    print("=" * 80)
    print(f"{'Dataset':<15} {'Nodes':<10} {'Edges':<12} {'Self-loops':<12} {'Multi-edges':<12}")
    print("-" * 80)
    
    for r in results:
        if 'error' in r:
            print(f"{r['dataset']:<15} {'ERROR':<10} {'ERROR':<12} {'ERROR':<12} {'ERROR':<12}")
        else:
            print(f"{r['dataset']:<15} {r['nodes']:<10,} {r['edges']:<12,} "
                  f"{r['selfloops']:<12,} {r['multiedges']:<12,}")
    
    return results

def save_training_graphs_for_kernels():
    """Save graphs in kernel format using exact training data"""
    
    print(f"\n💾 SAVING TRAINING GRAPHS FOR KERNELS")
    print("=" * 50)
    
    datasets = ['reddit', 'flickr', 'yelp']  # Start with these 3
    
    import os
    output_dir = "./training_graphs/"
    os.makedirs(output_dir, exist_ok=True)
    
    for dataset in datasets:
        try:
            print(f"\n📊 Processing {dataset}...")
            g, data = check_dataset_exact_training(dataset)
            
            # Get CSR format
            indptr, indices, edge_ids = g.adj_sparse('csr')
            
            # Save as binary files (same format as kernel expects)
            indptr_file = f"{output_dir}{dataset}.indptr"
            indices_file = f"{output_dir}{dataset}.indices"
            
            indptr.cpu().numpy().astype('int32').tofile(indptr_file)
            indices.cpu().numpy().astype('int32').tofile(indices_file)
            
            print(f"   ✅ Saved: {indptr_file}")
            print(f"   ✅ Saved: {indices_file}")
            print(f"   Nodes: {g.num_nodes():,}")
            print(f"   Edges: {g.num_edges():,}")
            
        except Exception as e:
            print(f"   ❌ Failed: {e}")

if __name__ == "__main__":
    # Check all datasets using exact training code
    results = check_all_datasets_training()
    
    # Save graphs for kernel use
    save_training_graphs_for_kernels()
    
    print(f"\n🎯 NOW GENERATE METADATA USING THESE TRAINING GRAPHS!")
    print(f"   Use files in ./training_graphs/ instead of ./graphs/")
