import argparse

import dgl
import dgl.nn as dglnn
from tensorboardX import SummaryWriter

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Linear
import torch.nn.init as init
from torch.autograd import Function
import math

from dgl import AddSelfLoop
from dgl.data import CiteseerGraphDataset, CoraGraphDataset, PubmedGraphDataset
from dgl.data import RedditDataset, FlickrDataset, YelpDataset
from utils.config import TrainConfig
import os
import utils.general_utils as general_utils
# from ogb.graphproppred import DglGraphPropPredDataset
from ogb.nodeproppred import DglNodePropPredDataset, Evaluator

from utils.proteins_loader import load_proteins
from utils.models import SAGE, GCN, GIN, GNN_res


# The flag below controls whether to allow TF32 on matmul. This flag defaults to False
# in PyTorch 1.12 and later.
torch.backends.cuda.matmul.allow_tf32 = True

# The flag below controls whether to allow TF32 on cuDNN. This flag defaults to True.
torch.backends.cudnn.allow_tf32 = True

pid = os.getpid()
print("Current process ID:", pid)


def evaluate(g, features, labels, mask, model, config, logger):
    model.eval()
    if config.dataset == 'ogbn-proteins':
        evaluator = Evaluator(name='ogbn-proteins')
        evaluator_wrapper = lambda pred, labels: evaluator.eval(
            {"y_pred": pred, "y_true": labels}
        )["rocauc"]
    with torch.no_grad():        
        logits = model(g, features)
        if config.dataset != 'ogbn-proteins':
            # return general_utils.accuracy(logits[mask], labels[mask])[0]
            return general_utils.compute_micro_f1(logits, labels, mask)
        else:
            return evaluator_wrapper(logits[mask], labels[mask])


def evaluate_masks(g, features, labels, masks, model, config, logger):
    model.eval()
    train_mask = masks[0]
    val_mask = masks[1]
    test_mask = masks[2]
    if config.dataset == 'ogbn-proteins':
        evaluator = Evaluator(name='ogbn-proteins')
        evaluator_wrapper = lambda pred, labels: evaluator.eval(
            {"y_pred": pred, "y_true": labels}
        )["rocauc"]
    with torch.no_grad():
        logits = model(g, features)
        if config.dataset != 'ogbn-proteins':
            train_acc = general_utils.compute_micro_f1(logits, labels, train_mask)
            val_acc = general_utils.compute_micro_f1(logits, labels, val_mask)
            test_acc = general_utils.compute_micro_f1(logits, labels, test_mask)
        else:
            train_acc = evaluator_wrapper(logits[train_mask], labels[train_mask])
            val_acc = evaluator_wrapper(logits[val_mask], labels[val_mask])
            test_acc = evaluator_wrapper(logits[test_mask], labels[test_mask])
        return train_acc, val_acc, test_acc

def train(g, features, labels, masks, model, config, logger, writer):
    # define train/val samples, loss function and optimizer
    train_mask = masks[0]
    if (config.dataset != 'yelp') and (config.dataset != 'ogbn-proteins'):
        loss_fcn = nn.CrossEntropyLoss()
    else:
        loss_fcn = F.binary_cross_entropy_with_logits
    optimizer = torch.optim.Adam(model.parameters(), lr=config.w_lr, weight_decay=config.w_weight_decay)
    if config.enable_lookahead: 
        optimizer = general_utils.Lookahead(optimizer)
    
    # Timing variables for performance analysis
    total_forward_time = 0.0
    total_backward_time = 0.0
    num_epochs_timed = 0
    warmup_epochs = 10  # Skip first few epochs for warmup
    
    # training loop
    best_val_accuracy = 0
    best_test_accuracy = 0
    for epoch in range(config.epochs):
        # torch.cuda.empty_cache()
        model.train()
        
        # Time forward pass (skip warmup epochs)
        if epoch >= warmup_epochs:
            torch.cuda.synchronize()
            forward_start = torch.cuda.Event(enable_timing=True)
            forward_end = torch.cuda.Event(enable_timing=True)
            forward_start.record()
        
        logits = model(g, features)
        loss = loss_fcn(logits[train_mask], labels[train_mask])
        
        if epoch >= warmup_epochs:
            forward_end.record()
            torch.cuda.synchronize()
            forward_time = forward_start.elapsed_time(forward_end)
            total_forward_time += forward_time
        
        optimizer.zero_grad()
        
        # Time backward pass (skip warmup epochs)
        if epoch >= warmup_epochs:
            torch.cuda.synchronize()
            backward_start = torch.cuda.Event(enable_timing=True)
            backward_end = torch.cuda.Event(enable_timing=True)
            backward_start.record()
        
        loss.backward()
        
        if epoch >= warmup_epochs:
            backward_end.record()
            torch.cuda.synchronize()
            backward_time = backward_start.elapsed_time(backward_end)
            total_backward_time += backward_time
            num_epochs_timed += 1
        
        optimizer.step()
        # torch.cuda.empty_cache()

        train_acc, val_acc, test_acc = evaluate_masks(g, features, labels, masks, model, config, logger)
        if val_acc > best_val_accuracy:
            best_val_accuracy = val_acc
            best_test_accuracy = test_acc
        writer.add_scalar('train/loss', loss.item(), epoch)
        writer.add_scalar('train/train_acc', train_acc, epoch)
        writer.add_scalar('train/val_acc', val_acc, epoch)
        writer.add_scalar('train/test_acc', test_acc, epoch)
        
        # Log timing information periodically (every 100 epochs after warmup)
        if epoch >= warmup_epochs and epoch % 100 == 0 and num_epochs_timed > 0:
            avg_forward = total_forward_time / num_epochs_timed
            avg_backward = total_backward_time / num_epochs_timed
            avg_total = avg_forward + avg_backward
            logger.info(f"Epoch {epoch:04d}: Avg Forward {avg_forward:.3f}ms, Avg Backward {avg_backward:.3f}ms, Total per epoch {avg_total:.3f}ms")
        
        logger.info(
            "Epoch {:04d}/{:04d}| Loss {:.4f} | Train Accuracy {:.4f} | Val Accuracy {:.4f} | Test Accuracy {:.4f} | Best val. Accuracy {:.4f} | Best test Accuracy {:.4f}".format(
                epoch, config.epochs, loss.item(), train_acc, val_acc, test_acc, best_val_accuracy, best_test_accuracy
            )
        )
    
    # Final timing report
    if num_epochs_timed > 0:
        avg_forward = total_forward_time / num_epochs_timed
        avg_backward = total_backward_time / num_epochs_timed
        avg_total = avg_forward + avg_backward
        logger.info("=" * 60)
        logger.info("ORIGINAL DGL TRAINING - FINAL TIMING REPORT")
        logger.info("=" * 60)
        logger.info(f"Average Forward Pass:  {avg_forward:.3f}ms")
        logger.info(f"Average Backward Pass: {avg_backward:.3f}ms")
        logger.info(f"Total per epoch:       {avg_total:.3f}ms")
        logger.info(f"Epochs measured:       {num_epochs_timed}")
        logger.info(f"Forward/Backward ratio: {avg_forward/avg_backward:.2f}")
        logger.info("=" * 60)
        
        # Also log to tensorboard
        writer.add_scalar('timing/avg_forward_ms', avg_forward, config.epochs)
        writer.add_scalar('timing/avg_backward_ms', avg_backward, config.epochs)
        writer.add_scalar('timing/total_per_epoch_ms', avg_total, config.epochs)


if __name__ == "__main__":

    config = TrainConfig()

    torch.manual_seed(config.seed)
    torch.cuda.manual_seed_all(config.seed)

    # tensorboard
    writer = SummaryWriter(log_dir=os.path.join(config.path, "tb"))
    writer.add_text('config', config.as_markdown(), 0)

    logger = general_utils.get_logger(os.path.join(config.path, "{}.log".format(config.dataset)))
    config.print_params(logger.info)

    torch.cuda.set_device(config.gpu)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    logger.info(f"Training with DGL built-in GraphConv module.")

    
    if "ogb" not in config.dataset:
        # load and preprocess dataset
        transform = (
            AddSelfLoop()
        )  # by default, it will first remove self-loops to prevent duplication
        if config.dataset == 'reddit':
            data = RedditDataset(transform=transform, raw_dir= config.data_path)
        elif config.dataset == 'flickr':
            data = FlickrDataset(transform=transform, raw_dir= config.data_path)
        elif config.dataset == 'yelp':
            data = YelpDataset(transform=transform, raw_dir= config.data_path)
        g = data[0]
        g = g.int().to(device)
        features = g.ndata["feat"]
        if config.dataset == 'yelp':
            labels = g.ndata["label"].float()#.float()
        else:
            labels = g.ndata["label"]
        masks = g.ndata["train_mask"].bool(), g.ndata["val_mask"].bool(), g.ndata["test_mask"].bool()
    elif "proteins" not in config.dataset:
        data = DglNodePropPredDataset(name=config.dataset, root = config.data_path)
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
        # masks = split_idx["train"].bool().to(device), split_idx["valid"].bool().to(device), split_idx["test"].bool().to(device)
        masks = train_mask_bin.to(device), valid_mask_bin.to(device), test_mask_bin.to(device)
        # print(split_idx["train"])
        # print(split_idx["train"].shape)
        # print(masks[1])
        # print(masks[1].shape)
        # print(masks[2])
        # print(masks[2].shape)
    ##### ogbn_proteins loader
    else:
        data, g, labels, train_idx, val_idx, test_idx = load_proteins(config.data_path)
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
    if config.dataset == 'ogbn-proteins':
        out_size = 112
    if config.selfloop:
        g = dgl.add_self_loop(g)

    if config.model == 'sage':
        model = SAGE(in_size, config.hidden_dim, config.hidden_layers, out_size, config.maxk, feat_drop=config.dropout, norm=config.norm, nonlinear = config.nonlinear).to(device)
    elif config.model == 'gcn':
        model = GCN(in_size, config.hidden_dim, config.hidden_layers, out_size, config.maxk, feat_drop=config.dropout, norm=config.norm, nonlinear = config.nonlinear).to(device)
    elif config.model == 'gin':
        model = GIN(in_size, config.hidden_dim, config.hidden_layers, out_size, config.maxk, feat_drop=config.dropout, norm=config.norm, nonlinear = config.nonlinear).to(device)
    elif config.model == 'gnn_res':
        model = GNN_res(in_size, config.hidden_dim, config.hidden_layers, out_size, config.maxk, feat_drop=config.dropout, norm=config.norm, nonlinear = config.nonlinear).to(device)

    # Print model and dataset information for timing context
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"Model parameters: {total_params:,} total, {trainable_params:,} trainable")
    logger.info(f"Graph statistics: {g.num_nodes()} nodes, {g.num_edges()} edges")
    logger.info(f"Average degree: {g.num_edges() / g.num_nodes():.2f}")
    logger.info(f"Feature dimension: {in_size}, Hidden dimension: {config.hidden_dim}")
    logger.info(f"MaxK value: {config.maxk}, Nonlinearity: {config.nonlinear}")

    # model training
    logger.info("Training...")
    train(g, features, labels, masks, model, config, logger, writer)

    # test the model
    logger.info("Testing...")
    acc = evaluate(g, features, labels, masks[2], model, config, logger)
    logger.info("Test accuracy {:.4f}".format(acc))