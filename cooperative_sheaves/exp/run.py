#! /usr/bin/env python
# Copyright 2022 Twitter, Inc.
# SPDX-License-Identifier: Apache-2.0

import sys
import os
import random
import torch
import torch.nn.functional as F
import git
import numpy as np
import wandb
from tqdm import tqdm
import torch_geometric.transforms as T
from torch.optim.lr_scheduler import ReduceLROnPlateau
from sklearn.metrics import roc_auc_score
import torch.nn as nn

# This is required here by wandb sweeps.
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from exp.parser import get_parser
from models.coopshv_model import CoopSheafDiffusion
from utils.data_utils import load_data


def reset_wandb_env():
    exclude = {
        "WANDB_PROJECT",
        "WANDB_ENTITY",
        "WANDB_API_KEY",
    }
    for k, v in os.environ.items():
        if k.startswith("WANDB_") and k not in exclude:
            del os.environ[k]


def train(model, optimizer, data, fold):
    model.train()
    optimizer.zero_grad()
    #out = model(data.x)[data.train_mask]
    #out = model(data.x, data.laplacian_eigenvector_pe)[data.train_mask[fold]]
    out = model(data.x, data.random_walk_pe)[data.train_mask[fold]]
    if data.name in ['minesweeper', 'tolokers', 'questions']:
        out = out.sigmoid().squeeze(1)
        loss = F.binary_cross_entropy(out, data.y[data.train_mask[fold]].float())
    else:
        loss = F.cross_entropy(out, data.y[data.train_mask[fold]])
    loss.backward()

    optimizer.step()
    del out


def test(model, data, fold):
    model.eval()
    with torch.no_grad():
        logits, accs, losses, preds = model(data.x, data.random_walk_pe), [], [], []#model(data.x, data.laplacian_eigenvector_pe), [], [], []
        for _, mask in data('train_mask', 'val_mask', 'test_mask'):
            if data.name in ['minesweeper', 'tolokers', 'questions']:
                pred = logits[mask[fold]].sigmoid().squeeze(1)
                acc = roc_auc_score(data.y[mask[fold]].cpu().numpy(), pred.cpu().numpy()) #metric for minesweeper, questions and tolokers
                loss = F.binary_cross_entropy(pred, data.y[mask[fold]].float())
            else:
                pred = logits[mask[fold]].max(1)[1]
                acc = pred.eq(data.y[mask[fold]]).sum().item() / mask[fold].sum().item()
                loss = F.cross_entropy(logits[mask[fold]], data.y[mask[fold]])

            preds.append(pred.detach().cpu())
            accs.append(acc)
            losses.append(loss.detach().cpu())
        return accs, preds, losses


def run_exp(args, data, model_cls, fold):
    #torch.set_default_dtype(torch.float64)
    #data = get_fixed_splits(data, args['dataset'], fold)
    data = data.to(args['device'])
    data.name = args['dataset']

    model = model_cls(data.edge_index, args)
    model = model.to(args['device'])
    #model = model.to(torch.double)

    optimizer = torch.optim.Adam(model.parameters(), lr=args['lr'], weight_decay=args['weight_decay'])
    # scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, min_lr=1e-5,
    #                           patience=args['lr_decay_patience'])

    epoch = 0
    best_val_acc = test_acc = 0
    best_val_loss = float('inf')
    val_loss_history = []
    val_acc_history = []
    best_epoch = 0
    bad_counter = 0

    for epoch in range(args['epochs']):
        train(model, optimizer, data, fold)

        [train_acc, val_acc, tmp_test_acc], preds, [
            train_loss, val_loss, tmp_test_loss] = test(model, data, fold)
        if fold == 0:
            res_dict = {
                f'fold{fold}_train_acc': train_acc,
                f'fold{fold}_train_loss': train_loss,
                f'fold{fold}_val_acc': val_acc,
                f'fold{fold}_val_loss': val_loss,
                f'fold{fold}_tmp_test_acc': tmp_test_acc,
                f'fold{fold}_tmp_test_loss': tmp_test_loss,
            }
            wandb.log(res_dict, step=epoch)

        #scheduler.step(best_val_acc)
        new_best_trigger = val_acc > best_val_acc if args['stop_strategy'] == 'acc' else val_loss < best_val_loss
        if new_best_trigger:
            best_val_acc = val_acc
            best_val_loss = val_loss
            test_acc = tmp_test_acc
            best_epoch = epoch
            bad_counter = 0
        else:
            bad_counter += 1

        if bad_counter == args['early_stopping']:
            break

    print(f"Fold {fold} | Epochs: {epoch} | Best epoch: {best_epoch}")
    print(f"Test acc: {test_acc:.4f}")
    print(f"Best val acc: {best_val_acc:.4f}")

    wandb.log({'best_test_acc': test_acc, 'best_val_acc': best_val_acc, 'best_epoch': best_epoch})
    keep_running = False if test_acc < args['min_acc'] else True

    return test_acc, best_val_acc, keep_running


if __name__ == '__main__':
    parser = get_parser()
    args = parser.parse_args()

    repo = git.Repo(search_parent_directories=True)
    sha = repo.head.object.hexsha

    print(args.model)

    # if args.model == 'DiagSheafODE':
    #     model_cls = DiagSheafDiffusion
    # elif args.model == 'BundleSheafODE':
    #     model_cls = BundleSheafDiffusion
    # elif args.model == 'GeneralSheafODE':
    #     model_cls = GeneralSheafDiffusion
    # if args.model == 'DiagSheaf':
    #     model_cls = DiscreteDiagSheafDiffusion
    # elif args.model == 'BundleSheaf':
    #     model_cls = DiscreteBundleSheafDiffusion
    # elif args.model == 'GeneralSheaf':
    #     model_cls = DiscreteGeneralSheafDiffusion
    # elif args.model == 'HyperbolicSheaf':
    #     model_cls = HyperbolicSheafDiffusion
    # elif args.model == 'FlatBundle':
    #     model_cls = FlatBundleDiffusion
    if args.model == "CoopSheaf":
        model_cls = CoopSheafDiffusion
    else:
        raise ValueError(f'Unknown model {args.model}')

    dataset = load_data(args.dataset)

    # Add extra arguments
    args.sha = sha
    args.graph_size = dataset.x.size(0)
    args.input_dim = dataset.num_features
    classes = dataset.y.unique().size(0)
    args.output_dim = 1 if classes == 2 else classes 
    args.device = torch.device(f'cuda:{args.cuda}' if torch.cuda.is_available() else 'cpu')
    assert args.normalised or args.deg_normalised
    if args.sheaf_decay is None:
        args.sheaf_decay = args.weight_decay

    if args.pe_size == 0:
        dataset.random_walk_pe = torch.empty(args.graph_size, 0)
        dataset.laplacian_eigenvector_pe = torch.empty(args.graph_size, 0)
    else:
        pe = T.AddRandomWalkPE(args.pe_size)
        dataset = pe(dataset)
    print(dataset)

    # Set the seed for everything
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)

    results = []
    print(f"Running with wandb account: {args.entity}")
    print(args)
    wandb.init(project="cooperative_sheaf_no-scalar", config=vars(args), entity=args.entity)

    for fold in tqdm(range(args.folds)):
        test_acc, best_val_acc, keep_running = run_exp(wandb.config, dataset, model_cls, fold)
        results.append([test_acc, best_val_acc])
        if not keep_running:
            break

    test_acc_mean, val_acc_mean = np.mean(results, axis=0) * 100
    test_acc_std = np.sqrt(np.var(results, axis=0)[0]) * 100

    wandb_results = {'test_acc': test_acc_mean, 'val_acc': val_acc_mean, 'test_acc_std': test_acc_std}
    wandb.log(wandb_results)
    wandb.finish()

    model_name = args.model if args.evectors == 0 else f"{args.model}+LP{args.evectors}"
    print(f'{model_name} on {args.dataset} | SHA: {sha}')
    print(f'Test acc: {test_acc_mean:.4f} +/- {test_acc_std:.4f} | Val acc: {val_acc_mean:.4f}')