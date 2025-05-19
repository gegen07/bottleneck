import torch
import torch.nn.functional as F
from torch_geometric.data import DataLoader
from tqdm import tqdm
from models.coopshv_model import CoopSheafDiffusion
from utils.data_utils import load_synthetic_data
from exp.parser import get_parser
import git
import wandb
import numpy as np
import random

from torch_geometric.nn.models import MLP, GCN, GraphSAGE, GAT

def train(model, optimizer, loader):
    model.train()
    total_loss = 0
    for data in loader:
        data = data.to(args.device)
        optimizer.zero_grad()
        #out = model(data.x, data.random_walk_pe, data.edge_index)
        out = model(data.x, data.edge_index)
        loss = F.mse_loss(out, data.y)  # MSE loss
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    return total_loss / len(loader)

def test(model, loader):
    model.eval()
    total_loss = 0
    with torch.no_grad():
        for data in loader:
            data = data.to(args.device)
            #out = model(data.x, data.random_walk_pe, data.edge_index)
            out = model(data.x, data.edge_index)
            loss = F.mse_loss(out, data.y)
            total_loss += loss.item()
    return total_loss / len(loader)

def run_exp(args, model_cls, train_data, test_data, fold):
    if model_cls.__name__ == 'CoopSheafDiffusion':
        model = model_cls(None, args)
    else:
        model = model_cls(in_channels=args['input_dim'],
                    hidden_channels=args['hidden_channels'],
                    out_channels=args['output_dim'],
                    num_layers=args['layers'])
    model = model.to(args.device)

    optimizer = torch.optim.Adam(model.parameters(), lr=args['lr'], weight_decay=args['weight_decay'])

    epoch = 0
    best_train_loss = float('inf')
    best_epoch = 0
    bad_counter = 0

    for epoch in range(args['epochs']):
        train_loss = train(model, optimizer, train_data)
        tmp_test_loss = test(model, test_data)
        if fold == 0:
            res_dict = {
                f'fold{fold}_train_loss': train_loss,
                f'fold{fold}_test_loss': tmp_test_loss,
            }
            wandb.log(res_dict, step=epoch)

        #scheduler.step(best_val_acc)
        new_best_trigger = train_loss < best_train_loss
        if new_best_trigger:
            best_train_loss = train_loss
            best_test_loss = tmp_test_loss
            best_epoch = epoch
            bad_counter = 0
        else:
            bad_counter += 1

        if bad_counter == args['early_stopping']:
            break

    print(f"Fold {fold} | Epochs: {epoch} | Best epoch: {best_epoch}")
    print(f"Best train loss: {best_train_loss:.4f} | Best test loss: {best_test_loss:.4f}")

    wandb.log({'best_train_loss': best_train_loss, 'best_test_loss': best_test_loss, 'best_epoch': best_epoch})

    return best_train_loss, best_test_loss

def compute_baseline2_mse(loader, N=24):
    sqrt_3N = (3 * N)**0.5
    expected_left = -sqrt_3N / 2
    expected_right = sqrt_3N / 2

    mse_list = []

    for data in loader:
        num_nodes = data.num_nodes
        half = num_nodes // 2
        size = data.y.shape[1]

        pred = torch.zeros_like(data.y)
        pred[:half] = expected_right * torch.ones(size)
        pred[half:] = expected_left * torch.ones(size)

        mse = F.mse_loss(pred, data.y)
        mse_list.append(mse.item())

    return np.mean(mse_list), np.std(mse_list)

if __name__ == '__main__':
    parser = get_parser()
    args = parser.parse_args()

    repo = git.Repo(search_parent_directories=True)
    sha = repo.head.object.hexsha

    print(args.model)

    if args.model == 'GCN':
        model_cls = GCN
    elif args.model == 'SAGE':
        model_cls = GraphSAGE
    elif args.model == 'GAT':
        model_cls = GAT
    elif args.model == 'MLP':
        model_cls = MLP
    elif args.model == 'CoopSheaf':
        model_cls = CoopSheafDiffusion
    else:
        raise ValueError(f"Unknown model: {args.model}")

    train_graphs = load_synthetic_data(args.dataset)
    test_graphs = load_synthetic_data(args.dataset)

    means = []
    stds = []

    # Create data loaders
    train_loader = DataLoader(train_graphs, batch_size=1, shuffle=False)
    test_loader = DataLoader(test_graphs, batch_size=1, shuffle=False)

    # Add extra arguments
    args.sha = sha
    args.graph_size = train_graphs[0].x.size(0)
    args.input_dim = train_graphs[0].num_features
    classes = train_graphs[0].y.unique().size(0)
    args.output_dim = train_graphs[0].y.size(1) 
    args.device = torch.device(f'cuda:{args.cuda}' if torch.cuda.is_available() else 'cpu')
    assert args.normalised or args.deg_normalised
    if args.sheaf_decay is None:
        args.sheaf_decay = args.weight_decay

    # Set the seed for everything
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)

    results = []
    print(f"Running with wandb account: {args.entity}")
    print(args)
    wandb.init(project="cooperative_sheaf_oversquashing", config=vars(args), entity=args.entity)

    for fold in tqdm(range(args.folds)):
        mean,std = compute_baseline2_mse(test_graphs)
        means.append(mean)
        stds.append(std)
        print(f"Baseline MSE: {mean:.4f} +/- {std:.4f}")
        train_loss, best_test_loss = run_exp(wandb.config, model_cls, train_loader, test_loader, fold)
        results.append([train_loss, best_test_loss])
        train_graphs = load_synthetic_data(args.dataset)
        test_graphs = load_synthetic_data(args.dataset)

        train_loader = DataLoader(train_graphs, batch_size=1, shuffle=False)
        test_loader = DataLoader(test_graphs, batch_size=1, shuffle=False)

    print(f"Baseline MSE: {np.mean(means):.4f} +/- {np.std(means):.4f}")
    train_loss_mean, test_loss_mean = np.mean(results, axis=0)
    test_loss_std = np.sqrt(np.var(results, axis=0)[1])

    wandb_results = {'train_loss': train_loss_mean, 'test_loss': test_loss_mean, 'test_loss_std': test_loss_std}
    wandb.log(wandb_results)
    wandb.finish()

    model_name = args.model if args.evectors == 0 else f"{args.model}+LP{args.evectors}"
    print(f'{model_name} on {args.dataset} | SHA: {sha}')
    print(f'Test loss: {test_loss_mean:.4f} +/- {test_loss_std:.4f} | Train loss: {train_loss_mean:.4f}')