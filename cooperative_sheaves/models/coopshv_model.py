import torch
import torch_sparse
import numpy as np
import torch.nn.functional as F

from torch import nn
from torch_geometric.utils import get_laplacian
from torch_geometric.nn.models import GraphSAGE
from torch_scatter import scatter_add
from typing import Tuple
from cooperative_sheaves.models.sheaf_base import SheafDiffusion
from cooperative_sheaves.models.orthogonal import Orthogonal


class CoopSheafLayer(SheafDiffusion):

    def __init__(self, args):
        super(CoopSheafLayer, self).__init__(args)
        assert args.d > 1
        #assert not self.deg_normalised

        self.lin_right_weights = nn.Linear(self.hidden_channels, self.hidden_channels, bias=False)
        self.lin_left_weights = nn.Linear(self.d, self.d, bias=False)
        nn.init.orthogonal_(self.lin_right_weights.weight.data)
        nn.init.eye_(self.lin_left_weights.weight.data)

        self.epsilons = nn.Parameter(torch.zeros((self.d, 1)))

    def get_param_size(self):
        if self.orth_trans in ['matrix_exp', 'cayley']:
            return self.d * (self.d + 1) // 2
        else:
            return self.d * (self.d - 1) // 2

    def left_right_linear(self, x, left, right):
        if self.left_weights:
            x = x.t().reshape(-1, self.d)
            x = left(x)
            x = x.reshape(-1, self.graph_size * self.d).t()

        if self.right_weights:
            x = right(x)

        return x

    def diffusion(self, x, maps, L):
         # naive_x = torch.zeros(self.graph_size, self.hidden_channels, self.d, device=x.device)
        # for i in range(self.graph_size): #sanity check for einsum
        #     print(i)
        #     O_i = O[i]
        #     for h in range(self.hidden_channels):
        #         naive_x[i, h] = O_i @ x[i, h]
        x = torch.einsum('nij,nhj->nhi', maps, x)
        #print(torch.allclose(naive_x, x, atol=1e-6))
        x = x.reshape(self.graph_size * self.d, -1)

        x = torch_sparse.spmm(L[0],
                              L[1],
                              x.size(0),
                              x.size(0),
                              x)
        
        x = x.view(self.graph_size, self.hidden_channels, self.d)
        x = torch.einsum('nij,nhj->nhi', maps.transpose(-2,-1), x)

        return x
    
    def forward(self, x, L_in, L_out, idx):
        self.graph_size = x.size(0)
        x = x.view(self.graph_size * self.d, -1)
        x0 = x
        x = self.left_right_linear(x, self.lin_left_weights, self.lin_right_weights)

        x = torch_sparse.spmm(idx, L_out, x.size(0), x.size(0), x)
        x = torch_sparse.spmm(idx, L_in, x.size(0), x.size(0), x)

        x = F.dropout(x, p=self.dropout, training=self.training)

        if self.use_act:
            x = F.gelu(x)

        x = x.reshape(self.graph_size * self.d, -1)
        x0 = (1 + torch.tanh(self.epsilons).tile(self.graph_size, 1)) * x0 - x
        x = x0

        return x.view(self.graph_size, self.hidden_channels, self.d)

class CoopSheafDiffusion(SheafDiffusion):
    def __init__(self, args):
        super(CoopSheafDiffusion, self).__init__(args)
        assert args.d > 1
        #assert not self.deg_normalised

        self.orth_transform = Orthogonal(d=self.d, orthogonal_map=self.orth_trans)

        self.in_maps_learner = nn.ModuleList(
            [FlatBundleLearnerVariant(
                self.d,
                self.hidden_channels,
                out_shape=(self.get_param_size(),),
                gnn_layers=self.gnn_layers,
                gnn_hidden=self.gnn_hidden,
                pe_size=self.pe_size,
                sheaf_act=self.sheaf_act) for _ in range(self.layers)] 
        )

        self.out_maps_learner = nn.ModuleList(
            [FlatBundleLearnerVariant(
                self.d,
                self.hidden_channels,
                out_shape=(self.get_param_size(),),
                gnn_layers=self.gnn_layers,
                gnn_hidden=self.gnn_hidden,
                pe_size=self.pe_size,
                sheaf_act=self.sheaf_act) for _ in range(self.layers)] 
        )
        
        #self.lin1 = nn.Linear(self.input_dim, self.hidden_channels * self.d)
        #self.lin2 = nn.Linear(self.hidden_channels * self.d, self.output_dim)

        self.coop_layers = nn.ModuleList(
            [CoopSheafLayer(args) for _ in range(self.layers)]
        )

        # self.layer_norms = nn.ModuleList(
        #     [nn.LayerNorm([self.hidden_channels, self.d]) for _ in range(self.layers)]
        # )

        #self.edge_weight = torch.nn.Linear(self.num_bundles * self.d * self.hidden_channels * 2, 1, bias=False)

        # self.laplacian = get_laplacian(
        #     self.edge_index,
        #     normalization='rw')

    def get_param_size(self):
        if self.orth_trans in ['matrix_exp', 'cayley']:
            return self.d * (self.d + 1) // 2
        else:
            return self.d * (self.d - 1) // 2
           
    def normalise(self, diag, tril, row, col):
        if tril.dim() > 2:
            assert tril.size(-1) == tril.size(-2)
            assert diag.dim() == 2
        d = diag.size(-1)

        diag_sqrt_inv = (diag + 1).pow(-0.5)
        diag_sqrt_inv = diag_sqrt_inv.view(-1, 1, 1) if tril.dim() > 2 else diag_sqrt_inv.view(-1, d)
        left_norm = diag_sqrt_inv[row]
        right_norm = diag_sqrt_inv[col]
        non_diag_maps = left_norm * tril * right_norm

        diag_sqrt_inv = diag_sqrt_inv.view(-1, 1, 1) if diag.dim() > 2 else diag_sqrt_inv.view(-1, d)
        diag_maps = diag_sqrt_inv**2 * diag

        return diag_maps, non_diag_maps
    
    def get_laplacian_indices(self):
        row, col = self.edge_index

        row_expand = row.view(-1, 1, 1) * self.d + torch.arange(self.d, device=self.device).view(1, 1, -1)
        row_expand = row_expand.expand(-1, self.d, -1)

        col_expand = col.view(-1, 1, 1) * self.d + torch.arange(self.d, device=self.device).view(1, -1, 1)
        col_expand = col_expand.expand(-1, -1, self.d)

        row_indices = row_expand.reshape(-1)
        col_indices = col_expand.reshape(-1)
        off_diag_indices = torch.stack([row_indices, col_indices], dim=0)

        arange_d = torch.arange(self.d, device=self.device)
        base = torch.arange(self.graph_size, device=self.device) * self.d

        diag_i = base.view(-1, 1, 1) + arange_d.view(1, 1, self.d).expand(self.graph_size, self.d, self.d)
        diag_j = base.view(-1, 1, 1) + arange_d.view(1, self.d, 1).expand(self.graph_size, self.d, self.d)
        diag_i = diag_i.reshape(-1)
        diag_j = diag_j.reshape(-1)

        diag_indices = torch.stack([diag_i, diag_j], dim=0)

        return diag_indices, off_diag_indices

    def laplacian_builder(self, S, T, c_S, c_T):
        row, col = self.edge_index
        
        S = self.orth_transform(S) * c_S[:, None, None]
        T = self.orth_transform(T) * c_T[:, None, None]
        S_maps, T_maps = S[row], T[col]

        off_diag_maps = -torch.bmm(T_maps.transpose(-2,-1), S_maps)
        in_diag_maps = scatter_add(c_T[col] ** 2, col, dim=0, dim_size=self.graph_size)[:, None]
        out_diag_maps = scatter_add(c_S[row] ** 2, row, dim=0, dim_size=self.graph_size)[:, None]

        in_diag_maps, in_off_diag_maps = self.normalise(in_diag_maps, off_diag_maps, col, row)
        out_diag_maps, out_off_diag_maps = self.normalise(out_diag_maps, off_diag_maps, row, col)

        eye = torch.eye(self.d, device=self.device).unsqueeze(0)
        in_diag_maps = (in_diag_maps.expand(-1, self.d).unsqueeze(-1) * eye).view(-1)
        in_off_diag_maps = in_off_diag_maps.view(-1)
        out_diag_maps = (out_diag_maps.expand(-1, self.d).unsqueeze(-1) * eye).view(-1)
        out_off_diag_maps = out_off_diag_maps.view(-1)

        diag_indices, off_diag_indices = self.get_laplacian_indices()

        L_in_values = torch.cat([in_off_diag_maps, in_diag_maps], dim=0)
        L_out_values = torch.cat([out_off_diag_maps, out_diag_maps], dim=0)
        L_idx = torch.cat([off_diag_indices, diag_indices], dim=1)

        return (L_in_values, L_out_values, L_idx)


    def forward(self, x, edge_index):
        self.graph_size = x.size(0)
        pe = torch.empty(self.graph_size, 0, device=x.device)
        self.edge_index = edge_index
        x = F.dropout(x, p=self.input_dropout, training=self.training)
        #x = self.lin1(x)

        # if self.use_act:
        #     x = F.gelu(x)

        #x = F.dropout(x, p=self.dropout, training=self.training)
        for in_maps, out_maps, coop_layer in zip(self.in_maps_learner,
                                                 self.out_maps_learner,
                                                 self.coop_layers):
                                                 #self.layer_norms):
            x = F.dropout(x, p=self.dropout, training=self.training)
            x_maps = x.reshape(self.graph_size, self.hidden_dim)
            to_be_S_maps, c_S = out_maps(x_maps, self.edge_index, pe)
            to_be_T_maps, c_T = in_maps(x_maps, self.edge_index, pe)#, edge_weights)
            L_in, L_out, L_idx = self.laplacian_builder(to_be_S_maps, to_be_T_maps, c_S, c_T)

            #x = F.dropout(x, p=self.dropout, training=self.training)
            x = coop_layer(x, L_in, L_out, L_idx)
            # if self.training:
            #     print("S_c min:", c_S.min())
            #     print("S_c max:", c_S.max())
            #     print("T_c min:", c_T.min())
            #     print("T_c max:", c_T.max())
            #     print('x min:', x.min())
            #     print('x max:', x.max())
        
        x = x.reshape(self.graph_size, -1)
        # x = self.lin2(x)
        #x = F.gelu(x)

        return x

class FlatBundleLearnerVariant(nn.Module):
    """Learns a bundle passing node features through a GNN or MLP + activation."""

    def __init__(self, d: int,  hidden_channels: int, out_shape: Tuple[int], gnn_layers: int, gnn_hidden: int, pe_size: int, sheaf_act="tanh"):
        super(FlatBundleLearnerVariant, self).__init__()
        assert len(out_shape) in [1, 2]
        self.out_shape = out_shape
        self.d = d
        self.hidden_channels = hidden_channels
        self.gnn_layers = gnn_layers

        if gnn_layers > 0:
            self.phi = GraphSAGE(
                hidden_channels * d + pe_size,
                gnn_hidden,
                num_layers=gnn_layers,
                out_channels=int(np.prod(out_shape) + 1),
                # dropout=0.2,
                # act='gelu',
                # project=True,
                # bias=True,
                )
        else:
            self.phi = nn.Sequential(
                nn.Linear(hidden_channels * d + pe_size, gnn_hidden),
                nn.ReLU(),
                nn.Linear(gnn_hidden, int(np.prod(out_shape)) + 1)
            )

        #self.lin = nn.Linear(hidden_channels * d + pe_size, int(np.prod(out_shape)) + 1)

    def forward(self, x, edge_index, pe, edge_weight=None):
        pe = pe.to(x.device)
        feat = torch.cat([x, pe], -1)
        
        if self.gnn_layers > 0:
            maps = self.phi(feat, edge_index, edge_weight=edge_weight)
        else:
            maps = self.phi(feat)

        maps = maps.view(-1, int(np.prod(self.out_shape))+1)
        c = maps[:, -1]

        return torch.tanh(maps[:, :-1]), torch.exp(c).clamp_max(10)