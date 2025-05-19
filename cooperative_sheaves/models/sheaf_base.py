# Copyright 2022 Twitter, Inc.
# SPDX-License-Identifier: Apache-2.0

import torch
from torch import nn


class SheafDiffusion(nn.Module):
    """Base class for sheaf diffusion models."""

    def __init__(self, args):
        super(SheafDiffusion, self).__init__()

        assert args.d > 0
        self.d = args.d
        #self.edge_index = edge_index

        self.hidden_dim = args.hidden_channels * self.d
        self.device = args.device
        #self.graph_size = args.graph_size
        self.layers = args.layers
        self.input_dropout = args.input_dropout
        self.dropout = args.dropout
        self.left_weights = args.left_weights
        self.right_weights = args.right_weights
        self.use_act = args.use_act
        #self.use_bias = args.use_bias
        self.input_dim = args.input_dim
        self.hidden_channels = args.hidden_channels
        self.output_dim = args.output_dim
        self.layers = args.layers
        self.sheaf_act = args.sheaf_act
        self.orth_trans = args.orth
        #self.use_edge_weights = args.edge_weights
        self.gnn_layers = args.gnn_layers
        self.gnn_hidden = args.gnn_hidden
        self.pe_size = args.pe_size

    def update_edge_index(self, edge_index):
        assert edge_index.max() <= self.graph_size
        self.edge_index = edge_index
        self.laplacian_builder = self.laplacian_builder.create_with_new_edge_index(edge_index)

    def grouped_parameters(self):
        sheaf_learners, others = [], []
        for name, param in self.named_parameters():
            if "sheaf_learner" in name:
                sheaf_learners.append(param)
            else:
                others.append(param)
        assert len(sheaf_learners) > 0
        assert len(sheaf_learners) + len(others) == len(list(self.parameters()))
        return sheaf_learners, others
