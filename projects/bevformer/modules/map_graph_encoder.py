import copy

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from mmcv import ConfigDict
from mmcv.runner.base_module import BaseModule, ModuleList
from mmcv.cnn.bricks.registry import FEEDFORWARD_NETWORK
from mmcv.cnn.bricks.transformer import POSITIONAL_ENCODING
from mmcv.cnn.bricks.transformer import build_feedforward_network, build_positional_encoding
from mmdet3d.models import NECKS, BACKBONES
from mmdet3d.models.builder import build_backbone
@POSITIONAL_ENCODING.register_module()
class SineContinuousPositionalEncoding(BaseModule):
    def __init__(self, 
                 num_feats,
                 temperature=10000,
                 normalize=False,
                 range=None,
                 scale=2 * np.pi,
                 offset=0.,
                 init_cfg=None):
        super(SineContinuousPositionalEncoding, self).__init__(init_cfg)
        self.num_feats = num_feats
        self.temperature = temperature
        self.normalize = normalize
        self.range = torch.tensor(range) if range is not None else None
        self.offset = torch.tensor(offset) if offset is not None else None
        self.scale = scale
    
    def forward(self, x):
        """
        x: [B, N, D]

        return: [B, N, D * num_feats]
        """
        B, N, D = x.shape
        if self.normalize:
            x = (x - self.offset.to(x.device)) / self.range.to(x.device) * self.scale
        dim_t = torch.arange(self.num_feats, dtype=torch.float32, device=x.device)
        dim_t = self.temperature**(2 * (dim_t // 2) / self.num_feats)
        pos_x = x[..., None] / dim_t  # [B, N, D, num_feats]
        pos_x = torch.stack(
            (pos_x[:, :, :, 0::2].sin(), pos_x[:, :, :, 1::2].cos()),
            dim=3).view(B, N, D * self.num_feats)
        return pos_x


@NECKS.register_module()
class MapGraphTransformer(BaseModule):
    def __init__(self, 
                 input_dim=30,  # 2 * 11 points + 8 classes
                 dmodel=256,
                 hidden_dim=256,  # set this to something smaller
                 nheads=8,
                 nlayers=6,
                 num_point_per_vec=11,
                 pts_dims=32,      #2*16(pos_encoder)
                 num_query=200,
                 num_layers=2,
                 batch_first=False,  # set to True
                 pos_encoder=None,
                 **kwargs):
        super(MapGraphTransformer, self).__init__(**kwargs)
        self.batch_dim = 0 if batch_first else 1
        self.map_embedding = nn.Linear(input_dim, 128)
        self.num_query = num_query
        if pos_encoder is not None:
            self.use_positional_encoding = True
            self.pos_encoder = build_positional_encoding(pos_encoder)
        else:
            self.use_positional_encoding = False
        
        encoder_layer = nn.TransformerEncoderLayer(d_model=dmodel, 
                                                        nhead=nheads,
                                                        dim_feedforward=hidden_dim,
                                                        batch_first=batch_first)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=nlayers)
        self.num_point_per_vec = num_point_per_vec
        self.pts_dims = pts_dims
        self.graph_pos = nn.Linear(num_point_per_vec*pts_dims, 128)
        self.embed_dims = dmodel
        self.num_layers = num_layers
        self.init_layers()
    def init_layers(self):
        """Initialize layers of the Detr3DTransformer."""
        proj_prior = []
        for _ in range(self.num_layers - 1):
                proj_prior.append(nn.Sequential(
                    nn.Linear(self.num_point_per_vec * self.pts_dims, 2 * self.embed_dims),
                    nn.ReLU(),
                    nn.Dropout(0.1)))
        proj_prior.append(nn.Linear(2*self.embed_dims, self.embed_dims))
        self.proj_prior = nn.Sequential(*proj_prior)
    # def init_weights(self):
    #     """Initialize the transformer weights."""
       
    #     xavier_init(self.proj_prior, distribution='uniform', bias=0
    def forward(self, map_graph, onehot_category):
        # batch_map_graph: list of polylines
        # batch, num_polylines * points * 2 (x, y)
        # onehot_category: batch, num_polylines * num_categories, onehot encoding of categories
        # TODO: make batched
        batch_graph_feats = []
        batch_polylines_pos = []
        for graph_polylines, onehot_cat in zip(map_graph, onehot_category):
            #position encoding，instance position
            if self.use_positional_encoding:
                #[num_polylines,num_points,32]
                graph_polylines = self.pos_encoder(graph_polylines)
            npolylines, npoints, pdim = graph_polylines.shape
            graph_polylines = graph_polylines.view(npolylines, npoints*pdim)
            polylines_pos = self.proj_prior(graph_polylines)
            batch_polylines_pos.append(polylines_pos)
            
            
            instance_pos = self.graph_pos(graph_polylines)
            if onehot_cat.shape[1] == 0:
                graph_feat = graph_polylines.view(npolylines, npoints * pdim)
            else:
                graph_feat = torch.cat([graph_polylines.view(npolylines, npoints * pdim), onehot_cat], dim=-1)

            # embed features
            graph_feat = self.map_embedding(graph_feat)  # num_polylines, dmodel // 2
            graph_feat = torch.cat([graph_feat, instance_pos], dim=-1)
            # transformer encoder
            graph_feat = self.transformer_encoder(graph_feat.unsqueeze(self.batch_dim))  # 1, num_polylines, hidden_dim
            # graph_feat = graph_feat.squeeze(self.batch_dim)  # num_polylines, hidden_dim
            batch_graph_feats.append(graph_feat.squeeze(self.batch_dim))
        
        prior_pos = torch.stack([
            torch.cat([polyline_pos[:self.num_query],  polyline_pos.new_zeros(self.num_query - polyline_pos.shape[0], polyline_pos.shape[1])], dim=0)
            if polyline_pos.shape[0] < self.num_query else polyline_pos[:self.num_query]   
            for polyline_pos in batch_polylines_pos], dim=0)
        print(prior_pos.shape)
        return batch_graph_feats, prior_pos
