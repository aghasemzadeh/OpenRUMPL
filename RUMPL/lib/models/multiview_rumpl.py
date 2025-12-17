# ------------------------------------------------------------------------------
# Copyright (c) 2024 UCLouvain. All rights reserved.
# Licensed under the GNU Affero General Public License v3.0 (AGPLv3).
#
# Author: Seyed Abolfazl Ghaemzadeh, ICTEAM, UCLouvain
# ------------------------------------------------------------------------------

# ------------------------------------------------------------------------------
# Modified by Yanjie Li (leeyegy@gmail.com)
# TokenPose + Sparse for 2D single person PE
# Multi-view
# cross-view Fusion
# ------------------------------------------------------------------------------
## Our PoseFormer model was revised from https://github.com/rwightman/pytorch-image-models/blob/master/timm/models/vision_transformer.py
# Written by Ce Zheng (cezheng@knights.ucf.edu)
# Modified by Qitao Zhao (qitaozhao@mail.sdu.edu.cn)
# ------------------------------------------------------------------------------

import math
import logging
from functools import partial
from collections import OrderedDict
from einops import rearrange, repeat

import torch
import torch.nn as nn
import torch.nn.functional as F

from timm.data import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD
from timm.models.helpers import load_pretrained
from timm.models.layers import DropPath, to_2tuple, trunc_normal_
from timm.models.registry import register_model
import os
from torchinfo import summary

logger = logging.getLogger(__name__)

class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class Attention(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0., learned_query=False, num_tokens=2):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        # NOTE scale factor was wrong in my original version, can set manually to be compat with prev weights
        self.scale = qk_scale or head_dim ** -0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)
        self.learned_query = learned_query
        if self.learned_query:
            self.Q_learned = torch.nn.Parameter(torch.randn(self.num_heads, num_tokens, dim // self.num_heads), requires_grad=learned_query)

    def forward(self, x, conf_weights=None):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]   # make torchscript happy (cannot use tensor as tuple)
        if self.learned_query:
            q = self.Q_learned.expand(B, -1, -1, -1)

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)
        if conf_weights is not None:
            attn = attn * conf_weights.unsqueeze(1)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class Block(nn.Module):

    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm, learned_query=False, num_tokens=2):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = Attention(
            dim, num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=drop, learned_query=learned_query, num_tokens=num_tokens)
        # NOTE: drop path for stochastic depth, we shall see if this is better than dropout here
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

    def forward(self, x, conf_weights=None):
        
        if conf_weights is not None:
            attn_output = self.attn(self.norm1(x), conf_weights)
            x = x + self.drop_path(attn_output)
        else:
            x = x + self.drop_path(self.attn(self.norm1(x)))
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x

class MultiView_RUMPL(nn.Module):
    def __init__(self, num_joints=17, in_chans=3, embed_dim_ratio=32, depth=4,
                 num_heads=8, mlp_ratio=2., qkv_bias=True, qk_scale=None,
                 drop_rate=0., attn_drop_rate=0., drop_path_rate=0.2,  norm_layer=None, num_views=5,
                 linear_weighted_mean=False,
                 hidden_dim=1024,
                 cfg=None,):
        """    ##########hybrid_backbone=None, representation_size=None,
        Args:
            num_frame (int, tuple): input frame number
            num_joints (int, tuple): joints number
            in_chans (int): number of input channels, 3D joints have 3 channels: (x,y,z)
            embed_dim_ratio (int): embedding dimension ratio
            depth (int): depth of transformer
            num_heads (int): number of attention heads
            mlp_ratio (int): ratio of mlp hidden dim to embedding dim
            qkv_bias (bool): enable bias for qkv if True
            qk_scale (float): override default qk scale of head_dim ** -0.5 if set
            drop_rate (float): dropout rate
            attn_drop_rate (float): attention dropout rate
            drop_path_rate (float): stochastic depth rate
            norm_layer: (nn.Module): normalization layer
        """
        super().__init__()

        
            
        self.num_joints = num_joints
        self.num_views = num_views
        self.embed_dim_ratio = embed_dim_ratio
        norm_layer = norm_layer or partial(nn.LayerNorm, eps=1e-6)
        
        # embed_dim = embed_dim_ratio * 2 * num_joints   # because we add ray tokens
        embed_dim_multiplier = 1
        out_dim = num_joints * 3     #### output dimension is num_joints * 3
        
        self.print_macs_summary = False
        
        self.hidden_dim = hidden_dim
        
        self.apply_view_fusion = cfg.NETWORK.APPLY_VIEW_FUSION
        
        self.add_view_enc = cfg.NETWORK.ADD_VIEW_ENCODING
        
        ## work with random number of views
        self.random_num_views = cfg.DATASET.TRAIN_RANDOM_NUM_VIEWS
        if not self.apply_view_fusion and self.random_num_views:
            raise 'This configuration is not possible!'
        if self.apply_view_fusion and self.random_num_views:
            self.max_num_views = cfg.DATASET.MAX_NUM_VIEWS
            self.min_num_views = cfg.DATASET.MIN_NUM_VIEWS
        
        
        ### spatial patch embedding
        self.point_3d_to_embedding = nn.Linear(in_chans, embed_dim_ratio)  # dummy line for legacy reasons
        
        self.apply_sine_encoding_on_points = cfg.DATASET.APPLY_SINE_ENCODING_ON_RAYS
        self.apply_sine_encoding_on_points_nerf = cfg.DATASET.APPLY_SINE_ENCODING_ON_RAYS_NERF
        if self.apply_sine_encoding_on_points and self.apply_sine_encoding_on_points_nerf:
            raise 'This configuration is not possible!'
        self.sine_d_model = cfg.DATASET.SINE_D_MODEL
        self.sine_L_nerf = cfg.DATASET.SINE_L_NERF
        
        
        self.feed_camera_calibration = cfg.NETWORK.FEED_CAMERA_CALIBRATION
        if self.feed_camera_calibration and self.apply_sine_encoding_on_points or self.feed_camera_calibration and self.apply_sine_encoding_on_points_nerf:
            raise 'This configuration is not possible!'
        self.use_only_2D = cfg.NETWORK.FEED_ONLY_2D
        if self.use_only_2D and self.feed_camera_calibration:
            raise 'This configuration is not possible!'
        if self.use_only_2D and self.apply_sine_encoding_on_points or self.use_only_2D and self.apply_sine_encoding_on_points_nerf: 
            raise 'This configuration is not possible!'
        
        #### depth information
        self.concat_depth_as_input = cfg.NETWORK.CONCAT_DEPTH_AS_INPUT
        if self.concat_depth_as_input:
            self.depth_to_embedding = nn.Linear(1, embed_dim_ratio)
            embed_dim_multiplier += 1
            
        #### no intersection features
        self.not_use_intersection_features = cfg.NETWORK.NOT_USE_INTERSECTION_FEATURES
            
        
        self.concat_direction_and_intersection_first = cfg.NETWORK.POSE_3D_FUSER_CONCAT_DIRECTION_INTERSECTION_FIRST
        if self.concat_direction_and_intersection_first and self.not_use_intersection_features:
            raise 'This configuration is not possible!'
        
        if self.apply_sine_encoding_on_points:
            if self.concat_direction_and_intersection_first and self.apply_view_fusion:
                self.encoding_to_embedding = nn.Linear(self.sine_d_model * 2, embed_dim_ratio)
            else:
                self.encoding_to_embedding = nn.Linear(self.sine_d_model, embed_dim_ratio)
        elif self.apply_sine_encoding_on_points_nerf:
            if self.concat_direction_and_intersection_first and self.apply_view_fusion:
                self.encoding_to_embedding = nn.Linear(self.sine_L_nerf * 2 * 3 * 2, embed_dim_ratio)
            else:
                self.encoding_to_embedding = nn.Linear(self.sine_L_nerf * 2 * 3, embed_dim_ratio)
        else:
            if self.concat_direction_and_intersection_first and self.apply_view_fusion:
                self.encoding_to_embedding = nn.Linear(3 * 2, embed_dim_ratio)
            else:
                self.encoding_to_embedding = nn.Linear(3, embed_dim_ratio)
            
        self.concat_confidence = cfg.NETWORK.POSEFORMER_CONCAT_CONFIDENCE_EMB
        if cfg.NETWORK.POSE_3D_FUSER_USE_MIDDLE_POINTS and self.concat_confidence:
            raise 'This configuration is not possible!'
        
        if self.concat_confidence:
            self.confidence_to_embedding = nn.Linear(1, embed_dim_ratio)    
            embed_dim_multiplier += 1
            
        # if self.feed_camera_calibration:
        #     self.camera_calibration_to_embedding = nn.Linear(17, embed_dim_ratio)
        #     self.encoding_to_embedding = nn.Linear(2, embed_dim_ratio)
        #     if self.concat_confidence:
        #         embed_dim_ratio *= 3    # because we concatenate camera calibration features and confidence
        #     else:
        #         embed_dim_ratio *= 2    # because we concatenate camera calibration features  
        # elif self.apply_view_fusion:
        #     if self.concat_confidence:
        #         if self.concat_direction_and_intersection_first:
        #             embed_dim_ratio *= 2   # because we concatenate direction and intersection features first
        #         else:
        #             embed_dim_ratio *= 3        # because we concatenate direction and intersection features and confidence
        #     elif self.concat_direction_and_intersection_first:
        #         embed_dim_ratio *= 1        # because we concatenate direction and intersection features first
        #     else:
        #         embed_dim_ratio *= 2        # because we concatenate direction and intersection features
        # else:
        #     if self.concat_confidence:
        #         embed_dim_ratio *= 2        # because we concatenate closest points and confidence
        
        if self.use_only_2D:
            self.encoding_to_embedding = nn.Linear(2, embed_dim_ratio)
        elif self.feed_camera_calibration:
            self.camera_calibration_to_embedding = nn.Linear(17, embed_dim_ratio)
            self.encoding_to_embedding = nn.Linear(2, embed_dim_ratio)
            embed_dim_multiplier += 1
        elif self.apply_view_fusion:
            if not self.concat_direction_and_intersection_first and not self.not_use_intersection_features:
                embed_dim_multiplier += 1
                
        embed_dim_ratio *= embed_dim_multiplier
        
        
        if self.apply_view_fusion and self.random_num_views:
            self.fusion_token = torch.nn.Parameter(torch.randn(1, 1, embed_dim_ratio), requires_grad=True)
            
        self.Spatial_pos_embed = nn.Parameter(torch.zeros(1, num_joints, embed_dim_ratio))
        
        self.pos_drop = nn.Dropout(p=drop_rate)
        
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]  # stochastic depth decay rule
        
        
        ### view fusion
        if self.apply_view_fusion:
            if not self.random_num_views:
                self.View_enc_learned = nn.Parameter(torch.zeros(1, num_views, embed_dim_ratio))
            else:
                self.View_enc_learned = nn.Parameter(torch.zeros(1, self.max_num_views + 1, embed_dim_ratio))
            self.blocks_view_fusion = nn.ModuleList([
                Block(
                    dim=embed_dim_ratio, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale,
                    drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[i], norm_layer=norm_layer)
                for i in range(depth)])
        
        ##### create FPT blocks
        num_tokens = num_joints
        self.blocks = nn.ModuleList([
            Block(
                dim=embed_dim_ratio, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale,
                drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[i], norm_layer=norm_layer, num_tokens=num_tokens)
            for i in range(depth)])
        
        

        self.Spatial_norm = norm_layer(embed_dim_ratio)

        self.View_norm = norm_layer(embed_dim_ratio)
        
        ####### A easy way to implement weighted mean
        self.linear_weighted_mean = linear_weighted_mean
        if self.linear_weighted_mean:
            self.weighted_mean = nn.Linear(num_views * embed_dim_ratio, embed_dim_ratio)
        else:
            self.weighted_mean = torch.nn.Conv1d(in_channels=num_views, out_channels=1, kernel_size=1)

        self.head = nn.Sequential(
            nn.LayerNorm(embed_dim_ratio),
            nn.Linear(embed_dim_ratio , 3),
        )
            
        
    
    def compute_sine_cosine_encoding(self, coords, d_model):
        """
        Compute sine-cosine positional encoding for a batch of coordinates.

        Args:
        coords (torch.Tensor): The coordinates of shape (batch, num_rays, 3).
        d_model (int): The dimension of the model.

        Returns:
        torch.Tensor: A positional encoding tensor of shape (batch, num_rays, d_model).
        """
        assert d_model % 6 == 0, "d_model should be a multiple of 6"

        batch_size, num_rays, _ = coords.shape
        device = coords.device

        div_term = torch.exp(torch.arange(0, d_model // 3, 2, dtype=torch.float, device=device) * -(torch.log(torch.tensor(10000.0, device=device)) / (d_model // 3)))

        # Compute positional encodings for each coordinate dimension
        pe_x = torch.zeros((batch_size, num_rays, d_model // 3), dtype=torch.float, device=device)
        pe_y = torch.zeros((batch_size, num_rays, d_model // 3), dtype=torch.float, device=device)
        pe_z = torch.zeros((batch_size, num_rays, d_model // 3), dtype=torch.float, device=device)

        pe_x[:, :, 0::2] = torch.sin(coords[:, :, 0:1] * div_term)
        pe_x[:, :, 1::2] = torch.cos(coords[:, :, 0:1] * div_term)
        pe_y[:, :, 0::2] = torch.sin(coords[:, :, 1:2] * div_term)
        pe_y[:, :, 1::2] = torch.cos(coords[:, :, 1:2] * div_term)
        pe_z[:, :, 0::2] = torch.sin(coords[:, :, 2:3] * div_term)
        pe_z[:, :, 1::2] = torch.cos(coords[:, :, 2:3] * div_term)

        # Concatenate the positional encodings
        positional_encoding = torch.cat((pe_x, pe_y, pe_z), dim=-1)

        return positional_encoding
    
    
    def compute_sine_cosine_encoding_nerf(self, coords, L=4):
        """
        Compute sine-cosine positional encoding for a batch of coordinates based on NeRF.

        Args:
            coords (torch.Tensor): (batch, num_rays, 3)
            L (int, optional): Defaults to 4.
        """
        
        batch_size, num_rays, _ = coords.shape
        device = coords.device
        
        mult_terms = 2.0 ** torch.arange(0, L, device=device) * math.pi
        
        e_x = torch.zeros((batch_size, num_rays, 2 * L), device=device)
        e_y = torch.zeros((batch_size, num_rays, 2 * L), device=device)
        e_z = torch.zeros((batch_size, num_rays, 2 * L), device=device)
        
        e_x[:, :, ::2] = torch.sin(coords[:, :, 0:1] * mult_terms)
        e_x[:, :, 1::2] = torch.cos(coords[:, :, 0:1] * mult_terms)
        e_y[:, :, ::2] = torch.sin(coords[:, :, 1:2] * mult_terms)
        e_y[:, :, 1::2] = torch.cos(coords[:, :, 1:2] * mult_terms)
        e_z[:, :, ::2] = torch.sin(coords[:, :, 2:] * mult_terms)
        e_z[:, :, 1::2] = torch.cos(coords[:, :, 2:] * mult_terms)
        
        encoded_coords = torch.cat((e_x, e_y, e_z), dim=-1)
        
        return encoded_coords
        

    def forward(self, x, is_training=True, **kwargs):
        # x: (b, num_joints, 1 or num_views, 4) or or (b, num_joints, num_views, 7) if apply_view_fusion
        b, num_joints, num_points, d = x.shape
        # depths = kwargs['depths'] if 'depths' in kwargs else None
        if self.apply_view_fusion:
            # assert d == 6, 'Input shape should be (b, num_joints, num_views, 6) if apply_view_fusion'
            if self.random_num_views and is_training:
                num_points = torch.randint(self.min_num_views, self.max_num_views + 1, (1,)).item()
                x = x[:, :, :num_points, :]
                
            if self.use_only_2D:
                joints_2d = x[:, :, :, :2]
                conf = x[:, :, :, 19:20] if self.concat_confidence else None
            elif self.feed_camera_calibration:
                joints_2d = x[:, :, :, :2]
                camera_calibration = x[:, :, :, 2:19]
                conf = x[:, :, :, 19:20] if self.concat_confidence else None
            else:
                direction_features = x[:, :, :, :3]
                intersection_features = x[:, :, :, 3:6]
                conf = x[:, :, :, 6:7] if self.concat_confidence else None
                if self.concat_depth_as_input:
                    depths = x[:, :, :, 7:8]
            
            if self.feed_camera_calibration or self.use_only_2D:
                x = self.encoding_to_embedding(joints_2d.view(b*num_joints, -1, 2))
                if self.feed_camera_calibration:
                    camera_calibration = self.camera_calibration_to_embedding(camera_calibration.view(b*num_joints, -1, 17))
                    x = torch.cat((x, camera_calibration), dim=-1)
                if self.concat_confidence:
                    conf_emb = self.confidence_to_embedding(conf.view(b*num_joints, -1, 1))
                    x = torch.cat((x, conf_emb), dim=-1)
            else:
                if self.apply_sine_encoding_on_points:
                    direction_features = self.compute_sine_cosine_encoding(direction_features.view(b*num_joints, -1, 3), self.sine_d_model)
                    intersection_features = self.compute_sine_cosine_encoding(intersection_features.view(b*num_joints, -1, 3), self.sine_d_model)
                elif self.apply_sine_encoding_on_points_nerf:
                    direction_features = self.compute_sine_cosine_encoding_nerf(direction_features.view(b*num_joints, -1, 3), self.sine_L_nerf)
                    intersection_features = self.compute_sine_cosine_encoding_nerf(intersection_features.view(b*num_joints, -1, 3), self.sine_L_nerf)
                else:
                    direction_features = direction_features.view(b*num_joints, -1, 3)
                    intersection_features = intersection_features.view(b*num_joints, -1, 3)
                
                # x = x.view(b*num_joints, num_points, -1)
                
                if self.not_use_intersection_features:
                    x = direction_features
                    x = self.encoding_to_embedding(x)
                elif self.concat_direction_and_intersection_first:
                    x = torch.cat((direction_features, intersection_features), dim=-1)
                    x = self.encoding_to_embedding(x)
                else:
                    direction_features = self.encoding_to_embedding(direction_features)
                    intersection_features = self.encoding_to_embedding(intersection_features)
                    x = torch.cat((direction_features, intersection_features), dim=-1)
                if self.concat_confidence:
                    conf_emb = self.confidence_to_embedding(conf.view(b*num_joints, -1, 1))
                    x = torch.cat((x, conf_emb), dim=-1)
                    
            if self.concat_depth_as_input:
                depth_emb = self.depth_to_embedding(depths.view(b*num_joints, -1, 1))
                x = torch.cat((x, depth_emb), dim=-1)
            
            if not self.random_num_views:
                if self.add_view_enc:
                    x += self.View_enc_learned
                
                x = x.view(b*num_joints, num_points, -1)
            
            if self.random_num_views:
                # Append the fusion token to the input
                fusion_token = self.fusion_token.expand(b*num_joints, -1, -1)  # Shape: [batch, 1, embed_dim]
                x = x.view(b*num_joints, num_points, -1)
                x = torch.cat([fusion_token, x], dim=1)  # Shape: [batch, num_tokens + 1, embed_dim]
                
                if self.add_view_enc:
                    x += self.View_enc_learned[:, :num_points + 1]
            
            x = self.pos_drop(x)
            for blk in self.blocks_view_fusion:
                x = blk(x)
            
            x = self.View_norm(x)
            
            # x = x.view(b, num_joints, num_points, -1)
            if not self.random_num_views:
                x = self.weighted_mean(x).squeeze(1)
            else:
                x = x[:, 0, :] # first token is the fusion token
            x = x.view(b, num_joints, -1)
            
        else:
            points = x[:, :, :, :3]
            conf = x[:, :, :, 3:4] if self.concat_confidence else None
            if self.apply_sine_encoding_on_points:
                points = self.compute_sine_cosine_encoding(points.view(b, -1, 3), self.sine_d_model)
            elif self.apply_sine_encoding_on_points_nerf:
                points = self.compute_sine_cosine_encoding_nerf(points.view(b, -1, 3), self.sine_L_nerf)
            
            x = self.encoding_to_embedding(points)
            if self.concat_confidence:
                conf_emb = self.confidence_to_embedding(conf.view(b, -1, 1))
                x = torch.cat((x, conf_emb), dim=-1)
            x = x.view(b, num_joints, num_points, -1)
            x = x.sum(dim=2)
        x += self.Spatial_pos_embed
        
        
        x = self.pos_drop(x)
        for ix, blk in enumerate(self.blocks):
            if ix == len(self.blocks) - 1:
                x = blk(x)
            x = blk(x)

        x = self.Spatial_norm(x)
        
        x = x.view(b, num_joints, -1)
            
        x = self.head(x)

        x = x.view(b, -1, 3)

        return x


class MultiView_RUMPL_G(nn.Module):
    def __init__(self, cfg, **kwargs):
        super(MultiView_RUMPL_G, self).__init__()

        print(cfg.NETWORK)
        # num_views = 5 if cfg.DATASET.TEST_DATASET.startswith('multiview_cmu_panoptic') else 4
        if cfg.DATASET.TEST_DATASET.startswith('multiview_cmu_panoptic') or cfg.DATASET.TEST_DATASET.startswith('multiview_amass_cmu_panoptic_pose_former'):
            num_views = 5
        else:
            num_views = 4
            
        if cfg.DATASET.TRAIN_VIEWS is not None:
            num_views = len(cfg.DATASET.TRAIN_VIEWS)
            if cfg.DATASET.USE_HELPER_CAMERAS:
                assert cfg.DATASET.TRAIN_VIEWS_HELPER is not None
                num_views += len(cfg.DATASET.TRAIN_VIEWS_HELPER)
                
        if cfg.DATASET.TRAIN_ON_ALL_CAMERAS and cfg.DATASET.TEST_ON_ALL_CAMERAS:
            num_views = cfg.DATASET.N_VIEWS_TRAIN_TEST_ALL
            
        if 'master_cam' in cfg.DATASET.TEST_DATASET:
            num_views = cfg.DATASET.N_MASTER_CAMERAS
                
            
        self.init_weights_from = cfg.NETWORK.INIT_WEIGHTS_FROM

        ##################################################
        self.features = MultiView_RUMPL(
                                 num_joints = cfg.NETWORK.NUM_JOINTS,
                                 embed_dim_ratio=cfg.NETWORK.DIM,
                                 depth=cfg.NETWORK.TRANSFORMER_DEPTH,
                                 num_heads=cfg.NETWORK.TRANSFORMER_HEADS,
                                 drop_rate=cfg.NETWORK.POSEFORMER_DROP_RATE,
                                 attn_drop_rate=cfg.NETWORK.POSEFORMER_ATTN_DROP_RATE,
                                 drop_path_rate=cfg.NETWORK.POSEFORMER_DROP_PATH_RATE,
                                 num_views=num_views,
                                 linear_weighted_mean=cfg.NETWORK.POSEFORMER_LINEAR_WEIGHTED_MEAN,
                                 hidden_dim=cfg.NETWORK.POSEFORMER_OUTPUT_HEAD_HIDDEN_DIM,
                                 cfg=cfg,
                                 )
        ###################################################3

    def forward(self, x, **kwargs):
        x = self.features(x, **kwargs)
        return x

    def init_weights(self, pretrained=''):
        if os.path.isfile(pretrained):
            if 'multiview_h36m' in pretrained or 'multiview_amass_h36m' in pretrained or 'multiview_cmu_panoptic' in pretrained or 'multiview_amass_cmu_panoptic_pose_former' in pretrained:
                # >>>>>>>>>>>>>>>>>>>>>>>>>>> from H36M pretrained >>>>>>>>>>>>>>>>>>>>>>>>>>>
                logger.info('=> loading Pretrained model {}'.format(pretrained))
                pretrained_state_dict = torch.load(pretrained, map_location='cpu')
                self.load_state_dict(pretrained_state_dict, strict=False)
            else:
                # >>>>>>>>>>>>>>>>>>>>>>>>>>> from COCO pretrained >>>>>>>>>>>>>>>>>>>>>>>>>>>
                logger.info('=> init final MLP head from normal distribution')
                for m in self.features.mlp_head.modules():
                    if isinstance(m, nn.Linear):
                        trunc_normal_(m.weight, std=.02)
                        if isinstance(m, nn.Linear) and m.bias is not None:
                            nn.init.constant_(m.bias, 0)

                pretrained_state_dict = torch.load(pretrained, map_location='cpu')
                logger.info('=> loading COCO Pretrained model {}'.format(pretrained))
                existing_state_dict = {}
                for name, m in pretrained_state_dict.items():
                    if name in self.state_dict():
                        #if 'mlp_head' in name or 'pos_embedding' in name or 'keypoint_token' in name or 'patch_to_embedding' in name:       # 2D Pos Embeddings
                        #    continue
                        if 'keypoint_token' in name:
                            new_m = torch.zeros(1, 17, 192)
                            # Human 36M -> MPII
                            # map_idx = [6, 2, 1, 0, 3, 4, 5, 7, 8, 9, 9, 13, 14, 15, 12, 11, 10]
                            # Human 36M -> COCO
                            map_idx = [12, 12, 14, 16, 11, 13, 15, 11, 1, 0, 2, 5, 7, 9, 6, 8, 10]
                            new_m[0] = m[0][map_idx]
                            m = new_m
                            print('Shift Token ...')

                        existing_state_dict[name] = m
                        logger.info(":: {} is loaded from {}".format(name, pretrained))
                        print('Size: ', m.shape)

                self.load_state_dict(existing_state_dict, strict=False)

        elif self.init_weights_from == 'xavier_uniform':
            logger.info('=> init weights from xavier uniform distribution')
            for m in self.modules():
                if not isinstance(m, MultiView_RUMPL_G) or not isinstance(m, MultiView_RUMPL):
                    nn.init.xavier_uniform_(m.weight)
                    if m.bias is not None:
                        nn.init.constant_(m.bias, 0)
                
        # >>>>>>>>>>>>>>>>>>>>>>>>>>> from scratch >>>>>>>>>>>>>>>>>>>>>>>>>>>
        else:
            logger.info('=> init weights from normal distribution')
            for m in self.modules():
                if isinstance(m, nn.Conv2d):
                    nn.init.normal_(m.weight, std=0.001)
                elif isinstance(m, nn.BatchNorm2d):
                    nn.init.constant_(m.weight, 1)
                    nn.init.constant_(m.bias, 0)
                elif isinstance(m, nn.ConvTranspose2d):
                    nn.init.normal_(m.weight, std=0.001)
                    if self.deconv_with_bias:
                        nn.init.constant_(m.bias, 0)


def get_multiview_rumpl_net(cfg, is_train, **kwargs):
    model = MultiView_RUMPL_G(cfg, **kwargs)
    if is_train and cfg.NETWORK.INIT_WEIGHTS:
        model.init_weights(cfg.NETWORK.PRETRAINED)

    return model