# ------------------------------------------------------------------------------
# Copyright (c) 2024 UCLouvain. All rights reserved.
# Licensed under the GNU Affero General Public License v3.0 (AGPLv3).
#
# Author: Seyed Abolfazl Ghaemzadeh, ICTEAM, UCLouvain
# ------------------------------------------------------------------------------
#
# Portions of this file are:
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# Written by Chunyu Wang (chnuwa@microsoft.com)
# ------------------------------------------------------------------------------

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import torch.nn as nn
import torch
import torch.nn.functional as F

class JointsMSELoss(nn.Module):
    def __init__(self, use_target_weight):
        super(JointsMSELoss, self).__init__()
        self.criterion = nn.MSELoss(reduction='mean')
        self.use_target_weight = use_target_weight

    def forward(self, output, target, target_weight):
        batch_size = output.size(0)
        num_joints = output.size(1)
        heatmaps_pred = output.reshape((batch_size, num_joints, -1)).split(1, 1)
        heatmaps_gt = target.reshape((batch_size, num_joints, -1)).split(1, 1)
        loss = 0

        for idx in range(num_joints):
            heatmap_pred = heatmaps_pred[idx].squeeze()
            heatmap_gt = heatmaps_gt[idx].squeeze()
            if self.use_target_weight:
                loss += self.criterion(heatmap_pred.mul(target_weight[:, idx]),
                                       heatmap_gt.mul(target_weight[:, idx]))
            else:
                loss += self.criterion(heatmap_pred, heatmap_gt)

        return loss

class MPJPE(nn.Module):
    def __init__(self, cfg=None):
        super(MPJPE, self).__init__()
        self.weight_axis = None
        if cfg is not None:
            self.weight_axis = torch.tensor(cfg.LOSS.WEIGHT_AXIS, requires_grad=False).cuda() if cfg.LOSS.WEIGHT_AXIS is not None else None
    
    def forward(self, output, target, w=None):
        """
        Mean per-joint position error (i.e. mean Euclidean distance),
        often referred to as "Protocol #1" in many papers.
        """
        assert output.shape == target.shape
        loss_x = torch.mean(torch.abs(output[:, :, 0] - target[:, :, 0]))
        loss_y = torch.mean(torch.abs(output[:, :, 1] - target[:, :, 1]))
        loss_z = torch.mean(torch.abs(output[:, :, 2] - target[:, :, 2]))
        if self.weight_axis is not None:
            return torch.mean(w * torch.norm((output - target) * self.weight_axis, dim=len(target.shape)-1)), [loss_x, loss_y, loss_z]
        return torch.mean(torch.norm(output - target, dim=len(target.shape)-1)), [loss_x, loss_y, loss_z]
    
class PlaneProjectionLoss(nn.Module):
    def __init__(self, cfg=None):
        super(PlaneProjectionLoss, self).__init__()
        self.weight_axis = None
        if cfg is not None:
            self.weight_axis = torch.tensor(cfg.LOSS.WEIGHT_AXIS, requires_grad=False).cuda() if cfg.LOSS.WEIGHT_AXIS is not None else None
            
    def project_point_to_plane(self, point, plane_normal):
        # Normalize the normal vector
        normal_unit = plane_normal / torch.norm(plane_normal, dim=-1, keepdim=True)
        plane_point = torch.zeros_like(point, device=point.device)
        
        # Compute the vector from the plane point to the point to project
        vector_to_point = point - plane_point
        
        # Calculate the distance from the point to the plane along the normal
        dist = torch.sum(vector_to_point * normal_unit, dim=-1, keepdim=True)
        
        # Compute the projection by subtracting the distance along the normal vector from the original point
        projection = point - dist * normal_unit
        
        return projection
    
    def forward(self, output, target, w=None, rays_direction=None):
        """
        Mean per-joint position error (i.e. mean Euclidean distance),
        often referred to as "Protocol #1" in many papers.
        """
        assert rays_direction is not None
        assert output.shape == target.shape
        output_projected = []
        target_projected = []
        for ray in rays_direction:
            output_projected.append(self.project_point_to_plane(output, ray))
            target_projected.append(self.project_point_to_plane(target, ray))
            
        output_projected = torch.stack(output_projected, dim=1)
        target_projected = torch.stack(target_projected, dim=1)
            
        
        loss_x = torch.mean(torch.abs(output[:, :, 0] - target[:, :, 0]))
        loss_y = torch.mean(torch.abs(output[:, :, 1] - target[:, :, 1]))
        loss_z = torch.mean(torch.abs(output[:, :, 2] - target[:, :, 2]))
        if self.weight_axis is not None:
            return torch.mean(w * torch.norm((output_projected - target_projected) * self.weight_axis, dim=len(target_projected.shape)-1)), [loss_x, loss_y, loss_z]
        return torch.mean(torch.norm(output_projected - target_projected, dim=len(target_projected.shape)-1)), [loss_x, loss_y, loss_z]
    
class KeypointsPoseL1Loss(nn.Module):
    def __init__(self, cfg) -> None:
        super().__init__()
        self.weight_axis = None
        if cfg is not None:
            self.weight_axis = torch.tensor(cfg.LOSS.WEIGHT_AXIS, requires_grad=False).cuda() if cfg.LOSS.WEIGHT_AXIS is not None else None
        self.l1_x = nn.L1Loss()
        self.l1_y = nn.L1Loss()
        self.l1_z = nn.L1Loss()
        
    def forward(self, output, target, w=None):
        """
        Mean per-joint position error (i.e. mean L1 distance)
        """
        assert output.shape == target.shape
        loss_x = self.l1_x(output[:, :, 0], target[:, :, 0])
        loss_y = self.l1_y(output[:, :, 1], target[:, :, 1])
        loss_z = self.l1_z(output[:, :, 2], target[:, :, 2])
        if self.weight_axis is not None:
            return torch.mean((loss_x * self.weight_axis[0] + loss_y * self.weight_axis[1] + loss_z * self.weight_axis[2])), [loss_x, loss_y, loss_z]
        return torch.mean(loss_x + loss_y + loss_z), [loss_x, loss_y, loss_z]
        # if self.weight_axis is not None:
        #     return torch.mean((self.l1_x(output[:, 0], target[:, 0]) + self.l1_y(output[:, 1], target[:, 1]) + self.l1_z(output[:, 2], target[:, 2])) * self.weight_axis)
        # return torch.mean(self.l1_x(output[:, 0], target[:, 0]) + self.l1_y(output[:, 1], target[:, 1]) + self.l1_z(output[:, 2], target[:, 2]))
    
class KeypointsPoseMSELoss(nn.Module):
    def __init__(self, cfg) -> None:
        super().__init__()
        self.weight_axis = None
        if cfg is not None:
            self.weight_axis = torch.tensor(cfg.LOSS.WEIGHT_AXIS, requires_grad=False).cuda() if cfg.LOSS.WEIGHT_AXIS is not None else None
        self.mse_x = nn.MSELoss()
        self.mse_y = nn.MSELoss()
        self.mse_z = nn.MSELoss()
        
    def forward(self, output, target, w=None):
        """
        Mean per-joint position error (i.e. mean L1 distance)
        """
        assert output.shape == target.shape
        loss_x = self.mse_x(output[:, :, 0], target[:, :, 0])
        loss_y = self.mse_y(output[:, :, 1], target[:, :, 1])
        loss_z = self.mse_z(output[:, :, 2], target[:, :, 2])
        if self.weight_axis is not None:
            return torch.mean((loss_x * self.weight_axis[0] + loss_y * self.weight_axis[1] + loss_z * self.weight_axis[2])), [loss_x, loss_y, loss_z]
        return torch.mean(loss_x + loss_y + loss_z), [loss_x, loss_y, loss_z]
        # if self.weight_axis is not None:
        #     return torch.mean((self.l1_x(output[:, 0], target[:, 0]) + self.l1_y(output[:, 1], target[:, 1]) + self.l1_z(output[:, 2], target[:, 2])) * self.weight_axis)
        # return torch.mean(self.l1_x(output[:, 0], target[:, 0]) + self.l1_y(output[:, 1], target[:, 1]) + self.l1_z(output[:, 2], target[:, 2]))
    
    
class Weighted_MPJPE(nn.Module):
    def __init__(self):
        super(Weighted_MPJPE, self).__init__()

    def forward(self, output, target, w):
        """
        Weighted mean per-joint position error (i.e. mean Euclidean distance)
        """
        assert output.shape == target.shape
        assert w.shape[0] == output.shape[0]
        w = w.squeeze()
        loss_x = torch.mean(torch.abs(output[:, :, 0] - target[:, :, 0]))
        loss_y = torch.mean(torch.abs(output[:, :, 1] - target[:, :, 1]))
        loss_z = torch.mean(torch.abs(output[:, :, 2] - target[:, :, 2]))
        return torch.mean(w * torch.norm(output - target, dim=len(target.shape)-1)), [loss_x, loss_y, loss_z]
    
    
class MPJPE_KADKHODA(nn.Module):
    def __init__(self, cfg=None):
        super(MPJPE_KADKHODA, self).__init__()
        self.weight_axis = None
        if cfg is not None:
            self.weight_axis = torch.tensor(cfg.LOSS.WEIGHT_AXIS, requires_grad=False).cuda() if cfg.LOSS.WEIGHT_AXIS is not None else None
    
    def forward(self, output, target, w=None):
        """
        Mean per-joint position error (i.e. mean Euclidean distance),
        often referred to as "Protocol #1" in many papers.
        """
        x1, x2, x3 = output
        y = target
        assert x3.shape == target.shape
        loss_x = torch.mean(torch.abs(x3[:, :, 0] - target[:, :, 0]))
        loss_y = torch.mean(torch.abs(x3[:, :, 1] - target[:, :, 1]))
        loss_z = torch.mean(torch.abs(x3[:, :, 2] - target[:, :, 2]))
        loss = torch.pow(F.pairwise_distance(x1, y), 2) + torch.pow(F.pairwise_distance(x2, y), 2) + torch.pow(F.pairwise_distance(x3, y), 2)
        return torch.mean(loss), [loss_x, loss_y, loss_z]
