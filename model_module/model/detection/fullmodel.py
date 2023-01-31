
import torch
from torch import nn
from einops import rearrange
from model_module.model.detection.base_lss_fpn import BaseLSSFPN
from model_module.model.detection.bev_depth_head import BEVDepthHead
from torch.cuda.amp.autocast_mode import autocast
import torch.nn.functional as F
from .modules import ResBlock, LayerNorm, ConvNeXtBlock
# __all__ = ['BaseBEVDepth']


'''
FullModel

- Our Depth Model(backbone + head) + voxel_pooling + BEVDepth Detection Head 
'''


class BaseBEVDepth(nn.Module):
    """Source code of `BEVDepth`, `https://arxiv.org/abs/2112.11790`.

    Args:
        backbone_conf (dict): Config of backbone.
        head_conf (dict): Config of head.
        is_train_depth (bool): Whether to return depth.
            Default: False.
    """

    # TODO: Reduce grid_conf and data_aug_conf
    def __init__(self, reduce_dim, downsample_factor, dbound, is_train_depth, backbone , voxel_pooling, head):
        super(BaseBEVDepth, self).__init__()
        self.is_train_depth = is_train_depth 
        self.backbone = backbone
        self.det_head = head
        self.voxel_pooling = voxel_pooling

        self.reduce_dim = reduce_dim
        self.downsample_factor = downsample_factor 
        self.dbound = dbound 
        self.depth_channels = int(
            (self.dbound[1] - self.dbound[0]) / self.dbound[2])

        self.depth_head = nn.Sequential(ConvNeXtBlock(self.reduce_dim, 0.0),     
                                    ConvNeXtBlock(self.reduce_dim, 0.0),
                                    nn.Conv2d(self.reduce_dim, self.depth_channels, kernel_size=1, stride=1, padding=0))


    def forward(
        self,
        batch,
        # x,
        # mats_dict,
        timestamps=None,
    ):
        """Forward function for BEVDepth

        Args:
            x (Tensor): Input ferature map.
            mats_dict(dict):
                sensor2ego_mats(Tensor): Transformation matrix from
                    camera to ego with shape of (B, num_sweeps,
                    num_cameras, 4, 4).
                intrin_mats(Tensor): Intrinsic matrix with shape
                    of (B, num_sweeps, num_cameras, 4, 4).
                ida_mats(Tensor): Transformation matrix for ida with
                    shape of (B, num_sweeps, num_cameras, 4, 4).
                sensor2sensor_mats(Tensor): Transformation matrix
                    from key frame camera to sweep frame camera with
                    shape of (B, num_sweeps, num_cameras, 4, 4).
                bda_mat(Tensor): Rotation matrix for bda with shape
                    of (B, 4, 4).
            timestamps (long): Timestamp.
                Default: None.

        Returns:
            tuple(list[dict]): Output results for tasks.
        """
        
        x = batch['image'].cuda() #x.shape : ([B, 6, 3, 256, 704])
        b, n, _, _, _ = x.shape
        intrinsics = torch.eye(4).expand(b, n, 4, 4)    # B, 6, 3, 3
        intrinsics = intrinsics.clone()
        intrinsics[:,:,:3,:3] = batch['intrinsics']     # B, 6, 4, 4
 
        mats_dict = {'sensor2ego_mats': batch['sensor2ego_mats'].unsqueeze(1).cuda(),
                    'intrin_mats': intrinsics.unsqueeze(1).cuda(),
                    'ida_mats': batch['ida_mats'].unsqueeze(1).cuda(),
                    'sensor2sensor_mats': batch['sensor2sensor_mats'].unsqueeze(1).cuda(),
                    'bda_mat': batch['bda_mats'].cuda()
                    }

        if self.is_train_depth and self.training: 
            imgs = rearrange(x, 'b n ... -> (b n) ...') # ([12, 3, 256, 704]) 


            seg_feats, depth_feats = self.backbone(imgs) # b*n , 64, 176, 64
            depth_bin = self.depth_head(depth_feats) # ([12, 112, 64, 176])
            depth_bin = depth_bin.softmax(1)         # b*n , 112, 64, 176
            x, depth_pred = self.voxel_pooling(seg_feats.unsqueeze(1), depth_bin.unsqueeze(1), mats_dict = mats_dict, timestamps = timestamps, is_return_depth=self.is_train_depth)
            # depth_pred: ([12, 112, 64, 176])
            det_pred = self.det_head(x) 

            return det_pred, depth_pred
        else:
            imgs = rearrange(x, 'b n ... -> (b n) ...')
            seg_feats, depth_feats = self.backbone(imgs) # b*n , 64, 176, 64
            depth_bin = self.depth_head(depth_feats)
            depth_bin = depth_bin.softmax(1)             # b*n , 112, 64, 176
            x  = self.voxel_pooling(seg_feats.unsqueeze(1), depth_bin.unsqueeze(1), mats_dict = mats_dict, timestamps = timestamps, is_return_depth=self.is_train_depth)
            det_pred = self.det_head(x)

            return det_pred

    def get_targets(self, gt_boxes, gt_labels):
        """Generate training targets for a single sample.

        Args:
            gt_bboxes_3d (:obj:`LiDARInstance3DBoxes`): Ground truth gt boxes.
            gt_labels_3d (torch.Tensor): Labels of boxes.

        Returns:
            tuple[list[torch.Tensor]]: Tuple of target including \
                the following results in order.

                - list[torch.Tensor]: Heatmap scores.
                - list[torch.Tensor]: Ground truth boxes.
                - list[torch.Tensor]: Indexes indicating the position \
                    of the valid boxes.
                - list[torch.Tensor]: Masks indicating which boxes \
                    are valid.
        """
        return self.det_head.get_targets(gt_boxes, gt_labels)

    def loss(self, targets, preds_dicts):
        """Loss function for BEVDepth.

        Args:
            gt_bboxes_3d (list[:obj:`LiDARInstance3DBoxes`]): Ground
                truth gt boxes.
            gt_labels_3d (list[torch.Tensor]): Labels of boxes.
            preds_dicts (dict): Output of forward function.

        Returns:
            dict[str:torch.Tensor]: Loss of heatmap and bbox of each task.
        """
        return self.det_head.loss(targets, preds_dicts)


    def depth_loss(self, depth_labels, depth_preds):
        depth_labels = self.get_downsampled_gt_depth(depth_labels)
        depth_preds = depth_preds.permute(0, 2, 3, 1).contiguous().view(
            -1, self.depth_channels)
        fg_mask = torch.max(depth_labels, dim=1).values > 0.0

        with autocast(enabled=False):
            depth_loss = (F.binary_cross_entropy(
                depth_preds[fg_mask],
                depth_labels[fg_mask],
                reduction='none',
            ).sum() / max(1.0, fg_mask.sum()))

        return 3.0 * depth_loss



    #! DO one-hot vector
    def get_downsampled_gt_depth(self, gt_depths):
        """
        Input:
            gt_depths: [B, N, H, W]
        Output:
            gt_depths: [B*N*h*w, d]
        """
        B, N, H, W = gt_depths.shape
        gt_depths = gt_depths.view(
            B * N,
            H // self.downsample_factor,
            self.downsample_factor,
            W // self.downsample_factor,
            self.downsample_factor,
            1,
        )
        gt_depths = gt_depths.permute(0, 1, 3, 5, 2, 4).contiguous()

        gt_depths = gt_depths.view(
            -1, self.downsample_factor * self.downsample_factor)

        #! if depth == 0: 1e-5 대입
        gt_depths_tmp = torch.where(gt_depths == 0.0,
                                    1e5 * torch.ones_like(gt_depths),
                                    gt_depths) 

        #* 각 256마다 min pooling, 가장 가까운 depth를 label로 쓰기 위해서 
        gt_depths = torch.min(gt_depths_tmp, dim=-1).values  
       
        gt_depths = gt_depths.view(B * N, H // self.downsample_factor,
                                   W // self.downsample_factor) 
        
        #* 2 ~ 58m -> 112 bin
        # (depth - (2-0.5)) /0.5
        gt_depths = (gt_depths -
                     (self.dbound[0] - self.dbound[2])) / self.dbound[2] 

        #! (depth < 112+1) and (depth > 0.0) 
        gt_depths = torch.where(
            (gt_depths < self.depth_channels + 1) & (gt_depths >= 0.0),
            gt_depths, torch.zeros_like(gt_depths)) 
        
        gt_depths = F.one_hot(gt_depths.long(),
                              num_classes=self.depth_channels + 1).view(
                                  -1, self.depth_channels + 1)[:, 1:]
        

        return gt_depths.float()


    def get_bboxes(self, preds_dicts, img_metas=None, img=None, rescale=False):
        """Generate bboxes from bbox head predictions.

        Args:
            preds_dicts (tuple[list[dict]]): Prediction results.
            img_metas (list[dict]): Point cloud and image's meta info.

        Returns:
            list[dict]: Decoded bbox, scores and labels after nms.
        """
        return self.det_head.get_bboxes(preds_dicts, img_metas, img, rescale)
