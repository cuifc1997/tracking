# ------------------------------------------------------------------------------
# Portions of this code are from
# CornerNet (https://github.com/princeton-vl/CornerNet)
# Copyright (c) 2018, University of Michigan
# Licensed under the BSD 3-Clause License
# ------------------------------------------------------------------------------
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import math
import torch
import torch.nn as nn
from .utils import _tranpose_and_gather_feat, _nms, _topk
import torch.nn.functional as F
from utils.image import draw_umich_gaussian

def _slow_neg_loss(pred, gt):
    '''focal loss from CornerNet'''
    pos_inds = gt.eq(1).float()
    neg_inds = gt.lt(1).float()

    neg_weights = torch.pow(1 - gt[neg_inds], 4)

    loss = 0
    pos_pred = pred[pos_inds]
    neg_pred = pred[neg_inds]

    pos_loss = torch.log(pos_pred) * torch.pow(1 - pos_pred, 2)
    neg_loss = torch.log(1 - neg_pred) * torch.pow(neg_pred, 2) * neg_weights

    num_pos  = pos_inds.float().sum()
    pos_loss = pos_loss.sum()
    neg_loss = neg_loss.sum()

    if pos_pred.nelement() == 0:
        loss = loss - neg_loss
    else:
        loss = loss - (pos_loss + neg_loss) / num_pos
    return loss

def _neg_loss(pred, gt):
    ''' Reimplemented focal loss. Exactly the same as CornerNet.
        Runs faster and costs a little bit more memory
      Arguments:
        pred (batch x c x h x w)
        gt_regr (batch x c x h x w)
    '''
    pos_inds = gt.eq(1).float()
    neg_inds = gt.lt(1).float()

    neg_weights = torch.pow(1 - gt, 4)

    loss = 0
    pos_loss = torch.log(pred) * torch.pow(1 - pred, 2) * pos_inds
    neg_loss = torch.log(1 - pred) * torch.pow(pred, 2) * neg_weights * neg_inds

    num_pos  = pos_inds.float().sum()
    pos_loss = pos_loss.sum()
    neg_loss = neg_loss.sum()
    if num_pos == 0:
        loss = loss - neg_loss
    else:
        loss = loss - (pos_loss + neg_loss) / num_pos
    return loss


def _only_neg_loss(pred, gt):
    gt = torch.pow(1 - gt, 4)
    neg_loss = torch.log(1 - pred) * torch.pow(pred, 2) * gt
    return neg_loss.sum()

class FastFocalLoss(nn.Module):
    '''
    Reimplemented focal loss, exactly the same as the CornerNet version.
    Faster and costs much less memory.
    '''
    def __init__(self, opt=None):
        super(FastFocalLoss, self).__init__()
        self.only_neg_loss = _only_neg_loss

    def forward(self, out, target, ind, mask, cat):
        '''
        Arguments:
          out, target: B x C x H x W
          ind, mask: B x M
          cat (category id for peaks): B x M
        '''
        neg_loss = self.only_neg_loss(out, target)
        pos_pred_pix = _tranpose_and_gather_feat(out, ind) # B x M x C
        pos_pred = pos_pred_pix.gather(2, cat.unsqueeze(2)) # B x M
        num_pos = mask.sum()
        pos_loss = torch.log(pos_pred) * torch.pow(1 - pos_pred, 2) * \
                   mask.unsqueeze(2)
        pos_loss = pos_loss.sum()
        if num_pos == 0:
            return - neg_loss
        return - (pos_loss + neg_loss) / num_pos

# def forward(self, out, target, ind, mask, cat, tracking_mask):
# 	"""
#
# 	:param out: Batch x Class x H x W, value=predict pos probability (0~1)
# 	:param target: Batch x Class x H x W, value=predict pos probability (0 or 1)
# 	:param ind: B x M, value = index
# 	:param mask: B x M, value = 1 or 0
# 	:param cat: B x M, value = class
# 	:param tracking_mask: B x M, value = mask of whether the target exist in previous frame
# 	:return:
# 	"""
# 	# negative samples loss (non-one)
# 	neg_loss = self.only_neg_loss(out, target)
#
# 	# predicted position
# 	pos_pred_pix = _tranpose_and_gather_feat(out, ind)  # B x M x C, value = prediction pos probability
#
# 	# cat.unsqueeze(2) : B x M -> B x M x 1, value = class, used to guide find probability of different class
# 	pos_pred = pos_pred_pix.gather(2, cat.unsqueeze(2))  # B x M, value = probability
# 	tracking_mask = tracking_mask[:, :, 0]
# 	mask_match = tracking_mask
# 	mask_new = (1 - tracking_mask) * mask
# 	num_pos = mask.sum()
# 	pos_loss = torch.log(pos_pred) * torch.pow(1 - pos_pred, 2) * mask_match.unsqueeze(2)
# 	pos_loss += 10 * torch.log(pos_pred) * torch.pow(1 - pos_pred, 2) * mask_new.unsqueeze(2)
# 	pos_loss = pos_loss.sum()
# 	if num_pos == 0:
# 		return - neg_loss
# 	return - (pos_loss + neg_loss) / num_pos

def _reg_loss(regr, gt_regr, mask):
    ''' L1 regression loss
      Arguments:
        regr (batch x max_objects x dim)
        gt_regr (batch x max_objects x dim)
        mask (batch x max_objects)
    '''
    num = mask.float().sum()
    mask = mask.unsqueeze(2).expand_as(gt_regr).float()

    regr = regr * mask
    gt_regr = gt_regr * mask

    regr_loss = nn.functional.smooth_l1_loss(regr, gt_regr, reduction='sum')
    regr_loss = regr_loss / (num + 1e-4)
    return regr_loss


class RegWeightedL1Loss(nn.Module):
    def __init__(self):
        super(RegWeightedL1Loss, self).__init__()

    def forward(self, output, mask, ind, target):
        pred = _tranpose_and_gather_feat(output, ind)
        # loss = F.l1_loss(pred * mask, target * mask, reduction='elementwise_mean')
        loss = F.l1_loss(pred * mask, target * mask, reduction='sum')
        loss = loss / (mask.sum() + 1e-4)
        return loss


class WeightedBCELoss(nn.Module):
    def __init__(self):
        super(WeightedBCELoss, self).__init__()
        self.bceloss = torch.nn.BCEWithLogitsLoss(reduction='none')

    def forward(self, output, mask, ind, target):
        # output: B x F x H x W
        # ind: B x M
        # mask: B x M x F
        # target: B x M x F
        pred = _tranpose_and_gather_feat(output, ind) # B x M x F
        loss = mask * self.bceloss(pred, target)
        loss = loss.sum() / (mask.sum() + 1e-4)
        return loss

class BinRotLoss(nn.Module):
    def __init__(self):
        super(BinRotLoss, self).__init__()

    def forward(self, output, mask, ind, rotbin, rotres):
        pred = _tranpose_and_gather_feat(output, ind)
        loss = compute_rot_loss(pred, rotbin, rotres, mask)
        return loss

def compute_res_loss(output, target):
    return F.smooth_l1_loss(output, target, reduction='elementwise_mean')

def compute_bin_loss(output, target, mask):
    mask = mask.expand_as(output)
    output = output * mask.float()
    return F.cross_entropy(output, target, reduction='elementwise_mean')

def compute_rot_loss(output, target_bin, target_res, mask):
    # output: (B, 128, 8) [bin1_cls[0], bin1_cls[1], bin1_sin, bin1_cos,
    #                 bin2_cls[0], bin2_cls[1], bin2_sin, bin2_cos]
    # target_bin: (B, 128, 2) [bin1_cls, bin2_cls]
    # target_res: (B, 128, 2) [bin1_res, bin2_res]
    # mask: (B, 128, 1)
    output = output.view(-1, 8)
    target_bin = target_bin.view(-1, 2)
    target_res = target_res.view(-1, 2)
    mask = mask.view(-1, 1)
    loss_bin1 = compute_bin_loss(output[:, 0:2], target_bin[:, 0], mask)
    loss_bin2 = compute_bin_loss(output[:, 4:6], target_bin[:, 1], mask)
    loss_res = torch.zeros_like(loss_bin1)
    if target_bin[:, 0].nonzero().shape[0] > 0:
        idx1 = target_bin[:, 0].nonzero()[:, 0]
        valid_output1 = torch.index_select(output, 0, idx1.long())
        valid_target_res1 = torch.index_select(target_res, 0, idx1.long())
        loss_sin1 = compute_res_loss(
            valid_output1[:, 2], torch.sin(valid_target_res1[:, 0]))
        loss_cos1 = compute_res_loss(
            valid_output1[:, 3], torch.cos(valid_target_res1[:, 0]))
        loss_res += loss_sin1 + loss_cos1
    if target_bin[:, 1].nonzero().shape[0] > 0:
        idx2 = target_bin[:, 1].nonzero()[:, 0]
        valid_output2 = torch.index_select(output, 0, idx2.long())
        valid_target_res2 = torch.index_select(target_res, 0, idx2.long())
        loss_sin2 = compute_res_loss(
            valid_output2[:, 6], torch.sin(valid_target_res2[:, 1]))
        loss_cos2 = compute_res_loss(
            valid_output2[:, 7], torch.cos(valid_target_res2[:, 1]))
        loss_res += loss_sin2 + loss_cos2
    return loss_bin1 + loss_bin2 + loss_res


class IDloss(nn.Module):
    def __init__(self):
        super(IDloss, self).__init__()
        self.gamma = 2.0
        self.alpha = 0.25


    def forward(self, cur_reid, pre_reid, cur_inds, pre_inds, tracking_mask):
        alpha = self.alpha
        gamma = self.gamma
        tracking_mask = tracking_mask[:, :, 0]
        a = cur_reid.shape[2]*cur_reid.shape[3]

        # circumstance index
        pre_circum_ind = torch.randint_like(pre_inds, low=0, high=a, dtype=pre_inds.dtype)
        cur_circum_ind = torch.randint_like(cur_inds, low=0, high=a, dtype=pre_inds.dtype)

        # fusion index
        cur_inds = cur_inds * tracking_mask.long() + cur_circum_ind * (1-tracking_mask.long())
        pre_inds = pre_inds * tracking_mask.long() + pre_circum_ind * (1-tracking_mask.long())

        # extract feature
        cur_feat = _tranpose_and_gather_feat(cur_reid, cur_inds) # batch x max_obj x channel
        pre_feat = _tranpose_and_gather_feat(pre_reid, pre_inds)

        # cur_feat = F.normalize(cur_feat, p=2, dim=2)
        # pre_feat = F.normalize(pre_feat, p=2, dim=2)


        # calculate simiarity
        pre_feat = pre_feat.permute(0, 2, 1).contiguous() # batch x channel x max_obj
        similarity = torch.matmul(cur_feat, pre_feat)
        similarity = F.sigmoid(similarity)

        # ground_truth
        groundtruth = torch.diag_embed(tracking_mask)
        # groundtruth = groundtruth.type_as(similarity)

        # get loss
        groundtruth = groundtruth.view(-1, 1)
        similarity = similarity.view(-1, 1)
        similarity = torch.cat((1 - similarity, similarity), dim=1)
        class_mask = torch.zeros(similarity.shape[0], similarity.shape[1]).cuda()
        class_mask.scatter_(1, groundtruth.view(-1, 1).long(), 1.)
        probs = (similarity * class_mask).sum(dim=1).view(-1, 1)
        probs = probs.clamp(min=0.0001, max=1.0)

        log_p = probs.log()
        alpha = torch.ones(similarity.shape[0], similarity.shape[1]).cuda()
        alpha[:, 0] = alpha[:, 0] * (1 - self.alpha)
        alpha[:, 1] = alpha[:, 1] * self.alpha
        alpha = (alpha * class_mask).sum(dim=1).view(-1, 1)
        batch_loss = -alpha * (torch.pow((1 - probs), self.gamma)) * log_p
        loss = batch_loss.mean()
        #
        # loss = F.binary_cross_entropy_with_logits(
        # 	similarity, groundtruth, reduction='none') * focal_weight
        # loss = torch.mean(loss)
        return loss

# class IDloss(nn.Module):
#     def __init__(self):
#         super(IDloss, self).__init__()
#         self.only_neg_loss = _only_neg_loss
#         self.sigma = 1
#
#     def forward(self, cur_reid, pre_reid, cts_int, tracking, tracking_mask, pre_cts_int, opt):
#         # cts_int and pre_cts_int are ground truth
#         sigma = self.sigma
#         mask = tracking_mask[:, :, 0]
#         # ret['tracking'][k] = pre_ct - ct_int
#         # ret['ind'][k] = ct_int[1] * self.opt.output_w + ct_int[0]
#         cur_inds = cts_int[:, :, 1]*opt.output_w + cts_int[:, :, 0]
#         # tracking is predicted
#         tracking = _tranpose_and_gather_feat(tracking, cur_inds)
#         pre_cts = cts_int.float() + tracking
#         pre_inds = pre_cts[:, :, 1]*opt.output_w + pre_cts[:, :, 0]
#
#         pre_feats = _tranpose_and_gather_feat(pre_reid, pre_inds.long())
#         pre_feats = F.normalize(pre_feats, p=2, dim=2, eps=1e-5)
#         cur_feats = _tranpose_and_gather_feat(cur_reid, cur_inds)
#         cur_feats = F.normalize(cur_feats, p=2, dim=2, eps=1e-5)
#         out = torch.sum(pre_feats * cur_feats, dim=2)
#
#         x = pre_cts[:, :, 0].float() - pre_cts_int[:, :, 0].float()
#         y = pre_cts[:, :, 1].float() - pre_cts_int[:, :, 1].float()
#         target = torch.pow(-(x * x + y * y) / (2 * sigma * sigma), 2)
#         target = mask * target
#
#         neg_loss = self.only_neg_loss(out, target)
#
#         num_pos = mask.sum()
#         pos_loss = torch.log(out) * torch.pow(1 - out, 2) * \
#                    mask
#         pos_loss = pos_loss.sum()
#         if num_pos == 0:
#             return - neg_loss
#         return - (pos_loss + neg_loss) / num_pos
