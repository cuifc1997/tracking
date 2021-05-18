from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import torch
import numpy as np
from torch.nn import functional as F
from torch import nn
try:
    from .DCNv2.dcn_v2 import DCN
except:
    print('import DCN failed')
    DCN = None
from .DCNv2.dcn_v2 import DCN_TraDeS

def fill_fc_weights(layers):
	for m in layers.modules():
		if isinstance(m, nn.Conv2d):
			if m.bias is not None:
				nn.init.constant_(m.bias, 0)

def _gather_feat(feat, ind):
    dim = feat.size(2)
    ind = ind.unsqueeze(2).expand(ind.size(0), ind.size(1), dim)
    feat = feat.gather(1, ind)
    return feat

def _tranpose_and_gather_feat(feat, ind):
    feat = feat.permute(0, 2, 3, 1).contiguous()
    feat = feat.view(feat.size(0), -1, feat.size(3))
    feat = _gather_feat(feat, ind)
    return feat

class BaseModel(nn.Module):
	def __init__(self, heads, head_convs, num_stacks, last_channel, opt=None):
		super(BaseModel, self).__init__()
		if opt is not None and opt.head_kernel != 3:
			print('Using head kernel:', opt.head_kernel)
			head_kernel = opt.head_kernel
		else:
			head_kernel = 3
		self.num_stacks = num_stacks
		self.heads = heads

		# TODO: ADD SOMETHING
		self.avgpool = nn.AvgPool2d(kernel_size=4, stride=4)
		self.shift = nn.Sequential(
			nn.Conv3d(128, 128, kernel_size=(2, 3, 3), stride=(1, 1, 1), padding=(0, 1, 1), groups=128),
			nn.BatchNorm3d(128),
			nn.ReLU(inplace=True),
			nn.Conv3d(128, 64, kernel_size=(1, 3, 3), stride=(1, 1, 1), padding=(0, 1, 1), groups=1),
			nn.BatchNorm3d(64),
			nn.ReLU(inplace=True)
		)

		self.shift_u = nn.Conv2d(64, 9, kernel_size=1, stride=1, padding=0, groups=1)
		self.tracking_u = nn.Conv2d(9, 1, kernel_size=1, stride=1, padding=0, groups=1)
		self.shift_v = nn.Conv2d(64, 9, kernel_size=1, stride=1, padding=0, groups=1)
		self.tracking_v = nn.Conv2d(9, 1, kernel_size=1, stride=1, padding=0, groups=1)
		self.dcn1_1 = DCN_TraDeS(64, 64, kernel_size=(3, 3), stride=1, padding=1, deformable_groups=1)

		self.re_ID = nn.Sequential(
			nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1, bias=True),
			nn.BatchNorm2d(64),
			nn.ReLU(inplace=True),
			nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1, bias=True),
			nn.BatchNorm2d(64),
			nn.ReLU(inplace=True),
			nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1, bias=True),
			# nn.BatchNorm2d(128),
			# nn.ReLU(inplace=True)
		)


		# ATTENTION
		self.attention = nn.Sequential(
			nn.Conv2d(128, 2, kernel_size=3, stride=1, padding=1, bias=True),
			nn.BatchNorm2d(2),
			nn.ReLU(inplace=True)
		)

		for head in self.heads:
			classes = self.heads[head]
			head_conv = head_convs[head]
			if len(head_conv) > 0:
				out = nn.Conv2d(head_conv[-1], classes,
				                kernel_size=1, stride=1, padding=0, bias=True)
				conv = nn.Conv2d(last_channel, head_conv[0],
				                 kernel_size=head_kernel,
				                 padding=head_kernel // 2, bias=True)
				convs = [conv]
				for k in range(1, len(head_conv)):
					convs.append(nn.Conv2d(head_conv[k - 1], head_conv[k],
					                       kernel_size=1, bias=True))
				if len(convs) == 1:
					fc = nn.Sequential(conv, nn.ReLU(inplace=True), out)
				elif len(convs) == 2:
					fc = nn.Sequential(
						convs[0], nn.ReLU(inplace=True),
						convs[1], nn.ReLU(inplace=True), out)
				elif len(convs) == 3:
					fc = nn.Sequential(
						convs[0], nn.ReLU(inplace=True),
						convs[1], nn.ReLU(inplace=True),
						convs[2], nn.ReLU(inplace=True), out)
				elif len(convs) == 4:
					fc = nn.Sequential(
						convs[0], nn.ReLU(inplace=True),
						convs[1], nn.ReLU(inplace=True),
						convs[2], nn.ReLU(inplace=True),
						convs[3], nn.ReLU(inplace=True), out)
				if 'hm' in head:
					fc[-1].bias.data.fill_(opt.prior_bias)
				else:
					fill_fc_weights(fc)
			else:
				fc = nn.Conv2d(last_channel, classes,
				               kernel_size=1, stride=1, padding=0, bias=True)
				if 'hm' in head:
					fc.bias.data.fill_(opt.prior_bias)
				else:
					fill_fc_weights(fc)
			self.__setattr__(head, fc)

	def img2feats(self, x):
		raise NotImplementedError

	def imgpre2feats(self, x, pre_img=None, pre_hm=None):
		raise NotImplementedError

	def forward(self, x, pre_img=None, pre_hm=None, pre_inds=None):
		if (pre_hm is not None) or (pre_img is not None):
			feats, pre_feats = self.imgpre2feats(x=x, pre_img=pre_img, pre_hm=None)
		else:
			feats = self.img2feats(x)
		out = []

		cur_feat = feats[0]
		pre_feat = pre_feats[0]

		cur_reid = self.re_ID(cur_feat)
		pre_reid = self.re_ID(pre_feat)

		# tracking
		feat_shift = torch.cat((cur_reid.unsqueeze(2), pre_reid.unsqueeze(2)), dim=2)
		feat_shift = self.shift(feat_shift)
		shift_u = self.shift_u(feat_shift.squeeze(2))
		trackinv_u = self.tracking_u(shift_u)
		shift_v = self.shift_v(feat_shift.squeeze(2)) # batch_size x 9 x h x w
		trackinv_v = self.tracking_v(shift_u)
		tracking = torch.cat((trackinv_u, trackinv_v), dim=1)

		# deformable conv
		deform_shift = torch.cat((shift_v, shift_u), dim=2).view(tracking.size(0), 9*2, tracking.size(2), tracking.size(3))
		deform_mask = torch.tensor(np.ones((tracking.shape[0], 9, tracking.shape[2], tracking.shape[3]), dtype=np.float32)).to(self.opt.device)
		pre_hm_avg = self.avgpool(pre_hm)
		pre_feat = torch.mul(pre_feat, pre_hm_avg)
		pre_feat = self.dcn1_1(pre_feat, deform_shift, deform_mask)

		cat_feats = torch.cat((cur_feat, pre_feat), dim=1)
		attention = self.attention(cat_feats)
		attention = F.softmax(attention, dim=1)
		feat_fusion = torch.mul(pre_feat, attention[:, 1, :, :].unsqueeze(1)) + torch.mul(cur_feat, attention[:, 0, :, :].unsqueeze(1))

		feats[0] = feat_fusion

		if self.opt.model_output_list:
			for s in range(self.num_stacks):
				z = []
				for head in sorted(self.heads):
					z.append(self.__getattr__(head)(feats[s]))
				out.append(z)
		else:
			for s in range(self.num_stacks):
				z = {}
				for head in self.heads:
					if head == 'tracking':
						z['tracking'] = tracking
						# z['attention'] = space_attention
					else:
						z[head] = self.__getattr__(head)(feats[s])
				z['cur_reid'] = cur_reid
				z['pre_reid'] = pre_reid
				out.append(z)
		return out

