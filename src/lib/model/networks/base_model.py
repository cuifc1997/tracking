from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import torch
from torch import nn
try:
    from .DCNv2.dcn_v2 import DCN
except:
    print('import DCN failed')
    DCN = None

def fill_fc_weights(layers):
	for m in layers.modules():
		if isinstance(m, nn.Conv2d):
			if m.bias is not None:
				nn.init.constant_(m.bias, 0)

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
		self.maxpool = nn.MaxPool2d(kernel_size=4, stride=4, padding=1)
		self.shift = nn.Sequential(
			DCN(128, 64, kernel_size=(3, 3), stride=1, padding=1, dilation=1, deformable_groups=1),
			nn.BatchNorm2d(64),
			nn.ReLU(inplace=True)
		)

		self.tracking_uv = nn.Sequential(
			# nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1, bias=True),
			DCN(64, 64, kernel_size=(3, 3), stride=1, padding=1, dilation=1, deformable_groups=1),
			nn.BatchNorm2d(64),
			nn.ReLU(inplace=True),
			nn.Conv2d(64, 2, kernel_size=1, stride=1, padding=0, bias=True)
		)

		# ATTENTION
		self.avg_pool = nn.AdaptiveAvgPool2d((1,1))
		self.max_pool = nn.AdaptiveMaxPool2d((1,1))
		self.share_layer_1 = nn.Sequential(
			nn.Linear(64, 16),
			nn.ReLU(inplace=True),
			nn.Linear(16, 64),
			nn.ReLU(inplace=True)
		)
		self.sigmoid = nn.Sigmoid()
		self.share_layer_2 = nn.Sequential(
			DCN(5, 5, kernel_size=(3, 3), stride=1, padding=1, dilation=1, deformable_groups=1),
			nn.ReLU(inplace=True),
			DCN(5, 1, kernel_size=(3, 3), stride=1, padding=1, dilation=1, deformable_groups=1),
			nn.Sigmoid()
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

	def forward(self, x, pre_img=None, pre_hm=None):
		if (pre_hm is not None) or (pre_img is not None):
			feats, pre_feats = self.imgpre2feats(x=x, pre_img=pre_img, pre_hm=pre_hm)
		else:
			feats = self.img2feats(x)

		feat_1 = feats[0]
		feat_2 = pre_feats[0]

		pre_hm_pool = self.maxpool(pre_hm)
		# feat = torch.cat((feat_1, feat_2), dim=2)
		for i in range(feat_1.shape[1]):
			if i == 0:
				feat_shift = torch.cat((feat_1[:, 0, :, :].unsqueeze(1), feat_2[:, 0, :, :].unsqueeze(1)), dim=1)
			else:
				layer_feat = torch.cat((feat_1[:, i, :, :].unsqueeze(1), feat_2[:, i, :, :].unsqueeze(1)), dim=1)
				feat_shift = torch.cat((feat_shift, layer_feat), dim=1)

		feat_shift = self.shift(feat_shift)
		tracking = self.tracking_uv(feat_shift)

		avg_pool = torch.mean(feat_2, dim=1).unsqueeze(1)
		max_pool = torch.max(feat_2, dim=1)[0].unsqueeze(1)
		space_attention = torch.cat((avg_pool, max_pool), dim=1)
		space_attention = torch.cat((space_attention, tracking), dim=1)
		space_attention = torch.cat((space_attention, pre_hm_pool), dim=1)
		space_attention = self.share_layer_2(space_attention)
		feat_fusion = torch.mul(space_attention, feat_2)

		feat_fusion = feat_fusion + feat_1

		out = []
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
					else:
						z[head] = self.__getattr__(head)(feats[s])
				out.append(z)
		return out
