import torch
from numpy import prod

from .STN import STN
from .utils import BareMLP, ODComponent


class Decoder(ODComponent):
	def __init__(self, bg_shape, obj_shape, hidden_size, cell_size, A):
		'''
		bg_shape: [C, H_img, W_img] or [N, C, H_img, W_img]\\
		obj_shape: see Encoder.__init__ \\
		hidden_size: 
		cell_size: see Encoder.__init__ \\
		A: see Encoder.__init__
		'''
		ODComponent.__init__(self, bg_shape, cell_size)
		self.bg_shape = bg_shape
		self.obj_shape = obj_shape
		self.nchannel = bg_shape[-3]
		self.OBJ = BareMLP(A, (self.nchannel + 1) * prod(obj_shape), hidden_size)
		self.stn = STN(self.img_shape[-2:])
			
	def forward(self, where, what, depth, pres, bg=None):
		'''
		where,    [N, H*W, 4] \\
		what,     [N, H*W, A] \\
		depth,    [N, H*W, 1] \\
		pres      [N, H*W, 1]
		'''
		device = where.device
		if bg is None: bg = torch.zeros(self.bg_shape, device=device)

		N, nobj, _ = what.shape
		objs = self.OBJ(what)       # [N, nobj, 4 * H_obj * W_obj]
		assert not torch.isnan(objs.detach()).max()
		pics = objs.reshape(N, nobj, *self.obj_shape, self.nchannel + 1)   # [N, H*W, H_obj, W_obj, C + 1]

		pics[..., :-1] *= pres.view(N, nobj, 1, 1, 1)
		importance = pics[..., -1:] * torch.sigmoid(-depth).reshape(N, nobj, 1, 1, 1)    # [N, H*W, H_obj, W_obj, 1]

		# flatten them
		pics = pics.flatten(0, 1)                   # [N*H*W, H_obj, W_obj, C + 1]
		importance = importance.flatten(0, 1)       # [N*H*W, H_obj, W_obj, 1]
		where = where.flatten(0, 1)                 # [N*H*W, 4]

		pics, alpha, importance = torch.split(                                  # [N, H*W, H_img, W_img, C / 1 / 1]
			self.stn(
				torch.cat((pics, importance), dim=-1).permute(0, 3, 1, 2), 
				where, True
			).view(N, nobj, *self.img_shape[-2:], self.nchannel + 2), 
			split_size_or_sections=[self.nchannel, 1, 1], 
			dim=-1
		)

		out = bg.repeat(N, 1, 1, 1) if bg.dim() == 3 else bg        # [N, C, H_img, W_img]
		out = out.to(device).permute(0, 2, 3, 1).unsqueeze(1)      # [N, 1, H_img, W_img, C]
		pics = alpha * pics + (1 - alpha) * out
		importance = importance / (importance.sum(dim=1, keepdim=True).clamp(min=1e-6))  # [N, H*W, H_img, W_img, 1]
		assert not torch.isnan(importance.detach()).max()
		del alpha

		out = (pics * importance).sum(dim=1)                    	# [N, H_img, W_img, C]
		assert not torch.isnan(out.detach()).max()
		return out.clamp(0, 1).permute(0, 3, 1, 2)
