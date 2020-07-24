from itertools import product

import torch
from numpy import prod
from torch.nn import Parameter

from .Extractor import ExtractNet  # CNN
from .STN import STN
from .utils import BareMLP, ODComponent
from torch.distributions import Normal, Bernoulli

class Encoder(ODComponent):
    device = torch.device('cpu')

    def __init__(
        self, cell_size, obj_shape, hidden_size, in_shape, 
        box_range, anchor_shape,
        F, A=50, neighbor = 1
    ):
        '''
        cell_size: [c_h, c_w] \\
        obj_shape: [H_obj, W_obj] \\
        hidden_size: [2, 2] or [2] \\
        in_shape: [C, H_img, W_img] \\
        box_range: [yx_min, yx_max, hw_min, hw_max] \\
        anchor_shape: [a_h, a_w] \\
        F: how many features will the CNN extract \\
        A: how many features will be used to describe the graph.
        '''
        ODComponent.__init__(
            self, in_shape, cell_size
        )

        self.box_range = box_range
        self.anchor_shape = anchor_shape

        self.A = A
        self.size = [4, A, 1, 1]
        self.neighbor = neighbor
        self.get_edge()

        self.image_size = torch.IntTensor(in_shape[1:])
        self.__box_args = torch.FloatTensor(box_range).view(2, 2).repeat(1, 2).view(4, 2)
        self.__anchor_shape = torch.IntTensor(anchor_shape)
        self.__hw = torch.IntTensor([self.H, self.W])

        # build encoder net
        self.Extracter = ExtractNet(
            in_shape=self.img_shape[-2:], 
            out_shape=(self.H, self.W),
            in_channel=in_shape[0],
            out_channel=F
        )
        self.LAT = BareMLP(
            F + (A + 6) * ((neighbor + 1) ** 2 - 1), 11,       # param_where as 2*4, depth as 2 * 1, param_pres as 1
            hidden_size[0]
        )
        self.stn = STN(obj_shape)
        self.OBJ = BareMLP(self.img_shape[0] * prod(obj_shape), 2 * A, hidden_size[1])
        
    def build_box(self, z_where, i: int, j: int):
        '''
        z_where: [N, 4] \\
        return: 
            box (relative to a cell): cell_y, cell_x, height, width
            obj_box (relative to the image): y_center, x_center, height, width
        '''
        device = z_where.device
        z_where = torch.sigmoid(z_where)    # [4]: y, x, h, w
        box_args = self.__box_args.to(device)
        box = (1 - z_where) * box_args[:, 0] + z_where * box_args[:, 1]
        # [N, 4]. cell_y, cell_x, height, width
        # cell_x and cell_y: see Eq(2. 4. )
        # height: sigmoid(z_h)
        # weight: sigmoid(z_w)

        image_size = self.image_size.to(device)
        anchor_size = self.__anchor_shape.to(device)
        hw = self.__hw.to(device)
        ij = torch.IntTensor([i, j]).to(device)

        obj_size = box[:, 2:] * anchor_size / image_size        # [N, 2]
        obj_center = (box[:, :2] + ij) / hw                       # [N, 2]

        del image_size, anchor_size, hw, ij
        return box, torch.cat((obj_center, obj_size), dim=-1)         # [N, 4]

    def get_edge(self):
        epsilon = torch.rand(sum(self.size), dtype=torch.float32)
        where, what, depth, pres = torch.split(epsilon, self.size)
        where = torch.sigmoid(where)
        depth = torch.sigmoid(depth)
        pres = torch.sigmoid(pres)
        self.edge = Parameter(torch.cat((where, what, depth, pres)), True)

    def to(self, device, *args, **argv):
        self.device = device
        ODComponent.to(self, device, *args, **argv)

    def forward(self, X):
        '''
        X: [N, C, H_img, W_img] \\
        output:
            z_where,    [N, H*W, 4]
            z_what,     [N, H*W, A]
            z_depth,    [N, H*W, 1]
            z_pres,     [N, H*W, 1]
            args, dict
        '''
        features = self.Extracter(X).permute(0, 2, 3, 1)        # [N, H, W, F]
        assert not torch.isnan(features.detach()).max()
        
        batch_size = X.shape[0]

        box = torch.empty(batch_size, self.H, self.W, 4, dtype=torch.float32, device=self.device)
        where = torch.empty(batch_size, self.H, self.W, 4, dtype=torch.float32, device=self.device)
        what = torch.empty(batch_size, self.H, self.W, self.A, dtype=torch.float32, device=self.device)
        depth = torch.empty(batch_size, self.H, self.W, 1, dtype=torch.float32, device=self.device)
        pres = torch.empty(batch_size, self.H, self.W, 1, dtype=torch.float32, device=self.device)
        
        param_where = torch.empty(batch_size, self.H, self.W, 2, 4, dtype=torch.float32, device=self.device)
        param_what = torch.empty(batch_size, self.H, self.W, 2, self.A, dtype=torch.float32, device=self.device)
        param_depth = torch.empty(batch_size, self.H, self.W, 2, 1, dtype=torch.float32, device=self.device)
        param_pres = torch.empty(batch_size, self.H, self.W, 1, dtype=torch.float32, device=self.device)

        edge = self.edge.repeat(batch_size, 1)
        def get_neighbor(i: int, j: int):
            xy = [torch.arange(-self.neighbor, 1)] * 2
            grid = torch.stack(torch.meshgrid(xy)).permute(1, 2, 0).reshape(-1, 2)[:-1] # [neighbor ^ 2 - 1, 2]
            grid[:, 0] += i
            grid[:, 1] += j
            
            return torch.cat(
                [
                    torch.cat((
                        box[:, i, j],   # [N, 4]
                        what[:, i, j],  # [N, A]
                        depth[:, i, j], # [N, 1]
                        pres[:, i, j]   # [N, 1]
                    ), dim=-1) if i >=0 and j >= 0 else edge for i, j in grid
                ], 
                dim=-1
            )   # [N, (A + 6) * neighbor ^ 2 - 1]
        norm_sample = lambda mu, sigma: mu + torch.randn_like(mu) * sigma
        gumbel_append = lambda beta: beta - torch.log(-torch.log(torch.rand_like(beta)))

        for H, W in product(range(self.H), range(self.W)):
            cat = torch.cat(
                (features[:, H, W], get_neighbor(H, W)),    # (A + 6) * neighbor ^ 2 - 1 + F
                dim=-1
            )
            params = self.LAT(cat)          # [N, 11]
            assert not torch.isnan(params.detach()).max()
            param_where[:, H, W, 0] = params[:, :4]
            param_where[:, H, W, 1] = params[:, 4:8].exp()
            box[:, H, W], obj_box = self.build_box(
                norm_sample(param_where[:, H, W, 0], param_where[:, H, W, 1]),
                H, W
            )
            where[:, H, W] = obj_box
            '''
            NOTE: Do NOT assign output#2 to where[:, H, W] deirectly!!!
                Since where is referenced in STN, the next modify of where will cut off BP.
                Consequentlly loss.backward raises fatal error.
            '''

            param_depth[:, H, W, 0, 0] = params[:, 9]
            param_depth[:, H, W, 1, 0] = params[:, 10].exp()
            depth[:, H, W, 0] = norm_sample(param_depth[:, H, W, 0, 0], param_depth[:, H, W, 1, 0])

            param_pres[:, H, W, 0] = torch.sigmoid(params[:, -1])       # [N]
            pi = torch.stack((1 - param_pres[:, H, W, 0], param_pres[:, H, W, 0]), 1)       # [N, 2]
            pres[:, H, W, 0] = torch.softmax(gumbel_append(pi) / 1e-9, dim=1)[:, 1]       # [N]

            params = self.OBJ(self.stn(X, obj_box))   # [N, 2A]
            assert not torch.isnan(params.detach()).max()
            param_what[:, H, W, 0] = params[:, :self.A]
            param_what[:, H, W, 1] = params[:, self.A:].exp()
            what[:, H, W] = norm_sample(param_what[:, H, W, 0], param_what[:, H, W, 1])

        return (
            where.flatten(1, 2), 
            what.flatten(1, 2), 
            depth.flatten(1, 2), 
            pres.flatten(1, 2), 
            {
                'where': Normal(param_where[..., 0, :], param_where[..., 1, :]),
                'what': Normal(param_what[..., 0, :], param_what[..., 1, :]),
                'depth': Normal(param_depth[..., 0, :], param_depth[..., 1, :]),
                'pres': param_pres,
            }
        )
