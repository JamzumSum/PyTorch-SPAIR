from torch.nn.functional import grid_sample, affine_grid
import torch

class STN(torch.nn.Module):
    def __init__(self, object_shape):
        '''
        object_shape: [H_obj, W_obj]
        '''
        torch.nn.Module.__init__(self)
        self.output_shape = object_shape
        
    def forward(self, image, z_where, inverse=False):
        """
        spatial transformer network used to scale and shift input according to z_where in:
                1/ x -> x_att   -- shapes (H, W) -> (attn_window, attn_window) -- thus inverse = False
                2/ y_att -> y   -- (attn_window, attn_window) -> (H, W) -- thus inverse = True

        image: [N, C, H_img, W_img]
        z_where: [N, 4]
        """

        device = image.device
        y, x, height, width = torch.chunk(z_where, 4, dim=-1)

        batch_size = image.shape[0]
        nchannel = image.shape[1]
        out_dims = [batch_size, nchannel] + list(self.output_shape) # [N, C, obj_h, obj_w]

        # Important: in order for scaling to work, we need to convert from top left corner of bbox to center of bbox
        y = (y ) * 2 - 1
        x = (x ) * 2 - 1

        theta = torch.zeros(2, 3).repeat(batch_size, 1, 1).to(device)

        # set scaling
        theta[:, 0, 0] = width.squeeze(1)
        theta[:, 1, 1] = height.squeeze(1)
        # set translation
        theta[:, 0, -1] = x.squeeze(1)
        theta[:, 1, -1] = y.squeeze(1)

        # inverse == upsampling
        if inverse:
            # convert theta to a square matrix to find inverse
            t = torch.Tensor([0., 0., 1.]).repeat(batch_size, 1, 1).to(device)
            t = torch.cat([theta, t], dim=-2)
            t = t.inverse()
            assert not torch.isnan(t.detach()).max()
            theta = t[:, :2, :]

        # 2. construct sampling grid
        grid = affine_grid(theta, out_dims, align_corners=True)

        # 3. sample image from grid
        input_glimpses = grid_sample(image, grid, padding_mode='border', align_corners=True)    # [N, C, obj_h, obj_w]

        assert not torch.isnan(input_glimpses.detach()).max()
        return input_glimpses.flatten(1)    # [N, C * obj_h * obj_w]