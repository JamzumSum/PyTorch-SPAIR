from .Encoder import Encoder
from .Decoder import Decoder
from torch.nn import Sequential, Module

class SPAIR(Sequential):
    encoder: Encoder
    decoder: Decoder

    def __init__(
        self, cell_size, obj_shape, hidden_size, img_shape, 
        box_range, anchor_shape,
        F, A, neighbor = 1
    ):
        Sequential.__init__(self)
        self.add_module('encoder', Encoder(
            cell_size, obj_shape, hidden_size[:2], img_shape, 
            box_range, anchor_shape,
            F, A, neighbor
        ))
        self.add_module('decoder', Decoder(img_shape, obj_shape, hidden_size[2:], cell_size, A))
    
    def to(self, device, *args, **argv):
        self.encoder.device = device
        self.decoder.device = device
        Sequential.to(self, device, *args, **argv)

    def forward(self, X, bg=None):
        return self.decoder(*self.encoder(X)[:-1], bg)
    