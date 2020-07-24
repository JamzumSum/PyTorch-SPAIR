from torch import nn
from collections import OrderedDict

class ExtractNet(nn.Sequential):
    '''
    Builds the CNN network for feature extraction.
    :param input_shape: H * W
    :param in_channel: 3 for RGB image, 1 for gray image
    :param out_channel:
    :return:
    '''
    def __init__(self, in_shape, out_shape, in_channel, out_channel):
        self.input_shape = in_shape
        raw_net = OrderedDict()
        H, W = out_shape
        input_size = in_shape[0] / 2
        out_size = input_size / 2
        temp = input_size
        count = 1
        raw_net['Conv_0'] = nn.Conv2d(in_channels=in_channel, out_channels=128, kernel_size=4, stride=2, padding=1)
        while temp != H:
            if H <= out_size:
                raw_net['Conv_%s'% count] = nn.Conv2d(in_channels=128, out_channels=128, kernel_size=4, stride=2, padding=1)  # 对输出图像尺寸裁一半
                input_size = out_size
                temp = input_size
                out_size = out_size / 2
                count += 1
            else:
                kz = input_size - H + 3
                raw_net['Conv_%s'% count] = nn.Conv2d(in_channels=128, out_channels=128, kernel_size=int(kz), stride=1, padding=1)
                temp = H
                count += 1
        raw_net['Conv_%s'% count] = nn.Conv2d(in_channels=128, out_channels=128, kernel_size=1, stride=1)
        raw_net['Conv_%s'% (count + 1)] = nn.Conv2d(in_channels=128, out_channels=128, kernel_size=1, stride=1)
        raw_net['Conv_out'] = nn.Conv2d(in_channels=128, out_channels=out_channel, kernel_size=1, stride=1)
        nn.Sequential.__init__(self, raw_net)
