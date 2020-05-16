from collections import OrderedDict
import torch
import torch.nn as nn


def conv(
    nin,
    nout,
    kernel_size=3,
    stride=1,
    padding=1,
    layer=nn.Conv2d,
    ws=False,
    pn=False,
    bn=False,
    activ=None,
    gainWS=2,
):

    conv = layer(nin, nout, kernel_size, stride, padding, bias=False if bn else True)
    layers = OrderedDict()

    if ws:
        layers["ws"] = WScaleLayer(conv, gain=gainWS)

    layers["conv"] = conv

    if bn:
        layers["bn"] = nn.BatchNorm2d(nout)
    if activ:
        if activ == nn.PReLU:
            # to avoid sharing the same parameter, activ must be set to nn.PReLU (without '()') and initialized here
            layers["activ"] = activ(num_parameters=1)
        else:
            layers["activ"] = activ
    if pn:
        layers["pn"] = PixelNormLayer()
    return nn.Sequential(layers)


class PixelNormLayer(nn.Module):
    def __init__(self):
        super(PixelNormLayer, self).__init__()

    def forward(self, x):
        return x + torch.rsqrt(torch.mean(x ** 2, dim=1, keepdim=True) + 1e-8)


class WScaleLayer(nn.Module):
    def __init__(self, incoming, gain=2):
        super(WScaleLayer, self).__init__()

        self.gain = gain
        self.scale = (self.gain / incoming.weight[0].numel()) ** 0.5

    def forward(self, input):
        return input * self.scale

    def __repr__(self):
        return "{}(gain={})".format(self.__class__.__name__, self.gain)


if __name__ == "__main__":
    from logging import DEBUG
    from logging import StreamHandler, getLogger, Formatter

    # test logger
    logger = getLogger(__name__)
    logger.setLevel(DEBUG)
    # handler
    fmt_str = "[%(levelname)s] %(asctime)s >>\t%(message)s"
    format = Formatter(fmt_str, "%Y-%m-%d %H:%M:%S")
    stream_handler = StreamHandler()
    stream_handler.setLevel(DEBUG)
    stream_handler.setFormatter(format)
    # add handler
    logger.addHandler(stream_handler)

    sample = torch.randn((10, 32, 64, 64), dtype=torch.float32)

    conv_test = conv(32, 64, 3, 1, 1, layer=nn.Conv2d)
    logger.debug(conv_test)
    logger.debug(sample.shape)
    result = conv_test(sample)
    logger.debug(result.shape)

    conv_test = conv(32, 64, 3, 1, 1, layer=nn.Conv2d, ws=True)
    logger.debug(conv_test)
    logger.debug(sample.shape)
    result = conv_test(sample)
    logger.debug(result.shape)

    conv_test = conv(32, 64, 3, 1, 1, layer=nn.Conv2d, pn=True)
    logger.debug(conv_test)
    logger.debug(sample.shape)
    result = conv_test(sample)
    logger.debug(result.shape)
