import torch
import torch.nn as nn
import torch.nn.functional as F
import math


def _is_static_pad(kernel_size, stride=1, dilation=1, **_):
    return stride == 1 and (dilation * (kernel_size - 1)) % 2 == 0


def _get_padding(kernel_size, stride=1, dilation=1, **_):
    padding = ((stride - 1) + dilation * (kernel_size - 1)) // 2
    return padding


def _calc_same_pad(i, k, s, d):
    return max((math.ceil(i / s) - 1) * s + (k - 1) * d + 1 - i, 0)


def _split_channels(num_chan, num_groups):
    split = [num_chan // num_groups for _ in range(num_groups)]
    split[0] += num_chan - sum(split)
    return split


class ChannelToSpace(nn.Module):

    def __init__(self, upscale_factor=2):
        super().__init__()
        self.bs = upscale_factor

    def forward(self, x):
        N, C, H, W = x.size()
        x = x.view(N, self.bs, self.bs, C // (self.bs ** 2), H, W)  # (N, bs, bs, C//bs^2, H, W)
        x = x.permute(0, 3, 4, 1, 5, 2).contiguous()  # (N, C//bs^2, H, bs, W, bs)
        x = x.view(N, C // (self.bs ** 2), H * self.bs, W * self.bs)  # (N, C//bs^2, H * bs, W * bs)
        return x


class SpaceToChannel(nn.Module):

    def __init__(self, downscale_factor=2):
        super().__init__()
        self.bs = downscale_factor

    def forward(self, x):
        N, C, H, W = x.size()
        x = x.view(N, C, H // self.bs, self.bs, W // self.bs, self.bs)  # (N, C, H//bs, bs, W//bs, bs)
        x = x.permute(0, 3, 5, 1, 2, 4).contiguous()  # (N, bs, bs, C, H//bs, W//bs)
        x = x.view(N, C * (self.bs ** 2), H // self.bs, W // self.bs)  # (N, C*bs^2, H//bs, W//bs)
        return x


class Conv2dSame(nn.Conv2d):
    """ Tensorflow like 'SAME' convolution wrapper for 2D convolutions
    """
    def __init__(self, in_channels, out_channels, kernel_size, stride=1,
                 padding=0, dilation=1, groups=1, bias=True):
        super(Conv2dSame, self).__init__(
            in_channels, out_channels, kernel_size, stride, 0, dilation,
            groups, bias)

    def forward(self, x):
        ih, iw = x.size()[-2:]
        kh, kw = self.weight.size()[-2:]
        pad_h = _calc_same_pad(ih, kh, self.stride[0], self.dilation[0])
        pad_w = _calc_same_pad(iw, kw, self.stride[1], self.dilation[1])
        if pad_h > 0 or pad_w > 0:
            x = F.pad(x, [pad_w//2, pad_w - pad_w//2, pad_h//2, pad_h - pad_h//2])
        return F.conv2d(x, self.weight, self.bias, self.stride,
                        self.padding, self.dilation, self.groups)


def conv2d_pad(in_chs, out_chs, kernel_size, **kwargs):
    padding = kwargs.pop('padding', '')
    kwargs.setdefault('bias', False)
    if isinstance(padding, str):
        # for any string padding, the padding will be calculated for you, one of three ways
        padding = padding.lower()
        if padding == 'same':
            # TF compatible 'SAME' padding, has a performance and GPU memory allocation impact
            if _is_static_pad(kernel_size, **kwargs):
                # static case, no extra overhead
                padding = _get_padding(kernel_size, **kwargs)
                return nn.Conv2d(in_chs, out_chs, kernel_size, padding=padding, **kwargs)
            else:
                # dynamic padding
                return Conv2dSame(in_chs, out_chs, kernel_size, **kwargs)
        elif padding == 'valid':
            # 'VALID' padding, same as padding=0
            return nn.Conv2d(in_chs, out_chs, kernel_size, padding=0, **kwargs)
        else:
            # Default to PyTorch style 'same'-ish symmetric padding
            padding = _get_padding(kernel_size, **kwargs)
            return nn.Conv2d(in_chs, out_chs, kernel_size, padding=padding, **kwargs)
    else:
        # padding was specified as a number or pair
        return nn.Conv2d(in_chs, out_chs, kernel_size, padding=padding, **kwargs)


class MuxConv(nn.Module):
    """ MuxConv
    """
    def __init__(self, in_channels, out_channels,
                 kernel_size=3, stride=1, padding='', scale_size=0, groups=1, depthwise=False, **kwargs):
        super(MuxConv, self).__init__()

        scale_size = scale_size if isinstance(scale_size, list) else [scale_size]
        assert len(set(scale_size)) > 1, "use regular convolution for faster inference"

        num_groups = len(scale_size)
        kernel_size = kernel_size if isinstance(kernel_size, list) else [kernel_size] * num_groups
        groups = groups if isinstance(groups, list) else [groups] * num_groups

        in_splits = _split_channels(in_channels, num_groups)
        out_splits = _split_channels(out_channels, num_groups)

        convs = []
        for k, in_ch, out_ch, scale, _group in zip(kernel_size, in_splits, out_splits, scale_size, groups):
            # padding = (k - 1) // 2
            if scale < 0:  # space-to-channel -> learn -> channel-to-space
                # if depthwise:
                _group = in_ch * 4
                convs.append(
                    nn.Sequential(
                        SpaceToChannel(2),
                        conv2d_pad(
                            in_ch * 4, out_ch * 4, k, stride=stride,
                            padding=padding, dilation=1, groups=_group, **kwargs),
                        ChannelToSpace(2),
                    )
                )
            elif scale > 0:  # channel-to-space -> learn -> space-to-channel
                # if depthwise:
                _group = in_ch // 4
                convs.append(
                    nn.Sequential(
                        ChannelToSpace(2),
                        conv2d_pad(
                            in_ch // 4, out_ch // 4, k, stride=stride,
                            padding=padding, dilation=1, groups=_group, **kwargs),
                        SpaceToChannel(2),
                    )
                )
            else:
                # if depthwise:
                _group = out_ch
                convs.append(
                    conv2d_pad(
                        in_ch, out_ch, k, stride=stride,
                        padding=padding, dilation=1, groups=_group, **kwargs))

        self.convs = nn.ModuleList(convs)
        self.splits = in_splits
        self.scale_size = scale_size

    def forward(self, x):
        x_split = torch.split(x, self.splits, 1)
        x_out = []
        for spx, conv in zip(x_split, self.convs):
            x_out.append(conv(spx))
        x = torch.cat(x_out, 1)
        return x


class MixedConv2d(nn.Module):
    """ Mixed Grouped Convolution
    Based on MDConv and GroupedConv in MixNet impl:
      https://github.com/tensorflow/tpu/blob/master/models/official/mnasnet/mixnet/custom_layers.py
    """

    def __init__(self, in_channels, out_channels, kernel_size=3,
                 stride=1, padding='', dilated=False, depthwise=False, **kwargs):
        super(MixedConv2d, self).__init__()

        kernel_size = kernel_size if isinstance(kernel_size, list) else [kernel_size]
        num_groups = len(kernel_size)
        in_splits = _split_channels(in_channels, num_groups)
        out_splits = _split_channels(out_channels, num_groups)
        if depthwise:
            conv_groups = out_splits
        else:
            groups = kwargs.pop('groups', 1)
            if groups > 1:
                conv_groups = _split_channels(groups, num_groups)
            else:
                conv_groups = [1] * num_groups

        for idx, (k, in_ch, out_ch) in enumerate(zip(kernel_size, in_splits, out_splits)):
            d = 1
            # FIXME make compat with non-square kernel/dilations/strides
            if stride == 1 and dilated:
                d, k = (k - 1) // 2, 3
            # conv_groups = out_ch if depthwise else kwargs.pop('groups', 1)
            # use add_module to keep key space clean
            self.add_module(
                str(idx),
                conv2d_pad(
                    in_ch, out_ch, k, stride=stride,
                    padding=padding, dilation=d, groups=conv_groups[idx], **kwargs)
            )
        self.splits = in_splits

    def forward(self, x):
        x_split = torch.split(x, self.splits, 1)
        x_out = [c(x) for x, c in zip(x_split, self._modules.values())]
        x = torch.cat(x_out, 1)
        return x


# helper method
def select_conv2d(in_chs, out_chs, kernel_size, **kwargs):
    scale_size = kwargs.pop('scales', 0)
    if isinstance(kernel_size, list) or isinstance(scale_size, list):
        # assert 'groups' not in kwargs  # only use 'depthwise' bool arg
        # We're going to use only lists for defining the MixedConv2d kernel groups,
        # ints, tuples, other iterables will continue to pass to normal conv and specify h, w.
        if isinstance(scale_size, list):
            return MuxConv(in_chs, out_chs, kernel_size, scale_size=scale_size, **kwargs)
        else:
            return MixedConv2d(in_chs, out_chs, kernel_size, **kwargs)
    else:
        depthwise = kwargs.pop('depthwise', False)
        groups = out_chs if depthwise else kwargs.pop('groups', 1)
        return conv2d_pad(in_chs, out_chs, kernel_size, groups=groups, **kwargs)

