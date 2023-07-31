# Copyright (c) OpenMMLab. All rights reserved.
import torch
import torch.nn as nn
from mmcv.cnn import ConvModule, DepthwiseSeparableConvModule
from mmcv.runner import BaseModule


class DarknetBottleneck(BaseModule):
    """The basic bottleneck block used in Darknet.

    Each ResBlock consists of two ConvModules and the input is added to the
    final output. Each ConvModule is composed of Conv, BN, and LeakyReLU.
    The first convLayer has filter size of 1x1 and the second one has the
    filter size of 3x3.

    Args:
        in_channels (int): The input channels of this Module.
        out_channels (int): The output channels of this Module.
        expansion (int): The kernel size of the convolution. Default: 0.5
        add_identity (bool): Whether to add identity to the out.
            Default: True
        use_depthwise (bool): Whether to use depthwise separable convolution.
            Default: False
        conv_cfg (dict): Config dict for convolution layer. Default: None,
            which means using conv2d.
        norm_cfg (dict): Config dict for normalization layer.
            Default: dict(type='BN').
        act_cfg (dict): Config dict for activation layer.
            Default: dict(type='Swish').
    """

    def __init__(self,
                 in_channels,
                 out_channels,
                 expansion=0.5,
                 add_identity=True,
                 use_depthwise=False,
                 conv_cfg=None,
                 norm_cfg=dict(type='BN', momentum=0.03, eps=0.001),
                 act_cfg=dict(type='Swish'),
                 init_cfg=None):
        super().__init__(init_cfg)
        hidden_channels = int(out_channels * expansion)
        conv = DepthwiseSeparableConvModule if use_depthwise else ConvModule
        self.conv1 = ConvModule(
            in_channels,
            hidden_channels,
            1,
            conv_cfg=conv_cfg,
            norm_cfg=norm_cfg,
            act_cfg=act_cfg)
        self.conv2 = conv(
            hidden_channels,
            out_channels,
            3,
            stride=1,
            padding=1,
            conv_cfg=conv_cfg,
            norm_cfg=norm_cfg,
            act_cfg=act_cfg)
        self.add_identity = \
            add_identity and in_channels == out_channels

    def forward(self, x):
        identity = x
        out = self.conv1(x)
        out = self.conv2(out)

        if self.add_identity:
            return out + identity
        else:
            return out


class CSPLayer(BaseModule):
    """Cross Stage Partial Layer.

    Args:
        in_channels (int): The input channels of the CSP layer.
        out_channels (int): The output channels of the CSP layer.
        expand_ratio (float): Ratio to adjust the number of channels of the
            hidden layer. Default: 0.5
        num_blocks (int): Number of blocks. Default: 1
        add_identity (bool): Whether to add identity in blocks.
            Default: True
        use_depthwise (bool): Whether to depthwise separable convolution in
            blocks. Default: False
        conv_cfg (dict, optional): Config dict for convolution layer.
            Default: None, which means using conv2d.
        norm_cfg (dict): Config dict for normalization layer.
            Default: dict(type='BN')
        act_cfg (dict): Config dict for activation layer.
            Default: dict(type='Swish')
    """

    def __init__(self,
                 in_channels,
                 out_channels,
                 expand_ratio=0.5,
                 num_blocks=1,
                 add_identity=True,
                 use_depthwise=False,
                 conv_cfg=None,
                 norm_cfg=dict(type='BN', momentum=0.03, eps=0.001),
                 act_cfg=dict(type='Swish'),
                 init_cfg=None):
        super().__init__(init_cfg)
        mid_channels = int(out_channels * expand_ratio)
        self.main_conv = ConvModule(
            in_channels,
            mid_channels,
            1,
            conv_cfg=conv_cfg,
            norm_cfg=norm_cfg,
            act_cfg=act_cfg)
        self.short_conv = ConvModule(
            in_channels,
            mid_channels,
            1,
            conv_cfg=conv_cfg,
            norm_cfg=norm_cfg,
            act_cfg=act_cfg)
        self.final_conv = ConvModule(
            2 * mid_channels,
            out_channels,
            1,
            conv_cfg=conv_cfg,
            norm_cfg=norm_cfg,
            act_cfg=act_cfg)

        self.blocks = nn.Sequential(*[
            DarknetBottleneck(
                mid_channels,
                mid_channels,
                1.0,
                add_identity,
                use_depthwise,
                conv_cfg=conv_cfg,
                norm_cfg=norm_cfg,
                act_cfg=act_cfg) for _ in range(num_blocks)
        ])

    def forward(self, x):
        x_short = self.short_conv(x)
        x_main = self.main_conv(x)
        x_main = self.blocks(x_main)
        x_final = torch.cat((x_main, x_short), dim=1)
        return self.final_conv(x_final)



class TinyCSPLayer(BaseModule):
    """
        "성공, 반드시 해야지."
    """

    def __init__(self,
                 in_channels,
                 out_channels,
                 expand_ratio=0.5,
                 num_blocks=1,
                 #add_identity=True,
                 #use_depthwise=False,
                 conv_cfg=None,
                 norm_cfg=dict(type='BN', momentum=0.03, eps=0.001),
                 # act_cfg=dict(type='Swish'),
                 act_cfg=dict(type='LeakyReLU'),
                 init_cfg=None):
        super().__init__(init_cfg)
        mid_channels = int(out_channels * expand_ratio)
        
        self.first_conv = ConvModule(
            in_channels,
            mid_channels,
            3,
            stride=1,
            padding=1,
            conv_cfg=conv_cfg,
            norm_cfg=norm_cfg,
            act_cfg=act_cfg)
        
        self.middle_conv = ConvModule(
            mid_channels,
            mid_channels,
            3,
            stride=1,
            padding=1,
            conv_cfg=conv_cfg,
            norm_cfg=norm_cfg,
            act_cfg=act_cfg)
        
        self.last_conv = ConvModule(
            2 * mid_channels,
            out_channels,
            1,
            conv_cfg=conv_cfg,
            norm_cfg=norm_cfg,
            act_cfg=act_cfg)
        
        self.short_conv = ConvModule(
            in_channels*2,
            in_channels,
            1,
            conv_cfg=conv_cfg,
            norm_cfg=norm_cfg,
            act_cfg=act_cfg)

    def forward(self, x):        
        f_out = self.first_conv(x)
        tiny_csp = self.middle_conv(f_out)
        m_out = torch.cat((tiny_csp, f_out),dim=1)
        l_out = self.last_conv(m_out)
        final = torch.cat((l_out, x), dim=1)
        final = self.short_conv(final)
        return final



class DWBottleneck(BaseModule):
    """
        마침내...
    """

    def __init__(self,
                 in_channels,
                 out_channels,
                 expansion=0.5,
                 add_identity=False,
                 use_depthwise=True,
                 conv_cfg=None,
                 norm_cfg=dict(type='BN', momentum=0.03, eps=0.001),
                 act_cfg=dict(type='LeakyReLU'),
                 init_cfg=None):
        super().__init__(init_cfg)
        hidden_channels = int(out_channels * expansion)
        conv = DepthwiseSeparableConvModule if use_depthwise else ConvModule
        self.conv1 = ConvModule(
            in_channels,
            hidden_channels,
            1,
            conv_cfg=conv_cfg,
            norm_cfg=norm_cfg,
            act_cfg=act_cfg)

        self.conv2 = conv(
            hidden_channels,
            hidden_channels,
            3,
            stride=1,
            padding=1,
            conv_cfg=conv_cfg,
            norm_cfg=norm_cfg,
            # act_cfg=act_cfg)
            act_cfg=None)
            # act_cfg=dict(type="None"))
            # act_cfg=dict())
        ''' 
            바로 위에 3*3 conv는 활성화 함수가 없어야 되는데, 적용 가능한지?
        '''

        self.conv3 = ConvModule(
            hidden_channels,
            out_channels,
            1,
            conv_cfg=conv_cfg,
            norm_cfg=norm_cfg,
            act_cfg=act_cfg)

    def forward(self, x):
        out = self.conv1(x)
        out = self.conv2(out)
        return self.conv3(out)


def channel_shuffle(x, groups=2):
    bat_size, channels, w, h = x.shape
    group_c = channels // groups
    x = x.view(bat_size, groups, group_c, w, h)
    x = torch.transpose(x, 1, 2).contiguous()
    x = x.view(bat_size, -1, w, h)
    return x


class TinyCSPLayer_v2(BaseModule):
    """
        "논문을 향하여"
    """

    def __init__(self,
                 in_channels,
                 out_channels,
                 expand_ratio=0.5,
                 num_blocks=1,
                 #add_identity=True,
                 #use_depthwise=False,
                 conv_cfg=None,
                 norm_cfg=dict(type='BN', momentum=0.03, eps=0.001),
                 act_cfg=dict(type='LeakyReLU'),
                 init_cfg=None):
        super().__init__(init_cfg)
        mid_channels = int(out_channels * expand_ratio)

        # act_cfg = dict(type='HardSwish')
    
        self.blocks = nn.Sequential(*[
                DWBottleneck(
                    mid_channels,
                    mid_channels,
                    1.0,
                    add_identity=False,
                    use_depthwise=True,
                    conv_cfg=conv_cfg,
                    norm_cfg=norm_cfg,
                    act_cfg=act_cfg) for _ in range(num_blocks)
            ])
        
    def forward(self, x):
        # channel split half 
        channels = x.shape[1]
        c = channels // 2
        x1 = x[:, :c, :, :]
        x2 = x[:, c:, :, :]

        # concatenate
        out = torch.cat((self.blocks(x2), x1), dim=1)

        # shuffle
        return channel_shuffle(out)

