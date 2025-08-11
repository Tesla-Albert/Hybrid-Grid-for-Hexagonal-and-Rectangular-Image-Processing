import warnings
from . import HexFrames as hnn
import torch.nn as nn
import torch.nn.functional as F
import torch
from typing import Union, Dict, Optional, Tuple
from mmcv.cnn.bricks.norm import build_norm_layer
from mmcv.cnn.bricks.padding import build_padding_layer
from mmcv.cnn.bricks.activation import build_activation_layer
from mmcv.cnn.bricks.registry import CONV_LAYERS, PADDING_LAYERS
from mmcv.utils import _BatchNorm, _InstanceNorm
from mmcv.cnn.utils import constant_init, kaiming_init

# TORCH_VERSION = torch.__version__

CONV_LAYERS.register_module('HexConv2d', module=hnn.HexConv2d)
# PADDING_LAYERS.register_module('zero', module=nn.ZeroPad2d)
# PADDING_LAYERS.register_module('reflect', module=nn.ReflectionPad2d)
# PADDING_LAYERS.register_module('replicate', module=nn.ReplicationPad2d)


def build_hexconv_layer(cfg: Optional[Dict], *args, **kwargs) -> nn.Module:
    """Build convolution layer.

    Args:
        cfg (None or dict): The conv layer config, which should contain:
            - type (str): Layer type.
            - layer args: Args needed to instantiate an conv layer.
        args (argument list): Arguments passed to the `__init__`
            method of the corresponding conv layer.
        kwargs (keyword arguments): Keyword arguments passed to the `__init__`
            method of the corresponding conv layer.

    Returns:
        nn.Module: Created conv layer.
    """
    if cfg is None:
        cfg_ = dict(type='HexConv2d')
    else:
        if not isinstance(cfg, dict):
            raise TypeError('cfg must be a dict')
        if 'type' not in cfg:
            raise KeyError('the cfg dict must contain the key "type"')
        cfg_ = cfg.copy()

    layer_type = cfg_.pop('type')
    if layer_type not in CONV_LAYERS:
        raise KeyError(f'Unrecognized layer type {layer_type}')
    else:
        conv_layer = CONV_LAYERS.get(layer_type)

    layer = conv_layer(*args, **kwargs, **cfg_)

    return layer

def build_hexpadding_layer(cfg: Dict, *args, **kwargs) -> nn.Module:
    """Build padding layer.

    Args:
        cfg (dict): The padding layer config, which should contain:
            - type (str): Layer type.
            - layer args: Args needed to instantiate a padding layer.

    Returns:
        nn.Module: Created padding layer.
    """
    return build_padding_layer(cfg, *args, **kwargs)

def build_hexnorm_layer(cfg: Dict,
                        num_features: int,
                        postfix: Union[int, str] = '') -> Tuple[str, nn.Module]:
    """Build normalization layer.
    When heximages are stored as offset-mode, the normalization
    Args:
        cfg (dict): The norm layer config, which should contain:

            - type (str): Layer type.
            - layer args: Args needed to instantiate a norm layer.
            - requires_grad (bool, optional): Whether stop gradient updates.
        num_features (int): Number of input channels.
        postfix (int | str): The postfix to be appended into norm abbreviation
            to create named layer.

    Returns:
        tuple[str, nn.Module]: The first element is the layer name consisting
        of abbreviation and postfix, e.g., bn1, gn. The second element is the
        created norm layer.
    """
    return build_norm_layer(cfg, num_features, postfix)
def build_hexactivation_layer(cfg: Dict) -> nn.Module:
    return build_activation_layer(cfg)




# 实验组
class HexConvModule(nn.Module):
    """A hexconv block that bundles hexconv/norm/activation layers.

        This block simplifies the usage of convolution layers, which are commonly
        used with a norm layer (e.g., BatchNorm) and activation layer (e.g., ReLU).
        It is based upon three build methods: `build_conv_layer()`,
        `build_norm_layer()` and `build_activation_layer()`.

        Besides, we add some additional features in this module.
        1. Automatically set `bias` of the conv layer.
        2. Spectral norm is supported.


        Args:
            in_channels (int): Number of channels in the input feature map.
                Same as that in ``nn._ConvNd``.
            out_channels (int): Number of channels produced by the convolution.
                Same as that in ``nn._ConvNd``.
            hexkernel_radius (int): Radius of the hexagonal convolving kernel.
                Same as that in ``hnn.HexConv2d``.
            stride (int): Stride of the convolution.
                Same as that in ``hnn.HexConv2d``.
            padding (int): Zero-padding added to both sides of
                the input. Same as that in ``hnn.HexConv2d``.
            dilation (int): Spacing between kernel elements.
                Same as that in ``hnn.HexConv2d``, but not realized yet.
            groups (int): Number of blocked connections from input channels to
                output channels. Same as that in ``hnn.HexConv2d``.
            bias (bool | str): If specified as `auto`, it will be decided by the
                norm_cfg. Bias will be set as True if `norm_cfg` is None, otherwise
                False. Default: "auto".
            conv_cfg (dict): Config dict for convolution layer. Default: None,
                which means using hnn.HexConv2d.
            norm_cfg (dict): Config dict for normalization layer. Default: None.
            act_cfg (dict): Config dict for activation layer.
                Default: dict(type='ReLU').
            inplace (bool): Whether to use inplace mode for activation.
                Default: True.
            with_spectral_norm (bool): Whether use spectral norm in conv module.
                Default: False.
            padding_mode (str): Default: 'zeros'.
            order (tuple[str]): The order of conv/norm/activation layers. It is a
                sequence of "conv", "norm" and "act". Common examples are
                ("conv", "norm", "act") and ("act", "conv", "norm").
                Default: ('conv', 'norm', 'act').
        """

    _abbr_ = 'conv_block'

    def __init__(self,
                 in_channels: int,
                 out_channels: int,
                 even_odd_offset: int,
                 hexkernel_radius: int,
                 stride: int = 1,
                 padding: int = 0,
                 dilation: int = 1,
                 groups: int = 1,
                 bias: Union[bool, str] = 'auto',
                 conv_cfg: Optional[Dict] = None,
                 norm_cfg: Optional[Dict] = None,
                 act_cfg: Optional[Dict] = dict(type='ReLU'),
                 inplace: bool = True,
                 with_spectral_norm: bool = False,
                 padding_mode: str = 'zeros',
                 order: tuple = ('conv', 'norm', 'act')):
        super().__init__()
        assert conv_cfg is None or isinstance(conv_cfg, dict)
        assert norm_cfg is None or isinstance(norm_cfg, dict)
        assert act_cfg is None or isinstance(act_cfg, dict)
        official_padding_mode = ['zeros', 'circular']
        self.conv_cfg = conv_cfg
        self.norm_cfg = norm_cfg
        self.act_cfg = act_cfg
        self.inplace = inplace
        self.with_spectral_norm = with_spectral_norm
        self.with_explicit_padding = padding_mode not in official_padding_mode
        self.order = order
        assert isinstance(self.order, tuple) and len(self.order) == 3
        assert set(order) == {'conv', 'norm', 'act'}

        self.with_norm = norm_cfg is not None
        self.with_activation = act_cfg is not None
        # if the conv layer is before a norm layer, bias is unnecessary.
        if bias == 'auto':
            bias = not self.with_norm
        self.with_bias = bias

        if self.with_explicit_padding:
            pad_cfg = dict(type=padding_mode)
            self.padding_layer = build_hexpadding_layer(pad_cfg, padding)

        # reset padding to 0 for conv module
        conv_padding = 0 if self.with_explicit_padding else padding
        # build convolution layer
        self.conv = build_hexconv_layer(
            conv_cfg,
            in_channels,
            out_channels,
            even_odd_offset,
            hexkernel_radius,
            stride=stride,
            padding=conv_padding,
            dilation=dilation,
            groups=groups,
            bias=bias)
        # export the attributes of self.conv to a higher level for convenience
        self.in_channels = self.conv.in_channels
        self.out_channels = self.conv.out_channels
        self.hexkernel_radius = self.conv.hexkernel_radius
        self.stride = self.conv.stride
        self.padding = padding
        self.dilation = self.conv.dilation
        # self.transposed = self.conv.transposed
        # self.output_padding = self.conv.output_padding
        self.groups = self.conv.groups

        if self.with_spectral_norm:
            self.conv = nn.utils.spectral_norm(self.conv)

        # build normalization layers
        if self.with_norm:
            # norm layer is after conv layer
            if order.index('norm') > order.index('conv'):
                norm_channels = out_channels
            else:
                norm_channels = in_channels
            self.norm_name, norm = build_hexnorm_layer(
                norm_cfg, norm_channels)  # type: ignore
            self.add_module(self.norm_name, norm)
            if self.with_bias:
                if isinstance(norm, (_BatchNorm, _InstanceNorm)):
                    warnings.warn(
                        'Unnecessary conv bias before batch/instance norm')
        else:
            self.norm_name = None  # type: ignore

        # build activation layer
        if self.with_activation:
            act_cfg_ = act_cfg.copy()  # type: ignore
            # nn.Tanh has no 'inplace' argument
            if act_cfg_['type'] not in [
                'Tanh', 'PReLU', 'Sigmoid', 'HSigmoid', 'Swish', 'GELU'
            ]:
                act_cfg_.setdefault('inplace', inplace)
            self.activate = build_hexactivation_layer(act_cfg_)

        # Use msra init by default
        self.init_weights()

    @property
    def norm(self):
        if self.norm_name:
            return getattr(self, self.norm_name)
        else:
            return None

    def init_weights(self):
        # 1. It is mainly for customized conv layers with their own
        #    initialization manners by calling their own ``init_weights()``,
        #    and we do not want ConvModule to override the initialization.
        # 2. For customized conv layers without their own initialization
        #    manners (that is, they don't have their own ``init_weights()``)
        #    and PyTorch's conv layers, they will be initialized by
        #    this method with default ``kaiming_init``.
        # Note: For PyTorch's conv layers, they will be overwritten by our
        #    initialization implementation using default ``kaiming_init``.
        if not hasattr(self.conv, 'init_weights'):
            if self.with_activation and self.act_cfg['type'] == 'LeakyReLU':
                nonlinearity = 'leaky_relu'
                a = self.act_cfg.get('negative_slope', 0.01)
            else:
                nonlinearity = 'relu'
                a = 0
            kaiming_init(self.conv, a=a, nonlinearity=nonlinearity)
        if self.with_norm:
            constant_init(self.norm, 1, bias=0)

    def forward(self,
                x: torch.Tensor,
                activate: bool = True,
                norm: bool = True) -> torch.Tensor:
        for layer in self.order:
            if layer == 'conv':
                if self.with_explicit_padding:
                    x = self.padding_layer(x)
                x = self.conv(x)
            elif layer == 'norm' and norm and self.with_norm:
                x = self.norm(x)
            elif layer == 'act' and activate and self.with_activation:
                x = self.activate(x)
        return x

