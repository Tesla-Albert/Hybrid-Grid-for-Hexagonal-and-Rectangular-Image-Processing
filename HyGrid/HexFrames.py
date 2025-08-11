import cv2
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import numpy as np
from torch import Tensor
from torch.nn import init
from typing import Optional
import warnings
warnings.filterwarnings("ignore")

def pad(input:torch.Tensor, padding :int = 0, mode = 'constant', value = 0) ->torch.Tensor:
    """
    函数功能：对输入的图像边缘像素填充，令其增加几行或几列
    :param input: 输入的图像
    :param padding: 在图像周围增加的行数目和列数目
    :return: 返回经过行列填充之后的图像
    """
    padded_input = F.pad(input, (padding, padding, padding, padding), mode, value)
    return padded_input
class HexConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, even_odd_offset, hexkernel_radius, stride=1,
                 padding=0, dilation=1, groups=1, bias=True,
                 padding_mode='constant', padding_value=0):
        super(HexConv2d, self).__init__()
        """
        双倍优化坐标卷积方案
        :param in_channels: 输入通道数
        :param out_channels: 输出通道数
        :param even_odd_offset: 首行奇偶性
        :param hexkernel_radius: 六边形卷积核半径
        :param stride: 卷积核移动步长
        :param padding: 填充
        :param dil: 扩张系数
        :param groups: 分组的组数
        :param bias: 是否偏置
        :param padding_mode: 填充行列数
        """
        self.in_channels = in_channels
        self.out_channels = out_channels

        self.even_odd_offset = even_odd_offset
        self.padded_even_odd_offset = (even_odd_offset + padding)%2
        # self.odd_kernelline_offset = (self.padded_even_odd_offset+hexkernel_radius-1+stride)%2
        # self.out_even_odd_offset = (even_odd_offset - padding + hexkernel_radius-1)%2

        self.hexkernel_radius = hexkernel_radius
        self.hexkernel_size = 2 * hexkernel_radius - 1
        # self.k_w = 2*dilation * (2*hexkernel_radius - 2) + 1
        # self.k_h = (self.hexkernel_size - 1) * dilation + 1
        self.kernelnum = 3 * hexkernel_radius**2 - 3 * hexkernel_radius + 1
        self.stride = stride
        self.sh = stride
        self.sw = stride * 2
        self.out_even_odd_offset = 0


        self.pad = padding
        self.groups = groups
        self.b = bias
        self.dilation = dilation


        self.padding_mode = padding_mode
        self.padding_value = padding_value

        if in_channels % groups != 0:
            raise ValueError('in_channels must be divisible by groups')
        if out_channels % groups != 0:
            raise ValueError('out_channels must be divisible by groups')


        self.kernel = nn.Parameter(torch.empty([out_channels, in_channels//groups, 1, self.kernelnum],dtype=torch.float))
        # self.kernel = nn.Parameter(
        #     torch.full([out_channels, in_channels, 1, self.kernelnum], fill_value = 1 / (self.kernelnum*in_channels), dtype=torch.float))
        if self.b==True:
            self.bias = nn.Parameter(torch.empty([out_channels,]))
        else:
            self.register_parameter('bias', None)

        self.k_w = 2 * self.dilation * (2 * self.hexkernel_radius - 2) + 1
        self.k_h = (self.hexkernel_size - 1) * self.dilation + 1
        self.reset_parameters()




    def reset_parameters(self):
        init.kaiming_uniform_(self.kernel, a=math.sqrt(5))
        if self.bias is not None:
            fan_in, _ = init._calculate_fan_in_and_fan_out(self.kernel)
            if fan_in != 0:
                bound = 1 / math.sqrt(fan_in)
                init.uniform_(self.bias, -bound, bound)
    def forward(self, input: Tensor) -> Tensor:
        """
        卷积
        :param input: 输入为标准的六边形灰度矩阵
                首先对周围一圈进行行列填充
                然后根据输入图像首行奇偶性和填充行数确定填充后图像的首行奇偶性
                接着按照首行奇偶性生成第一类图像，即双倍优化坐标的图像
                在宽度为[第二个像素~倒数第二个像素]的范围内进行卷积
        :return: 标准的六边形灰度矩阵
        """
        d = self.dilation
        input = input.to(self.kernel.dtype)
        weight = torch.zeros(
            [self.out_channels, self.in_channels // self.groups, self.k_h, self.k_w],
            device=input.device, dtype=self.kernel.dtype)

        sum = 0

        for i in range(self.hexkernel_size):
            t = int(np.abs(i - self.hexkernel_radius + 1)) # vertical cell-distance to central kernel
            ln = self.hexkernel_size - t # numnber of cells per line
            weight[:, :, i*d, t*d : t*d + (ln - 1) * 2 * d + 1 : 2*d] += self.kernel[:, :, 0, sum:ln + sum]
            sum += ln
        while input.dim() < 4:
            input = input.unsqueeze(0)
        input = pad(input, self.pad, self.padding_mode, self.padding_value)
        offset = self.padded_even_odd_offset
        # odd_kernelline_offset = self.odd_kernelline_offset
        type1image = heximage_to_type1(input, offset)
        

        if type1image[:,:,:,1:-self.sh].size(2)>=self.k_h \
            and type1image[:,:,:,1:-self.sh].size(3)>=self.k_w:
            evenconv = F.conv2d(input=type1image[:,:,:,1:-self.sh],
                            weight=weight,
                            bias=self.bias,
                            stride=(2*self.sh, self.sw),
                            groups=self.groups,
                            )
        else:
            evenconv = None
        if type1image[:,:,self.sh:,self.sh +1:].size(2) >= self.k_h \
                and type1image[:,:,self.sh:,self.sh +1:].size(3) >= self.k_w:
            oddconv = F.conv2d(input=type1image[:,:,self.sh:,self.sh +1:],
                            weight=weight,
                            bias=self.bias,
                            stride=(2*self.sh, self.sw),
                           groups=self.groups,
                           )
        else:
            oddconv = None
        if evenconv is not None and oddconv is not None:
            pad_width = evenconv.size(3) - oddconv.size(3)
            # print(pad_width)

            #这是为了将奇数行和偶数行的像元数目补成一样的，不是为了交错，所以就应该在一边补
            if pad_width > 0:
                evenconv = evenconv[:, :, :, :-pad_width]
            elif pad_width < 0:
                oddconv = oddconv[:, :, :, :pad_width]

            convedimage = torch.empty([oddconv.shape[0],
                                      oddconv.shape[1],
                                      oddconv.shape[2] + evenconv.shape[2],
                                      evenconv.shape[3]], device = input.device)
            convedimage[:, :, ::2, :] = evenconv
            convedimage[:, :, 1::2, :] = oddconv
        elif evenconv.size!=0:
            convedimage = evenconv
        elif oddconv.size!=0:
            convedimage = oddconv
        else:
            convedimage = torch.empty(0)
        return convedimage
    # def __repr__(self):
    #     return f"HexConv2d({self.in_channels}, {self.out_channels}, kernel_radius={self.hexkernel_radius}, groups={self.groups}, stride={self.sh}, padding={self.pad}, bias={self.b})"
    def extra_repr(self):
        s = ('{in_channels}, {out_channels}, kernel_radius={hexkernel_radius}'
             ', stride={stride}')
        if self.pad != (0,):
            s += ', padding={pad}'
        if self.dilation != (1,) :
            s += ', dilation={dilation}'
        if self.groups != 1:
            s += ', groups={groups}'
        if self.bias is None:
            s += ', bias=False'
        if self.padding_mode != 'zeros':
            s += ', padding_mode={padding_mode}'
        return s.format(**self.__dict__)

class HexConv2dAdaptivePadding(HexConv2d):
    """Implementation of 2D convolution in tensorflow with `padding` as "same",
    which applies padding to input (if needed) so that input image gets fully
    covered by filter and stride you specified. For stride 1, this will ensure
    that output image size is same as input. For stride of 2, output dimensions
    will be half, for example.

    Args:
        in_channels (int): Number of channels in the input image
        out_channels (int): Number of channels produced by the convolution
        kernel_size (int or tuple): Size of the convolving kernel
        stride (int or tuple, optional): Stride of the convolution. Default: 1
        padding (int or tuple, optional): Zero-padding added to both sides of
            the input. Default: 0
        dilation (int or tuple, optional): Spacing between kernel elements.
            Default: 1
        groups (int, optional): Number of blocked connections from input
            channels to output channels. Default: 1
        bias (bool, optional): If ``True``, adds a learnable bias to the
            output. Default: ``True``
    """

    def __init__(self,
                 in_channels: int,
                 out_channels: int,
                 even_odd_offset: int,
                 hexkernel_radius: int,
                 stride: int = 1,
                 padding: int = 0,
                 dilation: int = 1,
                 groups: int = 1,
                 bias: bool = True):
        super().__init__(in_channels,
                         out_channels,
                         even_odd_offset=even_odd_offset,
                         hexkernel_radius=hexkernel_radius,
                         stride=stride,
                         padding=0,
                         dilation=dilation,
                         groups=groups,
                         bias=bias)
    def __repr__(self):
        return f"HexConv2dAdaptivePadding({self.in_channels}, {self.out_channels}, kernel_radius={self.hexkernel_radius}, stride={self.sh}, padding={self.pad}, dilation={self.dilation}, groups={self.groups}, bias={self.b})"


    def forward(self, input: torch.Tensor) -> torch.Tensor:
        input = pad(input, self.pad, self.padding_mode, self.padding_value)
        self.pad = 0
        img_h, img_w = input.size()[-2:]
        hexkernel_radius = self.hexkernel_radius
        kernel_size = hexkernel_radius * 2 - 1
        stride= self.stride
        output_h = math.ceil(img_h / stride)
        output_w = math.ceil(img_w / stride)
        pad_h = (
            max((output_h - 1) * self.stride +
                (kernel_size - 1) * self.dilation + 1 - img_h, 0))
        pad_w = (
            max((output_w) * self.stride +
                (kernel_size - 1) * self.dilation + 1 - img_w, 0))
        if pad_h > 0 or pad_w > 0:
            input = F.pad(input, [
                pad_w // 2, pad_w - pad_w // 2, pad_h // 2, pad_h - pad_h // 2
            ])
        x = super().forward(input)

        return x

class HexPool2d(nn.Module):
    def __init__(self, method, kernel_size=2, stride=None,
                 padding = 0, even_odd_offset = 0,
                 padding_mode = 'constant', padding_value = 0,
                 ceil_mode: bool=False, count_include_pad: bool=True,
                 divisor_override: Optional[int] = None):
        super(HexPool2d, self).__init__()
        self.out_offset = 0
        self.offset = (even_odd_offset + padding) % 2
        self.PoolingMethods = {'max': max_pooling,
                               'min': min_pooling,
                               'average': average_pooling}
        self.method = self.PoolingMethods[method]
        if isinstance(kernel_size, int):
            kernel_size = [kernel_size, kernel_size]
        self.kernel_size = kernel_size
        self.kh, self.kw = kernel_size
        self.stride = stride
        if stride == None:
            self.stride = kernel_size
        if isinstance(stride, int):
            stride = [stride, stride]
        self.stride = stride
        self.sh, self.sw = self.stride
        self.padding = padding
        self.padding_mode = padding_mode
        self.padding_value = padding_value

        self.ceil_mode = ceil_mode
        self.count_include_pad = count_include_pad

    def forward(self, input):
        while input.dim() < 4:
            input = input.unsqueeze(0)
        input = pad(input,self.padding,self.padding_mode, self.padding_value)
        b, c, h, w = input.size()
        self.hn = h//self.sh
        self.wn = (w - self.sw//2 - self.sw)//self.sw + 1

        if self.ceil_mode:
            # pad zeros at right and bottom direction
            ph = (self.kh - h + self.hn * self.sh) % self.kh
            pw = (self.kw - w + (self.wn * self.sw + self.sw // 2)) % self.kw
            padding = (0, ph, 0, pw)
            input = F.pad(input, padding, mode='constant', value=0 if self.count_include_pad else float('nan'))

        b, c, h, w = input.size()
        self.hn = (h - self.kh) // self.sh + 1
        self.wn = (w - self.sw//2)//self.sw

        grid_global_index = torch.stack(
            torch.meshgrid(
                torch.arange(self.hn), torch.arange(self.wn)
            ),
            dim=-1
        ) # hn * wn (i, j)s
        grid_local_index = torch.stack(
            torch.meshgrid(
                torch.arange(self.kh), torch.arange(self.kw)
            ),
            dim=-1
        )

        grid_lefttop_elements_index = torch.stack(
            (self.sh * grid_global_index[:, :, 0],
             grid_global_index[:, :, 0] % 2 * self.sw // 2 + grid_global_index[:, :, 1] * self.sw),
            dim=-1
        )
        grid_elements_index = \
            grid_lefttop_elements_index.view(self.hn, self.wn, 1, 1, 2) + \
            grid_local_index.view(1, 1, self.kh, self.kw, 2)
        grid_elements_index_i = grid_elements_index[..., 0].view(-1)
        grid_elements_index_j = grid_elements_index[..., 1].view(-1)


        grid_elements = input.permute(2, 3, 0, 1)
        grid_elements = grid_elements[grid_elements_index_i, grid_elements_index_j]
        grid_elements = grid_elements.view(self.hn, self.wn, self.kh, self.kw, b, c)
        grid_elements = grid_elements.permute(4, 5, 0, 1, 2, 3)
        grid_elements = grid_elements.view(b, c, self.hn, self.wn, self.kh * self.kw)
        output = self.method(grid_elements)
        return output

    def extra_repr(self) -> str:
        return 'kernel_size={}, stride={}, padding={}'.format(
            self.kernel_size, self.stride, self.padding
        )


class HexAdaptivePool2d(nn.Module):
    def __init__(self,
                 outsize,
                 method,
                 padding = 0,
                 padding_mode = 'constant',
                 padding_value = 0):
        super().__init__()
        if isinstance(outsize, int):
            outsize = [outsize, outsize]
        else:
            raise Exception('outsize = 整数 s 或者列表[h, w]，其它的不行')
        self.hn, self.wn = outsize
        self.PoolingMethods = {'max': max_pooling,
                               'min': min_pooling,
                               'average': average_pooling,
                               'centroid': centroid_pooling}
        self.method = self.PoolingMethods[method]
    def forward(self, input):
        while input.dim() < 4:
            input = input.unsqueeze(0)
        b, c, h, w = input.size()
        input = torch.permute(input, (2, 3, 0, 1))
        grid_h = int(h/self.hn)
        if grid_h>1:
            grid_w = int(w/(self.wn + 0.5))
        else:
            grid_w = int(w/(self.wn))

        grid_global_index = torch.stack(
            torch.meshgrid(
                torch.arange(self.hn), torch.arange(self.wn)
            ),
            dim=-1
        )
        grid_local_index = torch.stack(
            torch.meshgrid(
                torch.arange(grid_h), torch.arange(grid_w)
            ),
            dim=-1
        )
        grid_lefttop_elements_index = torch.stack(
            (grid_h * grid_global_index[:, :, 0],
             grid_global_index[:, :, 0]%2 * grid_w//2 + grid_global_index[:, :, 1] * grid_w),
            dim=-1
        )
        grid_elements_index = \
            grid_lefttop_elements_index.view(self.hn, self.wn, 1, 1, 2) + \
            grid_local_index.view(1, 1, grid_h, grid_w, 2)
        grid_elements_index_i = grid_elements_index[..., 0].view(-1)
        grid_elements_index_j = grid_elements_index[..., 1].view(-1)

        grid_elements = input[grid_elements_index_i, grid_elements_index_j]
        grid_elements = grid_elements.view(self.hn, self.wn, grid_h, grid_w, b, c)
        grid_elements = grid_elements.permute(4, 5, 0, 1, 2, 3)
        grid_elements = grid_elements.view(b, c, self.hn, self.wn, grid_h * grid_w)
        output = self.method(grid_elements)
        return output
class HexGlobalPool2d(nn.Module):
    def __init__(self, method):
        super(HexGlobalPool2d, self).__init__()
        self.PoolingMethods = {'max': max_pooling,
                               'min': min_pooling,
                               'average': average_pooling,
                               'centroid': centroid_pooling}
        self.method = self.PoolingMethods[method]
    def forward(self, input):
        while input.dim() < 4:
            input = input.unsqueeze(0)
        unfolded = input.reshape(input.size(0), input.size(1), input.size(2)*input.size(3))
        return self.method(unfolded)

#--------------------------format conversion--------------------------#
def heximage_to_type1(input:torch.Tensor, even_odd_offset)->torch.Tensor:
    """
    将存储六边形像元灰度的矩阵转换为双倍优化存储矩阵
    :param input: 六边形像元的灰度矩阵[batch * channel * height * width]
    :param even_odd_offset: pad过后首行的奇偶性
    :return: 双倍优化存储图像矩阵
    """
    while input.dim() < 4:
        input = input.unsqueeze(0)
    tmp = input.repeat_interleave(2,dim = 3)
    oddlines = tmp[:, :, 1::2, :]
    evenlines = tmp[:, :, ::2, :]
    hex_oddlines = torch.cat(
        (torch.zeros([oddlines.shape[0], oddlines.shape[1], oddlines.shape[2], (1 + even_odd_offset) % 2],device= oddlines.device),
                oddlines,
                torch.zeros([oddlines.shape[0], oddlines.shape[1], oddlines.shape[2], 1 - (1 + even_odd_offset) % 2],device= oddlines.device)),
        3)
    hex_evenlines = torch.cat(
        (torch.zeros([evenlines.shape[0], evenlines.shape[1], evenlines.shape[2], (0 + even_odd_offset) % 2],device= oddlines.device),
                evenlines,
                torch.zeros([evenlines.shape[0], evenlines.shape[1], evenlines.shape[2], 1 - (0 + even_odd_offset) % 2],device= oddlines.device)),
        3)
    type1image = torch.empty([tmp.shape[0],
                              tmp.shape[1],
                              hex_evenlines.shape[2] + hex_oddlines.shape[2],
                              hex_evenlines.shape[3]],device= oddlines.device)
    type1image[:, :, ::2, :] = hex_evenlines
    type1image[:, :, 1::2, :] = hex_oddlines
    return type1image
def heximage_to_type2(input:torch.Tensor, even_odd_offset)->torch.Tensor:
    type1image = heximage_to_type1(input, even_odd_offset)
    type2image = type1image.repeat_interleave(2, dim=2)
    return type2image
def type1_to_heximage(input:torch.Tensor, even_odd_offset:int)->(torch.Tensor, int):
    """
    将双倍优化存储的图像转换为六边形带首行offset信息的六边形图像
    :param input: 双倍优化的灰度矩阵[batch * channel * height * (width * 2 + 1)]
    :param even_odd_offset: 首行缩进的性质
    :return: 六边形图像和offset
    """
    out = input[:,:,:,1::2]
    return out, even_odd_offset

#---------------------cell statistical properties---------------------#
def max_pooling(input):
    inputmax, _ =  torch.max(input.masked_fill(input.isnan(), float('-inf')), dim=-1)
    return inputmax
def min_pooling(input):
    inputmin, _ =  torch.min(input.masked_fill(input.isnan(), float('inf')), dim=-1)
    return  inputmin
def average_pooling(input):
    # 计算有效的数量
    count_non_nan = torch.where(input.isnan(), torch.zeros_like(input), torch.ones_like(input)).sum(dim=-1)

    # 计算有效值的总和
    sum_non_nan = torch.where(input.isnan(), torch.zeros_like(input), input).sum(dim=-1)

    # 计算均值
    mean_value = sum_non_nan / count_non_nan

    # 处理除零的情况
    mean_value[count_non_nan == 0] = float('nan')
    return mean_value




















