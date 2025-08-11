import matplotlib.pyplot as plt
import sys

try:
    from osgeo import gdal
    has_gdal = True
except ImportError:
    has_gdal = False
    print("Failed to import gdal.")
    sys.exit()

try:
    import mmcv
    has_mmcv = True
except ImportError:
    has_mmcv = False
    pass
try:
    import cv2
    has_cv2 = True
except ImportError:
    has_cv2 = False
    pass

if not has_gdal or not has_mmcv or not has_cv2:
    print("Failed to import gdal, mmcv, or cv2. Please install one of these packages.")
    sys.exit()


import torch
from .geometry_np import rect_to_hex_resample

import math
import os
import numpy as np
import warnings
warnings.filterwarnings("ignore")

class IMAGE:
    def __init__(self, pathname=None, data = None, geotrans = None, proj = None, backend='gdal'):
        if pathname is None and data is None:
            raise ValueError("pathname and data can not be None at the same time")
        if pathname is not None and data is not None:
            raise ValueError("pathname and data can not be Given at the same time")
        if pathname is not None:
            self.path = pathname
            if not os.path.exists(self.path):
                raise OSError("path dosen't exist.")
            file_name, file_extension = os.path.splitext(pathname)
            if file_extension in (".tif", ".TIF", ".tiff", ".TIFF",".jpg",".png",".jpeg", ".JPEG"):
                self.filetype = 1
                self.data = gdal.Open(self.path)
                self.height = self.data.RasterYSize
                self.width = self.data.RasterXSize
                self.bands = self.data.RasterCount
                self.geotrans = self.data.GetGeoTransform()
                self.proj = self.data.GetProjection()
            self.Image = self.LoadImageArray()
            if self.Image.ndim == 2:
                self.Image = np.broadcast_to(self.Image, (1, self.height, self.width))
        elif data is not None:
            if data.ndim == 2:
                data = np.broadcast_to(data, (1, data.shape[0], data.shape[1]))
            self.Image = data
            self.bands, self.height, self.width = data.shape
            self.geotrans = geotrans
            if self.geotrans == None:
                self.geotrans = (0, 1, 0, 0, 0, 1)
            self.proj = proj
            self.path = 'tmp.tif'
        self.shape = (self.bands, self.height, self.width)
        self.backend = backend

    def size(self, index):
        return self.data.shape[index]





    def Tiles(self):
        """
        对图像进行切片，大小为2000*2000
        控制数组中存储的图像数据大小，防止图像过大数组无法加载
        图像流式处理
        但是暂时不做实现，实现基础功能之后再说
        """
        pass
    def LoadImageArray(self,
                  w_range_start = 0,
                  h_range_start = 0,
                  w_range = None,
                  h_range = None):
        if w_range is None:
            w_range = self.width
        if h_range is None:
            h_range = self.height
        tmp_image = self.data.ReadAsArray(w_range_start,
                                          h_range_start,
                                          w_range,
                                          h_range)
        self.width = w_range - w_range_start
        self.height = h_range - h_range_start
        if self.bands == 1:
            tmp_image = np.expand_dims(tmp_image, axis=0)

        return tmp_image



    def ConvertToHexagon(self, interpolation='nearest'):
        return rect_to_hex_resample(
            self.Image,
            [self.height//2, self.width//2],
            interpolation=interpolation
        )
    def SaveImage(self, pathname):
        if 'int8' in self.Image.dtype.name:
            self.datatype = gdal.GDT_Byte
        elif 'int16' in self.Image.dtype.name:
            self.datatype = gdal.GDT_UInt16
        else:
            self.datatype = gdal.GDT_Byte

        file_name, file_extension = os.path.splitext(pathname)

        if file_extension in (".tif", ".TIF", ".tiff", ".TIFF", ".JPEG", ".jpg", ".jpeg", ".PNG", ".png"):
            self.filetype = 1

        if self.backend == 'gdal':
            drivername = None
            if drivername is not None:
                driver = gdal.GetDriverByName('GTiff')
            else:
                raise Exception(
                    "class IMAGE in HyGrid/Image.py: format of output is incorrect, the gdal drivername = None")

            self.dataset = driver.Create(pathname, self.Image.shape[2], self.Image.shape[1], self.Image.shape[0],
                                             self.datatype)  # (按宽×高来索引)
            self.dataset.SetGeoTransform(self.geotrans)
            if self.proj != None:
                self.dataset.SetProjection(self.proj)
            for i in range(self.bands):
                channel_i = self.dataset.GetRasterBand(i + 1)
                channel_i.WriteArray(self.Image[i])

            self.dataset.FlushCache()
        elif self.backend == 'mmcv':
            mmcv.imwrite(self.Image[::-1, ...].transpose(1, 2, 0), file_path=pathname)
        elif self.backend == 'cv2':
            cv2.imwrite(pathname, self.Image[::-1, ...].transpose(1, 2, 0))
    def imshow(self):
        image = self.Image.astype(np.uint8)
        if self.bands == 1:
            plt.imshow(image.squeeze(), cmap='gray')

        else:
            plt.imshow(image.transpose(1, 2, 0)[..., :3])
        plt.show()


# @cuda.jit()#显卡并行加速的核函数
# def Calc_Hexagonalization_on_gpu(Img, hexImg):#Img[c*h*w]
#     tx = cuda.blockIdx.x * cuda.blockDim.x + cuda.threadIdx.x
#     ty = cuda.blockIdx.y * cuda.blockDim.y + cuda.threadIdx.y
#     dw = int(ty & 1)
#     if (2 * ty) < Img.shape[-2] - 1 and (2 * tx) < (Img.shape[-1]  - 2):
#         for i in range(Img.shape[0]):
#             p = int(Img[i][2 * ty][2 * tx + dw] / 4 + \
#                 Img[i][2 * ty + 1][2 * tx + dw] / 4 + \
#                 Img[i][2 * ty][2 * tx + 1 + dw] / 4 + \
#                 Img[i][2 * ty + 1][2 * tx + 1 + dw] / 4)
#             hexImg[i][ty][tx] = p




if __name__ == '__main__':
    import pickle
    import cv2
    path = r"D:\HexagonalConvolution\Dataset\GID\original\test\labels\GF2_PMS1__L1A0001064454-MSS1_24label.png"
    imagedataset = IMAGE(path)
    image = imagedataset.LoadImageArray()
    heximg = imagedataset.ConvertToHexagon(image)
    #显示heximg图像
    cv2.namedWindow('heximg', cv2.WINDOW_NORMAL)
    cv2.resizeWindow('heximg', 800, 800)
    cv2.imshow("image",
               cv2.cvtColor(image.astype(np.uint8).transpose(1,2,0), cv2.COLOR_BGR2RGB))
    cv2.imshow("heximg",
               cv2.cvtColor(heximg.astype(np.uint8).transpose(1,2,0), cv2.COLOR_BGR2RGB))
    cv2.waitKey(0)

    print(image.shape)
    print(heximg.shape)

    # data = gdal.Open(path)
    # geotrans = data.GetGeoTransform()
    # proj = data.GetProjection()
    # tmp_image = data.ReadAsArray(w_range_start,
    #                              h_range_start,
    #                              w_range,
    #                              h_range)
    # SaveImage(tmp_image, save_path, geotrans, proj)
    #
    # def SaveImage(
    #         Image,
    #         proj,
    #         geotrans,
    #         save_path
    # ):
    #
    #     if 'int8' in Image.dtype.name:
    #         datatype = gdal.GDT_Byte
    #     elif 'int16' in Image.dtype.name:
    #         datatype = gdal.GDT_UInt16
    #     else:
    #         datatype = gdal.GDT_Byte
    #     driver = gdal.GetDriverByName("GTiff")
    #     dataset = driver.Create(save_path, Image.shape[2], Image.shape[1], Image.shape[0],
    #                             datatype)  # (按宽×高来索引)
    #     dataset.SetGeoTransform(geotrans)
    #     if proj != None:
    #         dataset.SetProjection(proj)
    #     for i in range(Image.shape[0]):
    #         channel_i = dataset.GetRasterBand(i + 1)
    #         channel_i.WriteArray(Image[i])
    #
    #     dataset.FlushCache()




