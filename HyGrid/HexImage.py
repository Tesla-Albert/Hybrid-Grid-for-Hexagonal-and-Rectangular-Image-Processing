import sys
import inspect
import pickle
from .Image import IMAGE
from OpenGL.GL import *
from OpenGL.arrays.vbo import VBO	# 引入VBO类
import numpy as np
from .HexPixelArt.window import Window
from .HexPixelArt.texture import Texture
from .HexPixelArt.hexagon_mosaic_shader import Hexagon_Mosaic_shader
import os
import tkinter as tk
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

import warnings

from warnings import simplefilter
simplefilter(action='ignore', category=FutureWarning)

if not has_gdal or not has_mmcv or not has_cv2:
    print("Failed to import gdal, mmcv, or cv2. Please install one of these packages.")
    sys.exit()


class HEXIMAGE(IMAGE):
    def __init__(self, pathname = None,heximagetype = None, data = None, geotrans = None, proj = None, even_odd_offset = False, backend='gdal'):
        """
        :heximagetype = None，表示读取的普通影像
        :heximagetype = 1，表示读取的六边形图像，使用双倍优化坐标存储
        :heximagetype = 2，表示读取的六边形图像，使用经过可视化改善的双倍优化坐标存储
        """
        # super().__init__(pathname, data, geotrans, proj, backend)
        if pathname is None and data is None:
            raise ValueError("pathname and data can not be None at the same time")
        if pathname is not None and data is not None:
            raise ValueError("pathname and data can not be Given at the same time")

        if pathname is not None:
            super().__init__(pathname, backend=backend)
            self.heximagetype = heximagetype
            file_name, file_extension = os.path.splitext(pathname)
            if file_extension in (".tif", ".TIF", ".tiff", ".TIFF",".jpg",".png",".jpeg", ".JPEG"):
                if heximagetype == None:
                    self.HexagonImage = self.ConvertToHexagon()
                    self.bands, self.height, self.width = self.HexagonImage.shape[0:3]

                elif heximagetype == 1:
                    tmp = self.LoadImageArray()
                    self.height = self.height
                    self.width = (self.width-1)//2
                    self.HexagonImage = np.zeros([self.bands, self.height, self.width])
                    self.HexagonImage[:, :, :] = tmp[:, :, 1::2]

                elif heximagetype == 2:
                    tmp = self.LoadImageArray()
                    if (self.width & 1) == 0:
                        zeros = np.zeros((self.bands, self.height, 1))
                        tmp = np.append(tmp, zeros, axis=2)
                        self.width += 1
                    self.height = self.height // 2
                    self.width = (self.width-1)//2
                    self.HexagonImage = np.zeros([self.bands,  self.height, self.width])
                    if tmp.ndim==3:
                        self.HexagonImage[:, :, :] = tmp[:, ::2, 1::2]
                    elif tmp.ndim==2:
                        self.HexagonImage[:, :, :] = tmp[::2, 1::2]


                else:
                    raise Exception("不支持的文件类型\n要么输入的是普通图像文件：None\n要么输入的是六边形图像通用格式：1\n要么输入后缀为‘.heximg'的六边形图像专用文件格式：2")
            elif file_extension == ".heximg":#将六边形图像矩阵按键值对存储
                self.datapath = pathname
                with open(pathname, "rb") as f:
                    self.Heximagedataset = pickle.load(f)
                self.filetype = 2
                self.height = self.Heximagedataset['height']
                self.width = self.Heximagedataset['width']
                self.bands = self.Heximagedataset['bands']
                self.geotrans = self.Heximagedataset['geotransform']
                self.proj = self.Heximagedataset['projection']
                self.even_odd_offset = self.Heximagedataset['offset']
                self.HexagonImage = self.Heximagedataset['HexMatrix']
                if self.HexagonImage.ndim < 3:
                    self.HexagonImage = np.broadcast_to(self.HexagonImage, (3, self.height, self.width))
        elif data is not None:
            if data.ndim == 2:
                data = np.broadcast_to(data, (1, data.shape[0], data.shape[1]))
            if heximagetype == None:
                self.HexagonImage = data
            elif heximagetype == 1:
                self.HexagonImage = data[:, :, 1:-1:2]
            elif heximagetype == 2:
                self.HexagonImage = data[:, ::2, 1:-1:2]
            self.bands = self.HexagonImage.shape[0]
            self.height = self.HexagonImage.shape[1]
            self.width = self.HexagonImage.shape[2]
            self.geotrans = geotrans
            if self.geotrans==None:
                self.geotrans = (0, 1, 0, 0, 0, 1)
            self.proj = proj
            self.path = inspect.signature(self.__init__).parameters['data'].name
            self.backend = backend



        self.even_odd_offset = int(even_odd_offset)
        self.shape = (self.bands, self.height, self.width)
    def size(self, index):
        return self.HexagonImage.shape[index]

    def build_Heximagedataset(self):
        self.Heximagedataset = {}
        self.Heximagedataset['height'] = self.height
        self.Heximagedataset['width'] = self.width
        self.Heximagedataset['bands'] = self.bands
        self.Heximagedataset['geotransform'] = self.geotrans
        self.Heximagedataset['projection'] = self.proj
        self.Heximagedataset['offset'] = self.even_odd_offset
        self.Heximagedataset['HexMatrix'] = self.HexagonImage

    def GenerateType1Image(self):
        bands = self.HexagonImage.shape[0]
        height = self.HexagonImage.shape[1]
        width = self.HexagonImage.shape[2]
        Heximg_type1 = np.zeros([bands, height, width * 2 + 1])
        tmp = np.repeat(self.HexagonImage, 2, axis=2)
        for c in range(bands):
            for i in range(height):
                if (i + self.even_odd_offset) % 2:
                    Heximg_type1[c,i] = np.insert(tmp[c][i], 0, 0)
                else:
                    Heximg_type1[c,i] = np.append(tmp[c][i], 0)
        geotrans_type1 = (self.geotrans[0], self.geotrans[1], self.geotrans[2],
                                   self.geotrans[3], self.geotrans[4], self.geotrans[5]*2,)
        return Heximg_type1, geotrans_type1
    def GenerateType2Image(self):
        bands = self.shape[0]
        height = self.shape[1]
        width = self.shape[2]
        Heximg_type2 = np.zeros([bands, height * 2, width * 2 + 1])
        tmp = np.repeat(np.repeat(self.HexagonImage, 2, axis=-2), 2, axis=-1)
        for c in range(bands):
            for i in range(height):
                if (i + self.even_odd_offset) % 2 :
                    Heximg_type2[c][2 * i] = np.insert(tmp[c][2 * i], 0, 0)
                    Heximg_type2[c][2 * i + 1] = Heximg_type2[c][2 * i]
                else:
                    Heximg_type2[c][2 * i] = np.append(tmp[c][2 * i], 0)
                    Heximg_type2[c][2 * i + 1] = Heximg_type2[c][2 * i]
        geotrans_type2 = (self.geotrans[0], self.geotrans[1], self.geotrans[2],
                          self.geotrans[3], self.geotrans[4], self.geotrans[5],)
        return Heximg_type2, geotrans_type2
    def SaveHexImage(self,pathname,imagetype = 1,filetype = 1):
        file_name, file_extension = os.path.splitext(pathname)
        if file_extension == ".heximg":
            filetype = 2
        if file_extension in (".tif", ".TIF", ".tiff", ".TIFF",".png", "bmp"):
            self.filetype = 1
        if file_extension in ("JPG",".jpg", "JPEG", "jpeg"):
            warnings.warn("jpg and jpeg are lossy compression formats, switching to png")
            file_extension = ".png"
        pathname = file_name + file_extension

        if filetype == 1:
            if imagetype == 1:
                tmp, geotrans_out = self.GenerateType1Image()
            else:
                tmp, geotrans_out = self.GenerateType2Image()

            if 'int8' in self.HexagonImage.dtype.name:
                self.datatype = gdal.GDT_Byte
                tmp = tmp.astype(np.uint8)
            elif 'int16' in self.HexagonImage.dtype.name:
                self.datatype = gdal.GDT_UInt16
                tmp = tmp.astype(np.uint16)
            else:
                self.datatype = gdal.GDT_Byte
                tmp = tmp.astype(np.uint8)

            if self.backend == 'gdal':
                driver = gdal.GetDriverByName("GTiff")
                self.Hex_dataset = driver.Create(pathname, tmp.shape[2], tmp.shape[1], tmp.shape[0],
                                                 self.datatype, options=["TILED=YES", "COMPRESS=LZW"])  # (按宽×高来索引)
                self.Hex_dataset.SetGeoTransform(geotrans_out)
                if self.proj != None:
                    self.Hex_dataset.SetProjection(self.proj)
                for i in range(tmp.shape[0]):
                    channel_i = self.Hex_dataset.GetRasterBand(i + 1)
                    channel_i.WriteArray(tmp[i])
                self.Hex_dataset.FlushCache()
            elif self.backend == 'mmcv':
                mmcv.imwrite(tmp[::-1, ...].transpose(1, 2, 0), pathname)
            elif self.backend == 'cv2':
                cv2.imwrite(pathname, tmp[::-1, ...].transpose(1, 2, 0))


        else:
            with open(pathname, "wb") as f:
                self.build_Heximagedataset()
                pickle.dump(self.Heximagedataset, f)
    def Hex_imshow(self):


        """triangle = np.array([
            -1 + (-1) * self.geotrans[2]/self.height, (-1) * self.geotrans[4]/self.width + (-1), 0, 0, 0.0,
             1 + (-1) * self.geotrans[2]/self.height,    1 * self.geotrans[4]/self.width + (-1), 0, 1, 0.0,
             1 +    1 * self.geotrans[2]/self.height,    1 * self.geotrans[4]/self.width +    1, 0, 1, 1.0,
            -1 +    1 * self.geotrans[2]/self.height, (-1) * self.geotrans[4]/self.width +    1, 0, 0, 1.0
        ], dtype=np.float32)"""

        theta = 0.0#rad
        triangle = np.array([
            -1 * np.cos(theta) + (-1) * (-np.sin(theta)), (-1) * np.sin(theta) + (-1) * (np.cos(theta)), 0, 0, 0.0,
            1 * np.cos(theta) + (-1) * (-np.sin(theta)), 1 * np.sin(theta) + (-1) * (np.cos(theta)), 0, 1, 0.0,
            1 * np.cos(theta) + 1 * (-np.sin(theta)), 1 * np.sin(theta) + 1 * (np.cos(theta)), 0, 1, 1.0,
            -1 * np.cos(theta) + 1 * (-np.sin(theta)), (-1) * np.sin(theta) + 1 * (np.cos(theta)), 0, 0, 1.0
        ], dtype=np.float32)
        w = Window(1500, 1500, self.path)

        tex = Texture(imgarr= self.HexagonImage, even_odd_offset= self.even_odd_offset)
        height, width = tex.TexSize()
        ratio = float(width) / float(height)
        root = tk.Tk()
        win_width = root.winfo_screenwidth()
        win_height = root.winfo_screenheight()
        root.destroy()
        def render():
            height, width = tex.TexSize()
            displayratio = min(win_height / height, win_width / width)
            displayheight = int(height * displayratio)
            displaywidth = int(width * displayratio)
            w.WindowResize(displaywidth, displayheight)

            vao = glGenVertexArrays(1)  # 创建VAO
            glBindVertexArray(vao)  # 绑定VAO

            vbo = VBO(triangle, usage=GL_STATIC_DRAW, target=GL_ARRAY_BUFFER)  # 创建VBO
            vbo.bind()  # 绑定VBO

            shader = Hexagon_Mosaic_shader()
            shader.setAttrib("aPos", 3, GL_FLOAT, 20, 0)
            shader.setAttrib("aTex", 2, GL_FLOAT, 20, 12)
            tex.setIndex(shader)
            shader.use()
            glBindVertexArray(vao)
            glDrawArrays(GL_QUADS, 0, 4)  # 绘制矩形
            for i in range(4):
                triangle[5 * i + 0] -= w.dx
                triangle[5 * i + 1] -= w.dy
                triangle[5 * i + 0] *= w.scale
                triangle[5 * i + 1] *= w.scale
            w.dx = 0
            w.dy = 0
            w.scale = 1
            tex.hierarchy += w.delta_hierarchy
            tex.img_serial_number += w.delta_img_serialNum

        w.loop(render)




