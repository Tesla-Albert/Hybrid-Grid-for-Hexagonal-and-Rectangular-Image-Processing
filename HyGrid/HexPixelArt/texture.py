# -*- coding:utf-8 -*-

from OpenGL.GL import *
from PIL import Image
import numpy as np


class Texture:
    def __init__(self, imgPath = None, imgarr = None, idx=0, texType=GL_TEXTURE_2D,
                 imgType=GL_RGB, innerType=None, even_odd_offset = 0, dataType=GL_UNSIGNED_BYTE):
        imagearray = imgarr
        if not innerType:
            innerType = imgType
        self.even_odd_offset = even_odd_offset
        self.idx = idx
        #创建纹理对象
        self.tex = glGenTextures(1)
        glBindTexture(GL_TEXTURE_2D, self.tex)
        self.img_serial_number = 1
        #读取纹理图片
        img = []
        if imgPath is not None:
            img = Image.open(imgPath)
            img = np.array(img, np.uint8)#h,w,c
        elif imagearray is not None:
            if imagearray.shape[0]==1:
                imagearray = np.repeat(imagearray, repeats=3, axis=0)
            img = imagearray.transpose(1, 2, 0)

        self.texHeight, self.texWidth, _=img.shape 
        while (img.shape[0])%4!=0:
            row_to_be_added = np.zeros((1,img.shape[1],img.shape[2]))
            img = np.append(img, row_to_be_added, axis = 0)
            self.texHeight += 1

        while (img.shape[1])%4!=0:
            colum_to_be_added = np.zeros((img.shape[0],1,img.shape[2]))
            img = np.append(img, colum_to_be_added, axis = 1)
            self.texWidth += 1
        if img.shape[2]==4:
           img = np.delete(img,3,axis = 2)

        #将纹理传给GPU
        glTexImage2D(texType, 0, GL_RGB, self.texWidth, self.texHeight, 0, GL_RGB, GL_UNSIGNED_BYTE, img)
        glGenerateMipmap(texType)

        #纹理设置
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_BORDER)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_BORDER)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR_MIPMAP_LINEAR)
        #纹理显示层次
        self.hierarchy = 0

    def setIndex(self, shader, name = None):
        if not name:
            name = "texture"+str(self.idx)
        glBindTexture(GL_TEXTURE_2D, self.tex)
        glActiveTexture(GL_TEXTURE0 + self.idx)
        shader.setUniform("tex", self.idx)
        shader.setUniform("size",np.array([self.texWidth,self.texHeight],dtype = np.float32))
        shader.setUniform("hexmosaicSizeRatio",float(2**(-self.hierarchy)))
        shader.setUniform("even_odd_offset", self.even_odd_offset)
    def TexSize(self):
        return self.texHeight, self.texWidth
    def free(self):
        if self.tex:
            del self.img
            #glDeleteTextures(1, [self.tex])
            #self.tex = 0
            d = 1
    def SwitchTexture(self, filename,texType=GL_TEXTURE_2D):
        #读取纹理图片
        img = []
        img = Image.open(filename)
        img = np.array(img, np.uint8)#h,w,c
        self.texHeight, self.texWidth, _=img.shape 
        while (img.shape[0])%4!=0:
            row_to_be_added = np.zeros((1,img.shape[1],img.shape[2]))
            img = np.append(img, row_to_be_added, axis = 0)
            self.texHeight += 1

        while (img.shape[1])%4!=0:
            colum_to_be_added = np.zeros((img.shape[0],1,img.shape[2]))
            img = np.append(img, colum_to_be_added, axis = 1)
            self.texWidth += 1
        if img.shape[2]==4:
           img = np.delete(img,3,axis = 2)
        glTexImage2D(texType, 0, GL_RGB, self.texWidth, self.texHeight, 0, GL_RGB, GL_UNSIGNED_BYTE, img)
        glGenerateMipmap(texType)
