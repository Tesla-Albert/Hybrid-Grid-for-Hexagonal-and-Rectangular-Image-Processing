# -*- coding:utf-8 -*-


from OpenGL.GL import shaders
from OpenGL.GL import *
import numpy as np



class Hexagon_Mosaic_shader:
    def __init__(self):    
        # 顶点着色器
        vs = """
        #version 330 core
        in vec3 aPos;
        in vec2 aTex;
        out vec2 uv;
        
        void main (void) {
            gl_Position = vec4(aPos,1.0);
            uv = vec2(aTex.x, 1 - aTex.y);
        }

        """       
        # 片元着色器
        fs = """
        #version 330 core
        in vec2 uv;
        out vec4 FragColor;
        uniform sampler2D tex;
        uniform vec2 size;
        uniform float hexmosaicSizeRatio;
        uniform int even_odd_offset;
        float sizex,sizey;
        
        

        void main() 
        {           
            sizex = size.x + 0.5;//=4(width-dimention must be padded so that 4 can be divided)
            
            sizey = size.y + 1;//=4
            float TR = 1 * hexmosaicSizeRatio;
            float TB = 0.5 * hexmosaicSizeRatio;

            float x = uv.x * sizex;//w,sizex = image_width,uv is Texture Coordinate,in [0,1]
            float y = uv.y * sizey;//h,sizey = image_height

            int wx = int(x / TB);
            int wy = int(y / TR);

            vec2 v1,v2,v;
            float deltax = 0;
            if(((wx+even_odd_offset)&1) == (wy&1)){
                v1 = vec2(TB * (float(wx)), TR * (float(wy)));
                v2 = vec2(TB * (float(wx)+1), TR * (float(wy)+1));                
                }
            else{
                v1 = vec2(TB * (float(wx)), TR * (float(wy)+1));
                v2 = vec2(TB * (float(wx)+1), TR * (float(wy)));
                }
            
            float s1 = (v1.x - x)*(v1.x - x) + (v1.y - y)*(v1.y - y);
            float s2 = (v2.x - x)*(v2.x - x) + (v2.y - y)*(v2.y - y);
            
            
            if(s1<s2){
                v = v1;               
            }
            else{
                v = v2;                
            }
            int vx,vy;
            vx = int(v.x/0.5);
            vy = int(v.y/1);
            float sx,sy;
            sx = (vx - 1 - (vy+1+even_odd_offset)%2)/2+0.5;
            sy = vy-0.5;
            
            vec4 color = texture2D(tex,vec2((sx)/size.x, (sy)/size.y));
            gl_FragColor = color;
        }

        """
        
        vsProgram = shaders.compileShader(vs, GL_VERTEX_SHADER)
        fsProgram = shaders.compileShader(fs, GL_FRAGMENT_SHADER)
        self.shader = shaders.compileProgram(vsProgram, fsProgram)
    
    def use(self):
        glUseProgram(self.shader)

    def setUniform(self, name, value):
        self.use()
        loc = glGetUniformLocation(self.shader, name)
        dtype = type(value)
        if dtype == np.ndarray:
            size = value.size
            dtype = value.dtype
            funcs = {
                "int32": [glUniform1i, glUniform2i, glUniform3i, glUniform4i],
                "uint": [glUniform1ui, glUniform2ui, glUniform3ui, glUniform4ui],
                "float32": [glUniform1f, glUniform2f, glUniform3f, glUniform4f],
                "double": [glUniform1d, glUniform2d, glUniform3d, glUniform4d]
                    }
            func = funcs[str(dtype)][size - 1]#调用函数库里的函数
            func(loc, *value)
            return
        elif dtype == int or dtype == np.int32 or dtype == np.int64:
            glUniform1i(loc, value)
        elif dtype == float or dtype == np.float64 or dtype == np.float32:
            glUniform1f(loc, value)
        else:
            raise RuntimeError("参数类型不晓得")


    def setAttrib(self, name, size, dtype, stride, offset):
        loc = glGetAttribLocation(self.shader, name)
        glVertexAttribPointer(loc, size, dtype, False, stride, ctypes.c_void_p(offset))
        glEnableVertexAttribArray(loc)
    






