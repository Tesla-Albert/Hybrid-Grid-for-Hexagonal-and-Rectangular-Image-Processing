# -*- coding:utf-8 -*-

from OpenGL.GL import *
from OpenGL.GL import shaders 
import glfw




class Window:
    def __init__(self, width, height, title, bgcolor = (0.0, 0.0, 0.0, 1.0)):

        #初始化
        if not glfw.init():
            raise RuntimeError("GLFW初始化失败")
        #创建窗口
        self.width, self.height, self.title, self.bgColor = width, height, title, bgcolor
        self.window = glfw.create_window(width, height, title, None, None)#暂时忽略后面的两个参数
        self.dx = self.dy = 0
        self.press_ADDorSub_flag = 0
        self.press_PAGEUPorPAGEDOWN_flag = 0
        self.delta_hierarchy= 0
        self.delta = 0
        self.delta1 = 0
        self.delta_img_serialNum = 0
        self.scale = 1
        self.firstMouse = False
        self.WholeRate = 1
        #显示窗口
        self.show()

    def show(self):
        glfw.make_context_current(self.window)#通知glfw将窗口的上下文设置为当前线程的主上下文
        glfw.set_window_size_limits(self.window, self.width, self.height, self.width, self.height)#使得窗口大小不能更改
        glViewport(0, 0, self.width, self.height)#设置视口
        glEnable(GL_CULL_FACE)#开启背面剔除
        glEnable(GL_DEPTH_TEST)#开启深度测试
    
    def WindowResize(self, newWidth, newHeight):
        self.width, self.height = newWidth, newHeight
        #显示窗口
        self.show()
    def WindowSwitchFile(self, newFilename):
        self.title = newFilename
        self.show()
    def loop(self, render):
        while not glfw.window_should_close(self.window):
            #清空缓冲区
            glClearColor(*self.bgColor)
            glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)#清除颜色缓冲、深度缓冲
        
            #在这里 可以绘制物体
        
        
            #……
            render()
        
            #交换缓冲区
            glfw.swap_buffers(self.window)
            #提交事件
            glfw.poll_events()
            #监测窗口是否关闭
            if glfw.get_key(self.window, glfw.KEY_ESCAPE) == glfw.PRESS:
                glfw.set_window_should_close(self.window, True)
            
                
            self.keymove()

            
            glfw.set_scroll_callback(self.window,self.ScrollCallback)
            glfw.set_cursor_pos_callback(self.window,self.cursor_pos_callback)
            glfw.set_mouse_button_callback(self.window, self.mouse_button_callback)
            glfw.set_framebuffer_size_callback(self.window,self.framebuffer_size_callback)

        #结束循环并销毁窗口
        glfw.destroy_window(self.window)
        glfw.terminate()
    def ScrollCallback(self, window, xoffset, yoffset):
        self.scale+=yoffset/2
        if self.scale>1:
            self.scale = 1.1
        if self.scale<1:
            self.scale = 0.9
        self.WholeRate *= self.scale
    def keymove(self):

        if glfw.get_key(self.window, glfw.KEY_UP)==glfw.PRESS or \
            glfw.get_key(self.window, glfw.KEY_W)==glfw.PRESS:
            self.dy = 0.01
        if glfw.get_key(self.window, glfw.KEY_DOWN)==glfw.PRESS or \
            glfw.get_key(self.window, glfw.KEY_S)==glfw.PRESS:
            self.dy = -0.01
        if glfw.get_key(self.window, glfw.KEY_LEFT)==glfw.PRESS or\
            glfw.get_key(self.window, glfw.KEY_A)==glfw.PRESS:
            self.dx = -0.01
        if glfw.get_key(self.window, glfw.KEY_RIGHT)==glfw.PRESS or\
            glfw.get_key(self.window, glfw.KEY_D)==glfw.PRESS:
            self.dx = 0.01

        
        if glfw.get_key(self.window, glfw.KEY_KP_ADD)==glfw.PRESS:
            self.press_ADDorSub_flag = 1 
            self.delta = 1
        if glfw.get_key(self.window, glfw.KEY_KP_SUBTRACT)==glfw.PRESS:
            self.press_ADDorSub_flag = 1 
            self.delta = -1
        if glfw.get_key(self.window, glfw.KEY_KP_ADD )==glfw.RELEASE and glfw.get_key(self.window, glfw.KEY_KP_SUBTRACT)==glfw.RELEASE:            
            if self.press_ADDorSub_flag ==1:
                self.delta_hierarchy = self.delta
            self.delta = 0



        if glfw.get_key(self.window, glfw.KEY_PAGE_UP)==glfw.PRESS:
            self.press_PAGEUPorPAGEDOWN_flag = 1 
            self.delta1 = -1
        if glfw.get_key(self.window, glfw.KEY_PAGE_DOWN)==glfw.PRESS:
            self.press_PAGEUPorPAGEDOWN_flag = 1 
            self.delta1 = 1
        if glfw.get_key(self.window, glfw.KEY_PAGE_UP )==glfw.RELEASE and glfw.get_key(self.window, glfw.KEY_PAGE_DOWN)==glfw.RELEASE:            
            if self.press_PAGEUPorPAGEDOWN_flag ==1:
                self.delta_img_serialNum = self.delta1
            self.delta1 = 0



    def cursor_pos_callback(self, window, xpos, ypos):
        self.cursor_x = xpos
        self.cursor_y = ypos
        if self.firstMouse:
            self.dx += (self.lastX - self.cursor_x)/(self.width)
            self.dy += (self.cursor_y - self.lastY)/(self.height)
        self.lastX = xpos
        self.lastY = ypos



    def framebuffer_size_callback(self, window, width, height):
        self.width = width
        self.height = height
        glViewport(0,0,self.width,self.height)
    def mouse_button_callback(self, window, button, action, mods):
        if action == glfw.PRESS:
            if button == glfw.MOUSE_BUTTON_LEFT:
                self.firstMouse = True
        if action == glfw.RELEASE:
            if button == glfw.MOUSE_BUTTON_LEFT:
                self.firstMouse = False
