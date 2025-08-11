from scipy.interpolate import griddata, interp2d
import numpy as np
import math
import torch
from numba import cuda
import warnings
warnings.filterwarnings("ignore")
@cuda.jit()
def resample_on_hexagonal_grids(sample_coords, image_array, new_image, method):
    """
    重采样
    Args:
        sample_coords: 2*h*w——(x, y)*h*w
        image_array: 3*h*w——(x, y, z)*h*w
    """
    # TODO 线性重采样写完后记得补上最邻近法的重采样

    method = method
    tj = cuda.blockIdx.x * cuda.blockDim.x + cuda.threadIdx.x  # w方向
    ti = cuda .blockIdx.y * cuda.blockDim.y + cuda.threadIdx.y # h方向

    h, w = image_array.shape[1:3]
    h1, w1 = sample_coords.shape[1:3]
    if ti < h1 and tj < w1:
        x_, y_ = sample_coords[:, ti, tj] # 几何坐标，笛卡尔坐标系，原点在图像中心
        # 几何坐标，坐标系原点从图像中心移动到图像左上角第一个像元中心的位置
        # 同时将笛卡尔坐标系转换为仿射坐标系
        i_ = x_ + (h - 1)/2
        j_ = 0.5 * i_ + (y_ + (w - 0.5)/2)
        # 仿射坐标的整数部分
        i_n = int(i_)
        j_n = int(j_)
        # 仿射坐标的小数部分
        i_f = i_ - i_n
        j_f = j_ - j_n


        # 采样点周围的四个原始数据点坐标
        # 输入为仿射整数坐标，输出，即{i_1, j_1; i_2, j_2; ...; i_4, j_4}为笛卡尔整数坐标
        i_1 = i_n
        j_1 = j_n     - (i_n + 1) // 2

        i_2 = i_n + 1
        j_2 = j_n     - (i_n + 2) // 2

        i_3 = i_n
        j_3 = j_n + 1 - (i_n + 1) // 2

        i_4 = i_n + 1
        j_4 = j_n + 1 - (i_n + 2) // 2

        if method == 2:
            # t1~t4为采样点周边四个原始数据点的取值
            if i_1 >= 0 and j_1 >= 0 and i_1 < h and j_1 < w:
                t1 = image_array[2][i_1, j_1]
            else: t1 = 0
            if i_2 >= 0 and j_2 >= 0 and i_2 < h and j_2 < w:
                t2 = image_array[2][i_2, j_2]
            else: t2 = 0
            if i_3 >= 0 and j_3 >= 0 and i_3 < h and j_3 < w:
                t3 = image_array[2][i_3, j_3]
            else: t3 = 0
            if i_4 >= 0 and j_4 >= 0 and i_4 < h and j_4 < w:
                t4 = image_array[2][i_4, j_4]
            else: t4 = 0
            # p.._x, p.._y为采样点周边四个原始数据点的仿射几何坐标，原点在左上角第一个像元中心（下午来改）
            # 计算的时候和仿射坐标i_, j_来得到每个值的权重
            p1_x = i_n
            p1_y = j_n
            p2_x = i_n + 1
            p2_y = j_n
            p3_x = i_n
            p3_y = j_n + 1
            p4_x = i_n + 1
            p4_y = j_n + 1
            # 对采样点进行双线性重采样的计算
            t = (
                    (p2_x - i_) / (p2_x - p1_x) * t1 + (i_ - p1_x) / (p2_x - p1_x) * t2
                ) * \
                (p3_y - j_) / (p3_y - p1_y) + \
                (
                    (p2_x - i_) / (p2_x - p1_x) * t1 + (i_ - p1_x) / (p2_x - p1_x) * t2
                ) * \
                (j_ - p1_y) / (p3_y - p1_y)
            #
            new_image[ti, tj] = t
            # print('bilinear not supported yet, linear is used')


        if method ==0 or method == 1:
            # t1~t3为采样点周边三个最近的原始数据点的取值
            if i_f > j_f:
                if  i_1 >= 0 and j_1 >= 0 and i_1 < h and j_1 < w:
                    t1 = image_array[2][i_1, j_1]
                else:
                    t1 = 0
                if i_2 >= 0 and j_2 >= 0 and i_2 < h and j_2 < w:
                    t2 = image_array[2][i_2, j_2]
                else:
                    t2 = 0
                if i_4 >= 0 and j_4 >= 0 and i_4 < h and j_4 < w:
                    t3 = image_array[2][i_4, j_4]
                else:
                    t3 = 0

                p1_x = i_n - (h - 1) / 2
                p1_y = j_n - i_n / 2 - (w - 0.5) / 2
                p2_x = (i_n + 1) - (h - 1) / 2
                p2_y = j_n - (i_n + 1) / 2 - (w - 0.5) / 2
                p3_x = (i_n + 1) - (h - 1) / 2
                p3_y = (j_n + 1) - (i_n + 1) / 2 - (w - 0.5) / 2
            else:
                if  i_1 >= 0 and j_1 >= 0 and i_1 < h and j_1 < w:
                    t1 = image_array[2][i_1, j_1]
                else:
                    t1 = 0
                if i_3 >= 0 and j_3 >= 0 and i_3 < h and j_3 < w:
                    t2 = image_array[2][i_3, j_3]
                else:
                    t2 = 0
                if i_4 >= 0 and j_4 >= 0 and i_4 < h and j_4 < w:
                    t3 = image_array[2][i_4, j_4]
                else:
                    t3 = 0


                p1_x = i_n - (h - 1) / 2
                p1_y = j_n - i_n / 2 - (w - 0.5) / 2
                p2_x = i_n - (h - 1) / 2
                p2_y = (j_n + 1) - i_n / 2 - (w - 0.5) / 2
                p3_x = (i_n + 1) - (h - 1) / 2
                p3_y = (j_n + 1) - (i_n + 1) / 2 - (w - 0.5) / 2

            #
            if method==0:
                    d1 = (x_ - p1_x) * (x_ - p1_x) + (y_ - p1_y) * (y_ - p1_y)
                    d2 = (x_ - p2_x) * (x_ - p2_x) + (y_ - p2_y) * (y_ - p2_y)
                    d3 = (x_ - p3_x) * (x_ - p3_x) + (y_ - p3_y) * (y_ - p3_y)
                    if d1<=d2 and d1<=d3:
                        new_image[ti, tj] = t1
                    if d2<d1 and d2<=d3:
                        new_image[ti, tj] = t2
                    if d3<=d1 and d3<d2:
                        new_image[ti, tj] = t3
            if method==1:#线性插值
                    S1 = 0.5*abs((x_ - p2_x) * (y_ - p3_y) - (y_ - p2_y) * (x_ - p3_x))
                    S2 = 0.5*abs((x_ - p1_x) * (y_ - p3_y) - (y_ - p1_y) * (x_ - p3_x))
                    S3 = 0.5*abs((x_ - p1_x) * (y_ - p2_y) - (y_ - p1_y) * (x_ - p2_x))
                    alpha = S1/(S1+S2+S3)
                    beta = S2/(S1+S2+S3)
                    gamma = S3/(S1+S2+S3)
                    new_image[ti, tj] = alpha*t1 + beta*t2 + gamma*t3

            # else:
            #     new_image[ti, tj] = 0
def image_geometric_transformation_gpu(image:np.array, H:np.array=np.eye(3), interpolation='nearest', offset=0)->np.array:
    # H = np.array([
    #     [h_scale, 0, 0],
    #     [0, w_scale, 0],
    #     [0, 0, 1]
    # ])
    method_dict = {
        'nearest':0,
        'linear':1,
        'bilinear':2,
    }
    method = method_dict[interpolation]
    if image.ndim==3:
        c, h, w = image.shape
    elif image.ndim==2:
        h, w = image.shape
        c = 1
        image = np.expand_dims(image, axis=0)
    else:
        raise Exception(f"dim of image should be 2 or 3, but got dim = {image.ndim} instead")
    # 生成原图格心
    imgcoor = np.mgrid[0:h:1., 0:w:1.]
    # h方向
    imgcoor[0] += 0.5
        # h方向中心化
    imgcoor[0] -= h / 2.
    # w方向
    imgcoor[1] += 0.5
    imgcoor[1][(1-offset)::2] += 0.5
        # w方向中心化
    imgcoor[1] -= (w + 0.5) / 2.
    # 转齐次坐标
    homogeneous_imgcoor = np.vstack([imgcoor, np.ones((1, imgcoor.shape[1], imgcoor.shape[2]))])
    # 新图格心坐标矩阵
    # new_homogeneous_imgcoor = np.einsum('ij, jkl -> ikl', H, homogeneous_imgcoor)
    # new_homogeneous_imgcoor = new_homogeneous_imgcoor / new_homogeneous_imgcoor[2]

    # 生成反变换的新图格心
    # 新图上界下界，由老图格心变换后得到其边界

    left_top = [-(h / 2 - 0.5), -((w + 0.5) / 2 - 0.5), 1.]
    right_top = [-(h / 2 - 0.5), (w + 0.5) / 2 - 0.5, 1.]
    left_bottom = [h / 2 - 0.5, -((w + 0.5) / 2 - 0.5), 1.]
    right_bottom = [h / 2 - 0.5, (w + 0.5) / 2 - 0.5, 1.]
    imgcorner = np.array([
        left_top, right_top, left_bottom, right_bottom
    ], dtype=np.double).T

    new_homogeneous_imgcorner = np.matmul(H, imgcorner)

    h_1_inf = np.min(new_homogeneous_imgcorner[0])
    w_1_inf = np.min(new_homogeneous_imgcorner[1])
    h_1_sup = np.max(new_homogeneous_imgcorner[0])
    w_1_sup = np.max(new_homogeneous_imgcorner[1])

    h1_f = (h_1_sup - h_1_inf)
    w1_f = (w_1_sup - w_1_inf)

    h1_start = h_1_inf + (h1_f - int(h1_f)) / 2
    h1_end = h_1_sup - (h1_f - int(h1_f)) / 2
    w1_start = w_1_inf + (w1_f + 0.5 - int(w1_f + 0.5)) / 2
    w1_end = w_1_sup - (w1_f + 0.5 - int(w1_f + 0.5)) / 2
    # 新图坐标系的定义
    # h_1 = int(h_1_sup - h_1_inf)
    # w_1 = int(w_1_sup - w_1_inf)
    new_img_local_coor = np.mgrid[int(h_1_inf):h_1_sup + 1:1., int(h_1_inf):h_1_sup + 0.5:1.]
    new_img_local_coor[1][1::2] += 0.5
    homogeneous_new_img_local_coor \
        = np.vstack(
        [new_img_local_coor,
         np.ones((1, new_img_local_coor.shape[1], new_img_local_coor.shape[2]))]
    )
    h_1 = new_img_local_coor.shape[1]
    w_1 = new_img_local_coor.shape[2]
    # 反变换格心
    new_homogeneous_imgcoor_inverse \
        = np.einsum(
        'ij, jkl -> ikl',
        np.linalg.inv(H),
        homogeneous_new_img_local_coor
    )
    new_image = np.empty((c, homogeneous_new_img_local_coor.shape[1], homogeneous_new_img_local_coor.shape[2]))
    for i in range(c):
        image_array = np.stack((imgcoor[0], imgcoor[1], image[i]))
        sample_coords = new_homogeneous_imgcoor_inverse[0:2]# 2 * h1 * w1
        new_image_tmp = np.zeros(homogeneous_new_img_local_coor.shape[1:3])
        torch.cuda.empty_cache()
        cu_image_array = cuda.to_device(image_array)
        cu_sample_coords = cuda.to_device(sample_coords)
        cu_new_image = cuda.to_device(new_image_tmp)
        threadsperblock = (32, 32)
        blockspergrid_y = int(math.ceil(h_1 / threadsperblock[0]))
        blockspergrid_x = int(math.ceil(w_1 / threadsperblock[1]))
        blockspergrid = (blockspergrid_x, blockspergrid_y)
        resample_on_hexagonal_grids[blockspergrid, threadsperblock](cu_sample_coords, cu_image_array, cu_new_image, method)
        new_image_tmp = cu_new_image.copy_to_host()
        new_image[i] = new_image_tmp





    # 插值



    return new_image.squeeze()

def image_geometric_transformation_cpu(image:np.array, H:np.array=np.eye(3), interpolation='nearest', offset=0)->np.array:
    # H = np.array([
    #     [h_scale, 0, 0],
    #     [0, w_scale, 0],
    #     [0, 0, 1]
    # ])
    h = image.shape[1]
    w = image.shape[2]
    # 生成原图格心
    imgcoor = np.mgrid[0:h:1., 0:w:1.]
    # h方向
    imgcoor[0] += 0.5
        # h方向中心化
    imgcoor[0] -= h / 2.
    # w方向
    imgcoor[1] += 0.5
    imgcoor[1][(1-offset)::2] += 0.5
        # w方向中心化
    imgcoor[1] -= (w + 0.5) / 2.
    # 转齐次坐标
    homogeneous_imgcoor = np.vstack([imgcoor, np.ones((1, imgcoor.shape[1], imgcoor.shape[2]))])
    # 格心在新图中的位置
    # new_homogeneous_imgcoor = np.einsum('ij, jkl -> ikl', H, homogeneous_imgcoor)
    # new_homogeneous_imgcoor = new_homogeneous_imgcoor / new_homogeneous_imgcoor[2]

    # 生成反变换的新图格心
    # 新图上界下界，由老图格心变换后得到其边界
    left_top = [-(h / 2 - 0.5), -((w + 0.5) / 2 - 0.5), 1.]
    right_top = [-(h / 2 - 0.5), (w + 0.5) / 2 - 0.5, 1.]
    left_bottom = [h / 2 - 0.5, -((w + 0.5) / 2 - 0.5), 1.]
    right_bottom = [h / 2 - 0.5, (w + 0.5) / 2 - 0.5, 1.]
    imgcorner = np.array([
        left_top, right_top, left_bottom, right_bottom
    ], dtype=np.double).T

    new_homogeneous_imgcorner = np.matmul(H, imgcorner)

    h_1_inf = np.min(new_homogeneous_imgcorner[0])
    w_1_inf = np.min(new_homogeneous_imgcorner[1])
    h_1_sup = np.max(new_homogeneous_imgcorner[0])
    w_1_sup = np.max(new_homogeneous_imgcorner[1])

    h1_f = (h_1_sup - h_1_inf)
    w1_f = (w_1_sup - w_1_inf)

    h1_start = h_1_inf + (h1_f - int(h1_f)) / 2
    h1_end = h_1_sup - (h1_f - int(h1_f)) / 2
    w1_start = w_1_inf + (w1_f + 0.5 - int(w1_f + 0.5)) / 2
    w1_end = w_1_sup - (w1_f + 0.5 - int(w1_f + 0.5)) / 2
    # 新图坐标系的定义
    # h_1 = int(h_1_sup - h_1_inf)
    # w_1 = int(w_1_sup - w_1_inf)
    new_img_local_coor = np.mgrid[int(h1_start):h1_end + 1:1., int(w1_start):w1_end + 0.5:1.]
    new_img_local_coor[1][1::2] += 0.5
    homogeneous_new_img_local_coor \
        = np.vstack(
        [new_img_local_coor,
         np.ones((1, new_img_local_coor.shape[1], new_img_local_coor.shape[2]))]
    )
    h_1 = new_img_local_coor.shape[1]
    w_1 = new_img_local_coor.shape[2]
    # 反变换格心
    new_homogeneous_imgcoor_inverse \
        = np.einsum(
        'ij, jkl -> ikl',
        np.linalg.inv(H),
        homogeneous_new_img_local_coor
    )
    xi = new_homogeneous_imgcoor_inverse[0].reshape(-1)
    yi = new_homogeneous_imgcoor_inverse[1].reshape(-1)
    x = imgcoor.reshape(2, -1)[0]
    y = imgcoor.reshape(2, -1)[1]

    # 插值
    newimg = np.empty((image.shape[0], h_1, w_1))

    for i in range(image.shape[0]):
        newimg[i] = griddata((x, y),
                             image[i].flatten(),
                             (xi, yi),
                             method=interpolation).reshape(h_1, w_1)
    return newimg

def image_geometric_transformation(img:np.array, H:np.array=np.eye(3), interpolation='nearest', offset=0, device='cuda0')->np.array:
    if device=='cuda0':
        return image_geometric_transformation_gpu(img, H, interpolation, offset)
    if device=='cpu':
        return image_geometric_transformation_cpu(img, H, interpolation, offset)


def hex_to_square_resample(hex_image, square_size=None, interpolation='nearest', offset=0):
    method_dict = {
        'nearest': 0,
        'linear': 1,
        'bilinear': 2,
    }
    method = method_dict[interpolation]
    if hex_image.ndim == 3:
        c, h, w = hex_image.shape
    elif hex_image.ndim == 2:
        h, w = hex_image.shape
        c = 1
        hex_image = np.expand_dims(hex_image, axis=0)
    else:
        raise Exception(f"dim of image should be 2 or 3, but got dim = {hex_image.ndim} instead")

    if square_size == None:
        square_size = (h, w)
    h1, w1 = square_size
    # generate central elements' coordinate array of original hexagonal grid
    imgcoor = np.mgrid[0:h:1., 0:w:1.]
    # h方向
    imgcoor[0] += 0.5
    # h方向中心化
    imgcoor[0] -= h / 2.
    # w方向
    imgcoor[1] += 0.5
    imgcoor[1][(1 - offset)::2] += 0.5
    # w方向中心化
    imgcoor[1] -= (w + 0.5) / 2.
    # 转齐次坐标
    # homogeneous_imgcoor = np.vstack([imgcoor, np.ones((1, imgcoor.shape[1], imgcoor.shape[2]))])
    # 方格图像上界下界
    left_top = [-(h / 2 - 0.5), -((w + 0.5) / 2 - 0.75), 1.]
    right_top = [-(h / 2 - 0.5), (w + 0.5) / 2 - 0.75, 1.]
    left_bottom = [h / 2 - 0.5, -((w + 0.5) / 2 - 0.75), 1.]
    right_bottom = [h / 2 - 0.5, (w + 0.5) / 2 - 0.75, 1.]
    # 图像角点坐标矩阵（把四个角点装载为一个矩阵）
    imgcorner = np.array([
        left_top, right_top, left_bottom, right_bottom
    ], dtype=np.double).T

    h_1_inf = np.min(imgcorner[0])
    w_1_inf = np.min(imgcorner[1])
    h_1_sup = np.max(imgcorner[0])
    w_1_sup = np.max(imgcorner[1])

    new_img_local_coor = np.array(
        np.meshgrid(np.linspace(h_1_inf, h_1_sup, h1), np.linspace(w_1_inf, w_1_sup, w1), indexing='ij')
    )
    homogeneous_new_img_local_coor \
        = np.vstack(
        [new_img_local_coor,
         np.ones((1, new_img_local_coor.shape[1], new_img_local_coor.shape[2]))]
    )
    h_1 = new_img_local_coor.shape[1]
    w_1 = new_img_local_coor.shape[2]
    # 反变换格心
    # new_homogeneous_imgcoor_inverse \
    #     = np.einsum(
    #     'ij, jkl -> ikl',
    #     np.linalg.inv(H),
    #     homogeneous_new_img_local_coor
    # )
    new_image = np.empty((c, homogeneous_new_img_local_coor.shape[1], homogeneous_new_img_local_coor.shape[2]))
    for i in range(c):
        image_array = np.stack((imgcoor[0], imgcoor[1], hex_image[i]))
        sample_coords = homogeneous_new_img_local_coor[0:2]  # 2 * h1 * w1
        new_image_tmp = np.zeros(homogeneous_new_img_local_coor.shape[1:3])
        torch.cuda.empty_cache()
        cu_image_array = cuda.to_device(image_array)
        cu_sample_coords = cuda.to_device(sample_coords)
        cu_new_image = cuda.to_device(new_image_tmp)
        threadsperblock = (32, 32)
        blockspergrid_y = int(math.ceil(h_1 / threadsperblock[0]))
        blockspergrid_x = int(math.ceil(w_1 / threadsperblock[1]))
        blockspergrid = (blockspergrid_x, blockspergrid_y)
        resample_on_hexagonal_grids[blockspergrid, threadsperblock](cu_sample_coords, cu_image_array, cu_new_image,
                                                                    method)
        new_image_tmp = cu_new_image.copy_to_host()
        new_image[i] = new_image_tmp
    return new_image

def hexresize(image, dsize, interpolation='linear'):
    if image.ndim == 3:
        c, h, w = image.shape
    elif image.ndim == 2:
        h, w = image.shape
        c = 1
    else:
        raise Exception(f"dim of image should be 2 or 3, but got dim = {image.ndim} instead")

    h1, w1 = dsize
    imgcoor = np.stack(
        np.meshgrid(
            np.arange(h),
            np.arange(w),
            indexing='ij'
        ),
        axis=0
    ).astype(float)  # (x(h, w), y(h, w)), dtype = torch.float32
    image = np.transpose(image, [1, 2, 0])  # c, h, w -> h, w, c
    # h方向
    imgcoor[0] += 0.5
    # h方向中心化
    imgcoor[0] -= h / 2.
    # w方向
    imgcoor[1] += 0.5
    imgcoor[1][(1 - offset)::2] += 0.5
    # w方向中心化
    imgcoor[1] -= (w + 0.5) / 2.

    # 原六边形图像上界下界（以砖墙格网代替）
    left_top = [-(h / 2 - 0.5), -((w + 0.5) / 2 - 0.5), 1.]
    right_top = [-(h / 2 - 0.5), (w + 0.5) / 2 - 0.5, 1.]
    left_bottom = [h / 2 - 0.5, -((w + 0.5) / 2 - 0.5), 1.]
    right_bottom = [h / 2 - 0.5, (w + 0.5) / 2 - 0.5, 1.]

    # 图像角点坐标矩阵（把四个角点装载为一个矩阵）
    imgcorner = np.array([
        left_top, right_top, left_bottom, right_bottom
    ]).transpose()

    h1_inf = np.min(imgcorner[0])
    w1_inf = np.min(imgcorner[1])
    h1_sup = np.max(imgcorner[0])
    w1_sup = np.max(imgcorner[1])

    new_img_local_coor = np.stack(
        np.meshgrid(
            np.linspace(h1_inf, h1_sup, h1),
            np.linspace(w1_inf, w1_sup, w1),
            indexing='ij'
        ),
        axis=0
    )
    homogeneous_new_img_local_coor \
        = np.vstack(
        [new_img_local_coor,
         np.ones((1, new_img_local_coor.shape[1], new_img_local_coor.shape[2]))]
    )
    h1 = new_img_local_coor.shape[1]
    w1 = new_img_local_coor.shape[2]
    # 反变换格心

    new_image = np.empty(
        (c,
         homogeneous_new_img_local_coor.shape[1],
         homogeneous_new_img_local_coor.shape[2]),
    )

    new_image = np.empty((c, homogeneous_new_img_local_coor.shape[1], homogeneous_new_img_local_coor.shape[2]))
    for i in range(c):
        image_array = np.stack((imgcoor[0], imgcoor[1], hex_image[i]))
        sample_coords = homogeneous_new_img_local_coor[0:2]  # 2 * h1 * w1
        new_image_tmp = np.zeros(homogeneous_new_img_local_coor.shape[1:3])
        torch.cuda.empty_cache()
        cu_image_array = cuda.to_device(image_array)
        cu_sample_coords = cuda.to_device(sample_coords)
        cu_new_image = cuda.to_device(new_image_tmp)
        threadsperblock = (32, 32)
        blockspergrid_y = int(math.ceil(h_1 / threadsperblock[0]))
        blockspergrid_x = int(math.ceil(w_1 / threadsperblock[1]))
        blockspergrid = (blockspergrid_x, blockspergrid_y)
        resample_on_hexagonal_grids[blockspergrid, threadsperblock](cu_sample_coords, cu_image_array, cu_new_image,
                                                                    method)
        new_image_tmp = cu_new_image.copy_to_host()
        new_image[i] = new_image_tmp
    return new_image


if __name__ == '__main__':
    from HexImage import HEXIMAGE
    from Image import IMAGE
    import mmcv
    # path = r"D:\mmsegmentation-0.29.1\Data\Hex_Potsdam\labels\test\1081_hex.tif"
    # path = r"D:\mmsegmentation-0.29.1\Data\Hex_Potsdam\images\test\52.jpg"
    path = r"D:\mmsegmentation-0.29.1\Data\Hex_FBP\images\test\GF2_PMS1__L1A0001064454-MSS1_0_0.tif"
    image = HEXIMAGE(path, heximagetype=2)
    # image.Hex_imshow()

    # H = np.array([
    #     [np.cos(np.pi/6), -np.sin(np.pi/6), 0],
    #     [np.sin(np.pi/6), np.cos(np.pi/6), 0],
    #     [0, 0, 1]
    # ])
    h_scale = 399/298
    w_scale = 399.5 / 298.5
    H = np.array([
            [4, 0, 0],
            [0, 4, 0],
            [0, 0, 1]
        ])

    # image1_Hexdata = image_geometric_transformation_gpu(image.HexagonImage, H, interpolation='linear')
    # image1_Hexdata = mmcv.imresize(
    #     image.HexagonImage.transpose(1, 2, 0),
    #     size=(2000, 2000),
    #     interpolation='bilinear',
    # ).transpose(2, 0, 1)
    # image1_data = hex_to_square_resample(
    #     image.HexagonImage,
    #     (1024, 1024),
    #     'nearest'
    # )
    image1_data = image_geometric_transformation_gpu(
        image.HexagonImage,
        H,
        interpolation='bilinear'
    )
    image1 = HEXIMAGE(data=image1_data)
    image1.Hex_imshow()
    print(image1.shape)
