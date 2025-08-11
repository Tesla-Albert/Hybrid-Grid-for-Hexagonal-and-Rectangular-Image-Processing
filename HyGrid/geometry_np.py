import numpy as np
from typing import Optional, Tuple, Union, List
import warnings
import cv2
warnings.filterwarnings("ignore")
def image_geometric_transformation(img:np.array, H:np.array=np.eye(3), interpolation='nearest', offset=0)->np.array:
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
    if img.ndim==3:
        c, h, w = img.shape
    elif img.ndim==2:
        h, w = img.shape
        c = 1
        img = np.expand_dims(img, axis=0)
    else:
        raise Exception(f"dim of image should be 2 or 3, but got dim = {img.ndim} instead")
    # image = torch.tensor(image, device='cuda')
    # H = torch.tensor(H).to(torch.float64)
    # 生成原图格心
    imgcoor = np.stack(
        np.meshgrid(
            np.arange(h),
            np.arange(w),
            indexing='ij'
        ),
        axis=0
    ).astype(float)
    img = img.transpose(1, 2, 0)# c, h, w -> h, w, c
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
    # homogeneous_imgcoor = np.vstack([imgcoor, np.ones((1, imgcoor.shape[1], imgcoor.shape[2]))])
    # 新图格心坐标矩阵
    # new_homogeneous_imgcoor = np.einsum('ij, jkl -> ikl', H, homogeneous_imgcoor)
    # new_homogeneous_imgcoor = new_homogeneous_imgcoor / new_homogeneous_imgcoor[2]

    # 生成反变换的新图格心
    # 新图上界下界，由老图格心变换后得到其边界
    # 新图四个角点
    left_top = [-(h/2-0.5), -((w+0.5)/2-0.5), 1.]
    right_top = [-(h/2-0.5),  (w+0.5)/2-0.5, 1.]
    left_bottom = [h/2-0.5, -((w+0.5)/2-0.5), 1.]
    right_bottom = [h/2-0.5,  (w+0.5)/2-0.5, 1.]
    imgcorner = np.array([
        left_top, right_top, left_bottom, right_bottom
    ]).transpose()
    new_homogeneous_imgcorner = np.matmul(H, imgcorner)

    h1_inf = np.min(new_homogeneous_imgcorner[0], axis=-1)
    w1_inf = np.min(new_homogeneous_imgcorner[1], axis=-1)
    h1_sup = np.max(new_homogeneous_imgcorner[0], axis=-1)
    w1_sup = np.max(new_homogeneous_imgcorner[1], axis=-1)

    h1_f = (h1_sup - h1_inf)
    w1_f = (w1_sup - w1_inf)


    # 新图坐标系的定义
    # 生成新图的格心矩阵(由于上下界由原图格心变换得到，所以新图格心的h坐标直接从下界开始，不加0.5
    # 而w坐标，从下界开始到上界-0.5(第1行，输出统一为offset=0，所以缩进0.5）+1. =到sup+0.5
    new_img_local_coor = np.stack(
        np.meshgrid(
        np.arange(h1_inf, h1_sup+1, 1),
        np.arange(w1_inf, w1_sup+0.5, 1),
        indexing='ij',
        ),
        axis=0
    )
    new_img_local_coor[1][1::2] += 0.5

    homogeneous_new_img_local_coor \
        = np.vstack(
        [new_img_local_coor,
         np.ones((1, new_img_local_coor.shape[1], new_img_local_coor.shape[2]))]
    )
    h1 = new_img_local_coor.shape[1]
    w1 = new_img_local_coor.shape[2]
    # 反变换格心
    new_homogeneous_imgcoor_inverse \
        = np.einsum(
        'ij, jkl -> ikl',
        np.linalg.inv(H),
        homogeneous_new_img_local_coor
    )
    new_image = np.empty(
        (c,
         homogeneous_new_img_local_coor.shape[1],
         homogeneous_new_img_local_coor.shape[2]),
    )
    # 插值的时候还是得利用仿射坐标来确定采样点在哪三个已知点之间
    x_, y_ = new_homogeneous_imgcoor_inverse[0], new_homogeneous_imgcoor_inverse[1]

    i_ = x_ + (h - 1) * 0.5
    j_ = 0.5 * i_ + y_ + (w - 0.5) * 0.5

    # 整数部分（采样点归属的单元）
    i_n = i_.astype(int)
    j_n = j_.astype(int)

    # 小数部分
    i_f = i_ - i_n
    j_f = j_ - j_n

    # 采样点周围四个真值点的仿射索引对应的offset索引
    i_1 = (i_n)
    j_1 = (j_n - ((i_n + 1) / 2).astype(int))
    i_2 = (i_n + 1)
    j_2 = (j_n - ((i_n + 2) / 2).astype(int))
    i_3 = (i_n)
    j_3 = (j_n + 1 - ((i_n + 1) / 2).astype(int))
    i_4 = (i_n + 1)
    j_4 = (j_n + 1 - ((i_n + 2) / 2).astype(int))

    # 取1就选2号点，取0就取3号点
    up_down_flag = (i_f > j_f)

    # range_flag = (i_1 >= 0 and j_1 >= 0 and i_1 < h and j_1 < w\
    #         and i_2 >= 0 and j_2 >= 0 and i_2 < h and j_2 < w\
    #         and i_4 >= 0 and j_4 >= 0 and i_4 < h and j_4 < w).float()+
    range_flag1 = (i_1 >= 0).astype(float) * (j_1 >= 0).astype(float) * \
                  (i_1 < h).astype(float) * (j_1 < w).astype(float)
    range_flag2 = (i_2 >= 0).astype(float) * (j_2 >= 0).astype(float) * \
                  (i_2 < h).astype(float) * (j_2 < w).astype(float)
    range_flag3 = (i_3 >= 0).astype(float) * (j_3 >= 0).astype(float) * \
                  (i_3 < h).astype(float) * (j_3 < w).astype(float)
    range_flag4 = (i_4 >= 0).astype(float) * (j_4 >= 0).astype(float) * \
                  (i_4 < h).astype(float) * (j_4 < w).astype(float)
    valid_indices1 = (range_flag1 == 1.)
    valid_indices2 = (range_flag2 == 1.)
    valid_indices3 = (range_flag3 == 1.)
    valid_indices4 = (range_flag4 == 1.)

    p1 = np.zeros((h1, w1, c), dtype=img.dtype)
    p2 = np.zeros((h1, w1, c), dtype=img.dtype)
    p3 = np.zeros((h1, w1, c), dtype=img.dtype)

    p1[valid_indices1] = img[i_1[valid_indices1], j_1[valid_indices1]]
    p2[valid_indices2 & up_down_flag] = img[i_2[valid_indices2 & up_down_flag], j_2[valid_indices2 & up_down_flag]]
    p2[valid_indices3 & ~up_down_flag] = img[i_3[valid_indices3 & ~up_down_flag], j_3[valid_indices3 & ~up_down_flag]]
    p3[valid_indices4] = img[i_4[valid_indices4], j_4[valid_indices4]]

    # 真值点仿射索引转笛卡尔坐标
    p1_x = i_n - ((h - 1) / 2)
    p1_y = j_n - i_n / 2 - (w - 0.5) / 2
    p2_x = (i_n + up_down_flag.astype(float)) - (h - 1) / 2
    p2_y = (j_n + 1 - up_down_flag.astype(float)) - (i_n + up_down_flag.astype(float)) / 2 - (w - 0.5) / 2
    p3_x = (i_n + 1) - (h - 1) / 2
    p3_y = (j_n + 1) - (i_n + 1) / 2 - (w - 0.5) / 2

    if method == 0:
        d1 = (x_ - p1_x) * (x_ - p1_x) + (y_ - p1_y) * (y_ - p1_y)
        d2 = (x_ - p2_x) * (x_ - p2_x) + (y_ - p2_y) * (y_ - p2_y)
        d3 = (x_ - p3_x) * (x_ - p3_x) + (y_ - p3_y) * (y_ - p3_y)

        d = np.stack((d1, d2, d3), axis=0)
        min_values, min_indices = np.min(d, axis=0)

        # 初始化 output_image
        new_image = np.zeros((h1, w1, c), dtype=img.dtype)
        new_image[min_indices == 0] = p1[min_indices == 0]  # 对应 d1 的颜色
        new_image[min_indices == 1] = p2[min_indices == 1]  # 对应 d2 的颜色
        new_image[min_indices == 2] = p3[min_indices == 2]  # 对应 d3 的颜色

    if method == 1:
        S1 = 0.5 * abs((x_ - p2_x) * (y_ - p3_y) - (y_ - p2_y) * (x_ - p3_x))
        S2 = 0.5 * abs((x_ - p1_x) * (y_ - p3_y) - (y_ - p1_y) * (x_ - p3_x))
        S3 = 0.5 * abs((x_ - p1_x) * (y_ - p2_y) - (y_ - p1_y) * (x_ - p2_x))
        alpha = np.expand_dims((S1 / (S1 + S2 + S3)), axis=-1)
        beta = np.expand_dims((S2 / (S1 + S2 + S3)), axis=-1)
        gamma = np.expand_dims((S3 / (S1 + S2 + S3)), axis=-1)
        new_image = alpha * p1 + beta * p2 + gamma * p3

    return np.transpose(new_image, [2, 0, 1]).squeeze()

def hex_to_rect_resample(hex_image, rect_dsize=None, interpolation='nearest', offset=0):
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
    else:
        raise Exception(f"dim of image should be 2 or 3, but got dim = {hex_image.ndim} instead")

    if rect_dsize == None:
        rect_dsize = (h, w)

    h1, w1 = rect_dsize

    image = np.array(hex_image)
    # generate central elements' coordinate array of original hexagonal grid
    # 生成原图格心
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
    # 插值的时候还是得利用仿射坐标来确定采样点在哪三个已知点之间
    x_, y_ = homogeneous_new_img_local_coor[0], homogeneous_new_img_local_coor[1]

    i_ = x_ + (h - 1) * 0.5
    j_ = 0.5 * i_ + y_ + (w - 0.5) * 0.5

    # 整数部分（采样点归属的单元）
    i_n = (i_).astype(int)
    j_n = (j_).astype(int)

    # 小数部分
    i_f = i_ - i_n.astype(np.float32)
    j_f = j_ - j_n.astype(np.float32)

    # 采样点周围四个真值点的仿射索引对应的offset索引
    i_1 = (i_n)
    j_1 = (j_n - ((i_n + 1) / 2).astype(int))
    i_2 = (i_n + 1)
    j_2 = (j_n - ((i_n + 2) / 2).astype(int))
    i_3 = (i_n)
    j_3 = (j_n + 1 - ((i_n + 1) / 2).astype(int))
    i_4 = (i_n + 1)
    j_4 = (j_n + 1 - ((i_n + 2) / 2).astype(int))

    # 取1就选2号点，取0就取3号点
    up_down_flag = (i_f > j_f)

    # range_flag = (i_1 >= 0 and j_1 >= 0 and i_1 < h and j_1 < w\
    #         and i_2 >= 0 and j_2 >= 0 and i_2 < h and j_2 < w\
    #         and i_4 >= 0 and j_4 >= 0 and i_4 < h and j_4 < w).float()+
    range_flag1 = (i_1 >= 0).astype(float) * (j_1 >= 0).astype(float) * \
                  (i_1 < h).astype(float) * (j_1 < w).astype(float)
    range_flag2 = (i_2 >= 0).astype(float) * (j_2 >= 0).astype(float) * \
                  (i_2 < h).astype(float) * (j_2 < w).astype(float)
    range_flag3 = (i_3 >= 0).astype(float) * (j_3 >= 0).astype(float) * \
                  (i_3 < h).astype(float) * (j_3 < w).astype(float)
    range_flag4 = (i_4 >= 0).astype(float) * (j_4 >= 0).astype(float) * \
                  (i_4 < h).astype(float) * (j_4 < w).astype(float)
    valid_indices1 = (range_flag1 == 1.)
    valid_indices2 = (range_flag2 == 1.)
    valid_indices3 = (range_flag3 == 1.)
    valid_indices4 = (range_flag4 == 1.)

    p1 = np.zeros((h1, w1, c), dtype=image.dtype)
    p2 = np.zeros((h1, w1, c), dtype=image.dtype)
    p3 = np.zeros((h1, w1, c), dtype=image.dtype)

    p1[valid_indices1] = image[i_1[valid_indices1], j_1[valid_indices1]]
    p2[valid_indices2 & up_down_flag] = image[i_2[valid_indices2 & up_down_flag], j_2[valid_indices2 & up_down_flag]]
    p2[valid_indices3 & ~up_down_flag] = image[i_3[valid_indices3 & ~up_down_flag], j_3[valid_indices3 & ~up_down_flag]]
    p3[valid_indices4] = image[i_4[valid_indices4], j_4[valid_indices4]]

    # 真值点仿射索引转笛卡尔坐标
    p1_x = i_n - ((h - 1) / 2)
    p1_y = j_n - i_n / 2 - (w - 0.5) / 2
    p2_x = (i_n + up_down_flag.astype(float)) - (h - 1) / 2
    p2_y = (j_n + 1 - up_down_flag.astype(float)) - (i_n + up_down_flag.astype(float)) / 2 - (w - 0.5) / 2
    p3_x = (i_n + 1) - (h - 1) / 2
    p3_y = (j_n + 1) - (i_n + 1) / 2 - (w - 0.5) / 2

    if method == 0:
        d1 = (x_ - p1_x) * (x_ - p1_x) + (y_ - p1_y) * (y_ - p1_y)
        d2 = (x_ - p2_x) * (x_ - p2_x) + (y_ - p2_y) * (y_ - p2_y)
        d3 = (x_ - p3_x) * (x_ - p3_x) + (y_ - p3_y) * (y_ - p3_y)

        d = np.stack((d1, d2, d3), axis=0)
        min_values, min_indices = np.min(d, axis=0)

        # 初始化 output_image
        new_image = np.zeros((h1, w1, c), dtype=image.dtype)
        new_image[min_indices == 0] = p1[min_indices == 0]  # 对应 d1 的颜色
        new_image[min_indices == 1] = p2[min_indices == 1]  # 对应 d2 的颜色
        new_image[min_indices == 2] = p3[min_indices == 2]  # 对应 d3 的颜色

    if method == 1:
        S1 = 0.5 * abs((x_ - p2_x) * (y_ - p3_y) - (y_ - p2_y) * (x_ - p3_x))
        S2 = 0.5 * abs((x_ - p1_x) * (y_ - p3_y) - (y_ - p1_y) * (x_ - p3_x))
        S3 = 0.5 * abs((x_ - p1_x) * (y_ - p2_y) - (y_ - p1_y) * (x_ - p2_x))
        alpha = np.expand_dims((S1 / (S1 + S2 + S3)), axis=-1)
        beta  = np.expand_dims((S2 / (S1 + S2 + S3)), axis=-1)
        gamma = np.expand_dims((S3 / (S1 + S2 + S3)), axis=-1)
        new_image = alpha * p1 + beta * p2 + gamma * p3

    return np.transpose(new_image, [2, 0, 1]).squeeze()

def rect_to_hex_resample(rect_image, hex_dsize=None, interpolation='nearest', offset=0):
    method_dict = {
        'nearest': 0,
        'bilinear': 1,
    }
    method = method_dict[interpolation]

    if rect_image.ndim == 3:
        c, h, w = rect_image.shape
    elif rect_image.ndim == 2:
        h, w = rect_image.shape
        c = 1
    else:
        raise Exception(f"dim of image should be 2 or 3, but got dim = {rect_image.ndim} instead")

    if hex_dsize == None:
        hex_dsize = (h, w)

    h1, w1 = hex_dsize

    image = np.array(rect_image)
    # generate central elements' coordinate array of original hexagonal grid
    # 生成原图格心
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
    # w方向中心化
    imgcoor[1] -= w / 2.


    # 方格图像上界下界
    left_top = [-(h / 2), -(w / 2 + 0.5), 1.]
    right_top = [-(h / 2), w / 2 + 0.5, 1.]
    left_bottom = [h / 2, -(w / 2 + 0.5), 1.]
    right_bottom = [h / 2, w / 2 + 0.5, 1.]
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
    # 插值的时候还是得利用仿射坐标来确定采样点在哪三个已知点之间
    x_, y_ = homogeneous_new_img_local_coor[0], homogeneous_new_img_local_coor[1]

    i_ = x_ + (h - 1) * 0.5
    j_ = y_ + (w - 1) * 0.5

    # 整数部分（采样点归属的单元）
    i_n = i_.astype(int)
    j_n = j_.astype(int)

    # 小数部分
    i_f = i_ - i_n.astype(np.float32)
    j_f = j_ - j_n.astype(np.float32)

    # 采样点周围四个真值点的仿射索引对应的offset索引
    i_1 = (i_n).astype(int)
    j_1 = (j_n).astype(int)
    i_2 = (i_n).astype(int)
    j_2 = (j_n + 1).astype(int)
    i_3 = (i_n + 1).astype(int)
    j_3 = (j_n).astype(int)
    i_4 = (i_n + 1).astype(int)
    j_4 = (j_n + 1).astype(int)


    # range_flag = (i_1 >= 0 and j_1 >= 0 and i_1 < h and j_1 < w\
    #         and i_2 >= 0 and j_2 >= 0 and i_2 < h and j_2 < w\
    #         and i_4 >= 0 and j_4 >= 0 and i_4 < h and j_4 < w).float()+
    range_flag1 = (i_1 >= 0).astype(float) * (j_1 >= 0).astype(float) * \
                  (i_1 < h).astype(float) * (j_1 < w).astype(float)
    range_flag2 = (i_2 >= 0).astype(float) * (j_2 >= 0).astype(float) * \
                  (i_2 < h).astype(float) * (j_2 < w).astype(float)
    range_flag3 = (i_3 >= 0).astype(float) * (j_3 >= 0).astype(float) * \
                  (i_3 < h).astype(float) * (j_3 < w).astype(float)
    range_flag4 = (i_4 >= 0).astype(float) * (j_4 >= 0).astype(float) * \
                  (i_4 < h).astype(float) * (j_4 < w).astype(float)
    valid_indices1 = (range_flag1 == 1.)
    valid_indices2 = (range_flag2 == 1.)
    valid_indices3 = (range_flag3 == 1.)
    valid_indices4 = (range_flag4 == 1.)

    p1 = np.zeros((h1, w1, c), dtype=image.dtype)
    p2 = np.zeros((h1, w1, c), dtype=image.dtype)
    p3 = np.zeros((h1, w1, c), dtype=image.dtype)
    p4 = np.zeros((h1, w1, c), dtype=image.dtype)

    p1[valid_indices1] = image[i_1[valid_indices1], j_1[valid_indices1]]
    p2[valid_indices2] = image[i_2[valid_indices2], j_2[valid_indices2]]
    p3[valid_indices3] = image[i_3[valid_indices3], j_3[valid_indices3]]
    p4[valid_indices4] = image[i_4[valid_indices4], j_4[valid_indices4]]

    # 真值点仿射索引转笛卡尔坐标
    p1_x = i_1
    p1_y = j_1
    p2_x = i_2
    p2_y = j_2
    p3_x = i_3
    p3_y = j_3
    p4_x = i_4
    p4_y = j_4

    if method == 0:
        d1 = (x_ - p1_x) * (x_ - p1_x) + (y_ - p1_y) * (y_ - p1_y)
        d2 = (x_ - p2_x) * (x_ - p2_x) + (y_ - p2_y) * (y_ - p2_y)
        d3 = (x_ - p3_x) * (x_ - p3_x) + (y_ - p3_y) * (y_ - p3_y)
        d4 = (x_ - p4_x) * (x_ - p4_x) + (y_ - p4_y) * (y_ - p4_y)

        d = np.stack((d1, d2, d3, d4), axis=0)
        min_values  = np.min(d, axis=0)
        min_indices = np.argmin(d, axis=0)
        # 初始化 output_image
        new_image = np.zeros((h1, w1, c), dtype=image.dtype)
        new_image[min_indices == 0] = p1[min_indices == 0] # corresponding value of d1
        new_image[min_indices == 1] = p2[min_indices == 1] # corresponding value of d2
        new_image[min_indices == 2] = p3[min_indices == 2] # corresponding value of d3
        new_image[min_indices == 3] = p4[min_indices == 3] # corresponding value of d4

    if method == 1:
        t1 = np.expand_dims(i_f, axis=-1) * p3 + np.expand_dims((1 - i_f), axis=-1) * p1
        t2 = np.expand_dims(i_f, axis=-1) * p4 + np.expand_dims((1 - i_f), axis=-1) * p2
        new_image = np.expand_dims(j_f, axis=-1) * t2 + np.expand_dims((1 - j_f), axis=-1) * t1

    return np.transpose(new_image, [2, 0, 1]).squeeze()
def hexresize(image:np.array, dsize, interpolation="linear", offset=0):
    """
    hexresize
    :param image: c x h x w
    :param dsize: new_h x new_w
    :param interpolation: 'linear' or 'nearest'
    :param offset: 0
    :return:
    """


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
        [new_img_local_coor, #2 x h x w
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

    # 插值的时候还是得利用仿射坐标来确定采样点在哪三个已知点之间
    x_, y_ = homogeneous_new_img_local_coor[0], homogeneous_new_img_local_coor[1] # (x, y, 1)—3 x h x w

    i_ = x_ + (h - 1) * 0.5
    j_ = 0.5 * i_ + y_ + (w - 0.5) * 0.5

    # 整数部分（采样点归属的单元）
    i_n = (i_).astype(int)
    j_n = (j_).astype(int)

    # 小数部分
    i_f = i_ - i_n.astype(np.float32)
    j_f = j_ - j_n.astype(np.float32)

    # 采样点周围四个真值点的仿射索引对应的offset索引
    i_1 = (i_n)
    j_1 = (j_n - ((i_n + 1) / 2).astype(int))
    i_2 = (i_n + 1)
    j_2 = (j_n - ((i_n + 2) / 2).astype(int))
    i_3 = (i_n)
    j_3 = (j_n + 1 - ((i_n + 1) / 2).astype(int))
    i_4 = (i_n + 1)
    j_4 = (j_n + 1 - ((i_n + 2) / 2).astype(int))

    # 取1就选2号点，取0就取3号点
    up_down_flag = (i_f > j_f)

    # range_flag = (i_1 >= 0 and j_1 >= 0 and i_1 < h and j_1 < w\
    #         and i_2 >= 0 and j_2 >= 0 and i_2 < h and j_2 < w\
    #         and i_4 >= 0 and j_4 >= 0 and i_4 < h and j_4 < w).float()+
    range_flag1 = (i_1 >= 0).astype(float) * (j_1 >= 0).astype(float) * \
                  (i_1 < h).astype(float) * (j_1 < w).astype(float)
    range_flag2 = (i_2 >= 0).astype(float) * (j_2 >= 0).astype(float) * \
                  (i_2 < h).astype(float) * (j_2 < w).astype(float)
    range_flag3 = (i_3 >= 0).astype(float) * (j_3 >= 0).astype(float) * \
                  (i_3 < h).astype(float) * (j_3 < w).astype(float)
    range_flag4 = (i_4 >= 0).astype(float) * (j_4 >= 0).astype(float) * \
                  (i_4 < h).astype(float) * (j_4 < w).astype(float)
    valid_indices1 = (range_flag1 == 1.)
    valid_indices2 = (range_flag2 == 1.)
    valid_indices3 = (range_flag3 == 1.)
    valid_indices4 = (range_flag4 == 1.)

    p1 = np.zeros((h1, w1, c), dtype=image.dtype)
    p2 = np.zeros((h1, w1, c), dtype=image.dtype)
    p3 = np.zeros((h1, w1, c), dtype=image.dtype)

    p1[valid_indices1] = image[i_1[valid_indices1], j_1[valid_indices1]]
    p2[valid_indices2 & up_down_flag] = image[i_2[valid_indices2 & up_down_flag], j_2[valid_indices2 & up_down_flag]]
    p2[valid_indices3 & ~up_down_flag] = image[i_3[valid_indices3 & ~up_down_flag], j_3[valid_indices3 & ~up_down_flag]]
    p3[valid_indices4] = image[i_4[valid_indices4], j_4[valid_indices4]]

    # 真值点仿射索引转笛卡尔坐标
    p1_x = i_n - ((h - 1) / 2)
    p1_y = j_n - i_n / 2 - (w - 0.5) / 2
    p2_x = (i_n + up_down_flag.astype(float)) - (h - 1) / 2
    p2_y = (j_n + 1 - up_down_flag.astype(float)) - (i_n + up_down_flag.astype(float)) / 2 - (w - 0.5) / 2
    p3_x = (i_n + 1) - (h - 1) / 2
    p3_y = (j_n + 1) - (i_n + 1) / 2 - (w - 0.5) / 2

    if interpolation == 'nearest':
        d1 = (x_ - p1_x) * (x_ - p1_x) + (y_ - p1_y) * (y_ - p1_y)
        d2 = (x_ - p2_x) * (x_ - p2_x) + (y_ - p2_y) * (y_ - p2_y)
        d3 = (x_ - p3_x) * (x_ - p3_x) + (y_ - p3_y) * (y_ - p3_y)

        d = np.stack((d1, d2, d3), axis=0)
        min_values, min_indices = np.min(d, axis=0)

        # 初始化 output_image
        new_image = np.zeros((h1, w1, c), dtype=image.dtype)
        new_image[min_indices == 0] = p1[min_indices == 0]  # 对应 d1 的颜色
        new_image[min_indices == 1] = p2[min_indices == 1]  # 对应 d2 的颜色
        new_image[min_indices == 2] = p3[min_indices == 2]  # 对应 d3 的颜色

    if interpolation == 'linear':
        S1 = 0.5 * abs((x_ - p2_x) * (y_ - p3_y) - (y_ - p2_y) * (x_ - p3_x))
        S2 = 0.5 * abs((x_ - p1_x) * (y_ - p3_y) - (y_ - p1_y) * (x_ - p3_x))
        S3 = 0.5 * abs((x_ - p1_x) * (y_ - p2_y) - (y_ - p1_y) * (x_ - p2_x))
        alpha = np.expand_dims((S1 / (S1 + S2 + S3)), axis=-1)
        beta = np.expand_dims((S2 / (S1 + S2 + S3)), axis=-1)
        gamma = np.expand_dims((S3 / (S1 + S2 + S3)), axis=-1)
        new_image = alpha * p1 + beta * p2 + gamma * p3

    return np.transpose(new_image, [2, 0, 1]).squeeze()

def heximpad(img: np.ndarray,
          *,
          shape: Optional[Tuple[int, int]] = None,
          padding: Union[int, tuple, None] = None,
          pad_val: Union[float, List] = 0,
          padding_mode: str = 'constant') -> np.ndarray:

    assert (shape is not None) ^ (padding is not None)
    if shape is not None:
        # pad at only right and bottom
        width = max(shape[1] - img.shape[1], 0)
        height = max(shape[0] - img.shape[0], 0)
        padding = (0, 0, width, height)

    # check pad_val
    if isinstance(pad_val, tuple):
        assert len(pad_val) == img.shape[-1]
    elif not isinstance(pad_val, numbers.Number):
        raise TypeError('pad_val must be a int or a tuple. '
                        f'But received {type(pad_val)}')

    # check padding
    if isinstance(padding, tuple) and len(padding) in [2, 4]:
        if len(padding) == 2:
            padding = (padding[0] - padding[0], padding[1], padding[0], padding[1])
    elif isinstance(padding, numbers.Number):
        padding = (padding, padding, padding, padding)
    else:
        raise ValueError('Padding must be a int or a 2, or 4 element tuple.'
                         f'But received {padding}')

    # check padding mode
    assert padding_mode in ['constant', 'edge', 'reflect', 'symmetric']

    border_type = {
        'constant': cv2.BORDER_CONSTANT,
        'edge': cv2.BORDER_REPLICATE,
        'reflect': cv2.BORDER_REFLECT_101,
        'symmetric': cv2.BORDER_REFLECT
    }
    img = cv2.copyMakeBorder(
        img,
        padding[1] - padding[1]%2,
        padding[3] + padding[1]%2,
        padding[0],
        padding[2],
        border_type[padding_mode],
        value=pad_val)

    return img

def hex_impad_to_multiple(img: np.ndarray,
                      divisor: int,
                      pad_val: Union[float, List] = 0) -> np.ndarray:
    """Pad an image to ensure each edge to be multiple to some number.

    Args:
        img (ndarray): Image to be padded.
        divisor (int): Padded image edges will be multiple to divisor.
        pad_val (Number | Sequence[Number]): Same as :func:`impad`.

    Returns:
        ndarray: The padded image.
    """
    pad_h = int(np.ceil(img.shape[0] / divisor)) * divisor
    pad_w = int(np.ceil(img.shape[1] / divisor)) * divisor
    return heximpad(img, shape=(pad_h, pad_w), pad_val=pad_val)
if __name__ == '__main__':
    from HexImage import HEXIMAGE
    from Image import IMAGE
    # import cv2
    # path = r"D:\mmsegmentation-0.29.1\Data\Hex_FBP\images\test\GF2_PMS1__L1A0001064454-MSS1_0_0.tif"
    # image = HEXIMAGE(path, heximagetype=2)
    # image.Hex_imshow()
    # print(image.shape)
    #
    # image1_data = hexresize(
    #     image.HexagonImage,
    #     (1024, 1024),
    #     'linear'
    # )
    # image1 = HEXIMAGE(data=image1_data)
    # image1.Hex_imshow()
    # print(image1.shape)
    path = r"D:\mmsegmentation-0.29.1\Data\ADEChallengeData2016\images\validation\ADE_val_00001232.jpg"
    # path = r"D:\mmsegmentation-0.29.1\Data\ADEChallengeData2016\images\validation\ADE_val_00001732.jpg"

    image = IMAGE(pathname=path)
    image.imshow()
    image1_data = rect_to_hex_resample(
        image.Image,
        (256, 341),
        'bilinear'
    )
    image1 = HEXIMAGE(data=image1_data)
    image1.Hex_imshow()
    print(image1.shape)
    image2 = HEXIMAGE(pathname=path, heximagetype=2)
    image2.Hex_imshow()
