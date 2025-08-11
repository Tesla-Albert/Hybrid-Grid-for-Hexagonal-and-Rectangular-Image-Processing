from scipy.interpolate import griddata, interp2d
import numpy as np
import torch

import warnings
warnings.filterwarnings("ignore")
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
    image = torch.tensor(image, device='cuda')
    H = torch.tensor(H).to(torch.float64)
    # 生成原图格心
    imgcoor = torch.stack(
        torch.meshgrid(
            torch.arange(h, device=image.device),
            torch.arange(w, device=image.device)
        ),
        dim=0
    ).to(torch.double)# (x(h, w), y(h, w)), dtype = torch.float32
    image = image.permute(1, 2, 0)# c, h, w -> h, w, c
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
    imgcorner = torch.tensor([
        left_top, right_top, left_bottom, right_bottom
    ], dtype=torch.double).T
    new_homogeneous_imgcorner = torch.matmul(H, imgcorner).to('cuda')

    h1_inf = torch.min(new_homogeneous_imgcorner[0]).item()
    w1_inf = torch.min(new_homogeneous_imgcorner[1]).item()
    h1_sup = torch.max(new_homogeneous_imgcorner[0]).item()
    w1_sup = torch.max(new_homogeneous_imgcorner[1]).item()

    h1_f = (h1_sup - h1_inf)
    w1_f = (w1_sup - w1_inf)


    # 新图坐标系的定义
    # 生成新图的格心矩阵(由于上下界由原图格心变换得到，所以新图格心的h坐标直接从下界开始，不加0.5
    # 而w坐标，从下界开始到上界-0.5(第1行，输出统一为offset=0，所以缩进0.5）+1. =到sup+0.5
    new_img_local_coor = torch.stack(
        torch.meshgrid(
        torch.arange(h1_inf, h1_sup+1, 1).double(),
        torch.arange(w1_inf, w1_sup+0.5, 1).double()
        ),
        dim=0
    ).to(torch.double)
    new_img_local_coor[1][1::2] += 0.5

    homogeneous_new_img_local_coor \
        = torch.vstack(
        [new_img_local_coor,
         torch.ones((1, new_img_local_coor.shape[1], new_img_local_coor.shape[2]))]
    )
    h1 = new_img_local_coor.shape[1]
    w1 = new_img_local_coor.shape[2]
    # 反变换格心
    new_homogeneous_imgcoor_inverse \
        = torch.einsum(
        'ij, jkl -> ikl',
        torch.linalg.inv(H),
        homogeneous_new_img_local_coor.to(torch.double)
    ).to(torch.float)
    new_image = torch.empty(
        (c,
         homogeneous_new_img_local_coor.shape[1],
         homogeneous_new_img_local_coor.shape[2]),
        device=image.device
    )
    # 插值的时候还是得利用仿射坐标来确定采样点在哪三个已知点之间
    x_, y_ = new_homogeneous_imgcoor_inverse[0], new_homogeneous_imgcoor_inverse[1]

    i_ = x_ + (h - 1) * 0.5
    j_ = 0.5 * i_ + y_ + (w - 0.5) * 0.5

    # 整数部分（采样点归属的单元）
    i_n = i_.to(int)
    j_n = j_.to(int)

    # 小数部分
    i_f = i_ - i_n.to(torch.float32)
    j_f = j_ - j_n.to(torch.float32)

    # 采样点周围四个真值点的仿射索引对应的offset索引
    i_1 = (i_n).to('cuda')
    j_1 = (j_n - ((i_n + 1) / 2).to(int)).to('cuda')
    i_2 = (i_n + 1).to('cuda')
    j_2 = (j_n - ((i_n + 2) / 2).to(int)).to('cuda')
    i_3 = (i_n).to('cuda')
    j_3 = (j_n + 1 - ((i_n + 1) / 2).to(int)).to('cuda')
    i_4 = (i_n + 1).to('cuda')
    j_4 = (j_n + 1 - ((i_n + 2) / 2).to(int)).to('cuda')

    # 取1就选2号点，取0就取3号点
    up_down_flag = (i_f > j_f).to('cuda')

    # range_flag = (i_1 >= 0 and j_1 >= 0 and i_1 < h and j_1 < w\
    #         and i_2 >= 0 and j_2 >= 0 and i_2 < h and j_2 < w\
    #         and i_4 >= 0 and j_4 >= 0 and i_4 < h and j_4 < w).float()+
    range_flag1 = (i_1 >= 0).float() * (j_1 >= 0).float() * \
                  (i_1 < h).float() * (j_1 < w).float()
    range_flag2 = (i_2 >= 0).float() * (j_2 >= 0).float() * \
                  (i_2 < h).float() * (j_2 < w).float()
    range_flag3 = (i_3 >= 0).float() * (j_3 >= 0).float() * \
                  (i_3 < h).float() * (j_3 < w).float()
    range_flag4 = (i_4 >= 0).float() * (j_4 >= 0).float() * \
                  (i_4 < h).float() * (j_4 < w).float()
    valid_indices1 = (range_flag1 == 1.).to('cuda')
    valid_indices2 = (range_flag2 == 1.).to('cuda')
    valid_indices3 = (range_flag3 == 1.).to('cuda')
    valid_indices4 = (range_flag4 == 1.).to('cuda')

    p1 = torch.zeros((h1, w1, c), dtype=image.dtype, device=image.device)
    p2 = torch.zeros((h1, w1, c), dtype=image.dtype, device=image.device)
    p3 = torch.zeros((h1, w1, c), dtype=image.dtype, device=image.device)

    p1[valid_indices1] = image[i_1[valid_indices1], j_1[valid_indices1]]
    p2[valid_indices2 & up_down_flag] = image[i_2[valid_indices2 & up_down_flag], j_2[valid_indices2 & up_down_flag]]
    p2[valid_indices3 & ~up_down_flag] = image[i_3[valid_indices3 & ~up_down_flag], j_3[valid_indices3 & ~up_down_flag]]
    p3[valid_indices4] = image[i_4[valid_indices4], j_4[valid_indices4]]

    # 真值点仿射索引转笛卡尔坐标
    p1_x = i_n - ((h - 1) / 2)
    p1_y = j_n - i_n / 2 - (w - 0.5) / 2
    p2_x = (i_n + up_down_flag.float().cpu()) - (h - 1) / 2
    p2_y = (j_n + 1 - up_down_flag.float().cpu()) - (i_n + up_down_flag.float().cpu()) / 2 - (w - 0.5) / 2
    p3_x = (i_n + 1) - (h - 1) / 2
    p3_y = (j_n + 1) - (i_n + 1) / 2 - (w - 0.5) / 2

    if method == 0:
        d1 = (x_ - p1_x) * (x_ - p1_x) + (y_ - p1_y) * (y_ - p1_y)
        d2 = (x_ - p2_x) * (x_ - p2_x) + (y_ - p2_y) * (y_ - p2_y)
        d3 = (x_ - p3_x) * (x_ - p3_x) + (y_ - p3_y) * (y_ - p3_y)

        d = torch.stack((d1, d2, d3), dim=0)
        min_values, min_indices = torch.min(d, dim=0)

        # 初始化 output_image
        new_image = torch.zeros((h1, w1, c), dtype=image.dtype, device=image.device)
        new_image[min_indices == 0] = p1[min_indices == 0]  # 对应 d1 的颜色
        new_image[min_indices == 1] = p2[min_indices == 1]  # 对应 d2 的颜色
        new_image[min_indices == 2] = p3[min_indices == 2]  # 对应 d3 的颜色

    if method == 1:
        S1 = 0.5 * abs((x_ - p2_x) * (y_ - p3_y) - (y_ - p2_y) * (x_ - p3_x))
        S2 = 0.5 * abs((x_ - p1_x) * (y_ - p3_y) - (y_ - p1_y) * (x_ - p3_x))
        S3 = 0.5 * abs((x_ - p1_x) * (y_ - p2_y) - (y_ - p1_y) * (x_ - p2_x))
        alpha = (S1 / (S1 + S2 + S3)).unsqueeze(-1).to('cuda')
        beta = (S2 / (S1 + S2 + S3)).unsqueeze(-1).to('cuda')
        gamma = (S3 / (S1 + S2 + S3)).unsqueeze(-1).to('cuda')
        new_image = alpha * p1 + beta * p2 + gamma * p3

    return new_image.permute(2, 0, 1).squeeze().cpu().numpy()

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
    else:
        raise Exception(f"dim of image should be 2 or 3, but got dim = {hex_image.ndim} instead")

    if square_size == None:
        square_size = (h, w)

    h1, w1 = square_size

    image = torch.tensor(hex_image, device='cuda')
    # generate central elements' coordinate array of original hexagonal grid
    # 生成原图格心
    imgcoor = torch.stack(
        torch.meshgrid(
            torch.arange(h, device=image.device),
            torch.arange(w, device=image.device)
        ),
        dim=0
    ).to(torch.double)  # (x(h, w), y(h, w)), dtype = torch.float32
    image = image.permute(1, 2, 0)  # c, h, w -> h, w, c
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
    imgcorner = torch.tensor([
        left_top, right_top, left_bottom, right_bottom
    ], dtype=torch.double).T


    h1_inf = torch.min(imgcorner[0]).item()
    w1_inf = torch.min(imgcorner[1]).item()
    h1_sup = torch.max(imgcorner[0]).item()
    w1_sup = torch.max(imgcorner[1]).item()

    new_img_local_coor = torch.stack(
        torch.meshgrid(
        torch.linspace(h1_inf, h1_sup, h1).double(),
        torch.linspace(w1_inf, w1_sup, w1).double()
        ),
        dim=0
    ).to(torch.double)
    # new_img_local_coor = np.array(
    #     np.meshgrid(np.linspace(h_1_inf, h_1_sup, h1), np.linspace(w_1_inf, w_1_sup, w1), indexing='ij')
    # )
    homogeneous_new_img_local_coor \
        = torch.vstack(
        [new_img_local_coor,
         torch.ones((1, new_img_local_coor.size(1), new_img_local_coor.size(2)))]
    )
    h1 = new_img_local_coor.size(1)
    w1 = new_img_local_coor.size(2)
    # 反变换格心

    new_image = torch.empty(
        (c,
         homogeneous_new_img_local_coor.shape[1],
         homogeneous_new_img_local_coor.shape[2]),
        device=image.device
    )
    # 插值的时候还是得利用仿射坐标来确定采样点在哪三个已知点之间
    x_, y_ = homogeneous_new_img_local_coor[0], homogeneous_new_img_local_coor[1]

    i_ = x_ + (h - 1) * 0.5
    j_ = 0.5 * i_ + y_ + (w - 0.5) * 0.5

    # 整数部分（采样点归属的单元）
    i_n = i_.to(int)
    j_n = j_.to(int)

    # 小数部分
    i_f = i_ - i_n.to(torch.float32)
    j_f = j_ - j_n.to(torch.float32)

    # 采样点周围四个真值点的仿射索引对应的offset索引
    i_1 = (i_n).to('cuda')
    j_1 = (j_n - ((i_n + 1) / 2).to(int)).to('cuda')
    i_2 = (i_n + 1).to('cuda')
    j_2 = (j_n - ((i_n + 2) / 2).to(int)).to('cuda')
    i_3 = (i_n).to('cuda')
    j_3 = (j_n + 1 - ((i_n + 1) / 2).to(int)).to('cuda')
    i_4 = (i_n + 1).to('cuda')
    j_4 = (j_n + 1 - ((i_n + 2) / 2).to(int)).to('cuda')

    # 取1就选2号点，取0就取3号点
    up_down_flag = (i_f > j_f).to('cuda')

    # range_flag = (i_1 >= 0 and j_1 >= 0 and i_1 < h and j_1 < w\
    #         and i_2 >= 0 and j_2 >= 0 and i_2 < h and j_2 < w\
    #         and i_4 >= 0 and j_4 >= 0 and i_4 < h and j_4 < w).float()+
    range_flag1 = (i_1 >= 0).float() * (j_1 >= 0).float() * \
                  (i_1 < h).float() * (j_1 < w).float()
    range_flag2 = (i_2 >= 0).float() * (j_2 >= 0).float() * \
                  (i_2 < h).float() * (j_2 < w).float()
    range_flag3 = (i_3 >= 0).float() * (j_3 >= 0).float() * \
                  (i_3 < h).float() * (j_3 < w).float()
    range_flag4 = (i_4 >= 0).float() * (j_4 >= 0).float() * \
                  (i_4 < h).float() * (j_4 < w).float()
    valid_indices1 = (range_flag1 == 1.).to('cuda')
    valid_indices2 = (range_flag2 == 1.).to('cuda')
    valid_indices3 = (range_flag3 == 1.).to('cuda')
    valid_indices4 = (range_flag4 == 1.).to('cuda')

    p1 = torch.zeros((h1, w1, c), dtype=image.dtype, device=image.device)
    p2 = torch.zeros((h1, w1, c), dtype=image.dtype, device=image.device)
    p3 = torch.zeros((h1, w1, c), dtype=image.dtype, device=image.device)

    p1[valid_indices1] = image[i_1[valid_indices1], j_1[valid_indices1]]
    p2[valid_indices2 & up_down_flag] = image[i_2[valid_indices2 & up_down_flag], j_2[valid_indices2 & up_down_flag]]
    p2[valid_indices3 & ~up_down_flag] = image[i_3[valid_indices3 & ~up_down_flag], j_3[valid_indices3 & ~up_down_flag]]
    p3[valid_indices4] = image[i_4[valid_indices4], j_4[valid_indices4]]

    # 真值点仿射索引转笛卡尔坐标
    p1_x = i_n - ((h - 1) / 2)
    p1_y = j_n - i_n / 2 - (w - 0.5) / 2
    p2_x = (i_n + up_down_flag.float().cpu()) - (h - 1) / 2
    p2_y = (j_n + 1 - up_down_flag.float().cpu()) - (i_n + up_down_flag.float().cpu()) / 2 - (w - 0.5) / 2
    p3_x = (i_n + 1) - (h - 1) / 2
    p3_y = (j_n + 1) - (i_n + 1) / 2 - (w - 0.5) / 2

    if method == 0:
        d1 = (x_ - p1_x) * (x_ - p1_x) + (y_ - p1_y) * (y_ - p1_y)
        d2 = (x_ - p2_x) * (x_ - p2_x) + (y_ - p2_y) * (y_ - p2_y)
        d3 = (x_ - p3_x) * (x_ - p3_x) + (y_ - p3_y) * (y_ - p3_y)

        d = torch.stack((d1, d2, d3), dim=0)
        min_values, min_indices = torch.min(d, dim=0)

        # 初始化 output_image
        new_image = torch.zeros((h1, w1, c), dtype=image.dtype, device=image.device)
        new_image[min_indices == 0] = p1[min_indices == 0]  # 对应 d1 的颜色
        new_image[min_indices == 1] = p2[min_indices == 1]  # 对应 d2 的颜色
        new_image[min_indices == 2] = p3[min_indices == 2]  # 对应 d3 的颜色

    if method == 1:
        S1 = 0.5 * abs((x_ - p2_x) * (y_ - p3_y) - (y_ - p2_y) * (x_ - p3_x))
        S2 = 0.5 * abs((x_ - p1_x) * (y_ - p3_y) - (y_ - p1_y) * (x_ - p3_x))
        S3 = 0.5 * abs((x_ - p1_x) * (y_ - p2_y) - (y_ - p1_y) * (x_ - p2_x))
        alpha = (S1 / (S1 + S2 + S3)).unsqueeze(-1).to('cuda')
        beta = (S2 / (S1 + S2 + S3)).unsqueeze(-1).to('cuda')
        gamma = (S3 / (S1 + S2 + S3)).unsqueeze(-1).to('cuda')
        new_image = alpha * p1 + beta * p2 + gamma * p3

    return new_image.permute(2, 0, 1).squeeze().cpu().numpy()

def image_geometric_transformation_cpu(img:np.array, H:np.array=np.eye(3), interpolation='nearest', offset=0)->np.array:
    # H = np.array([
    #     [h_scale, 0, 0],
    #     [0, w_scale, 0],
    #     [0, 0, 1]
    # ])
    c, h, w = image.shape
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
    newimg = np.empty((img.shape[0], h_1, w_1))

    for i in range(img.shape[0]):
        newimg[i] = griddata((x, y),
                             img[i].flatten(),
                             (xi, yi),
                             method=interpolation).reshape(h_1, w_1)
    return newimg

def image_geometric_transformation(img:np.array, H:np.array=np.eye(3), interpolation='nearest', offset=0, device='cuda0')->np.array:
    if device=='cuda0':
        return image_geometric_transformation_gpu(img, H, interpolation, offset)
    if device=='cpu':
        return image_geometric_transformation_cpu(img, H, interpolation, offset)


# def hex_to_square_resample(hex_image, square_size=None, interpolation='nearest', offset=0):
#     method_dict = {
#         'nearest': 0,
#         'linear': 1,
#         'bilinear': 2,
#     }
#     method = method_dict[interpolation]
#     if hex_image.ndim == 3:
#         c, h, w = hex_image.shape
#     elif hex_image.ndim == 2:
#         h, w = hex_image.shape
#         c = 1
#         hex_image = np.expand_dims(hex_image, axis=0)
#     else:
#         raise Exception(f"dim of image should be 2 or 3, but got dim = {hex_image.ndim} instead")
#
#     if square_size == None:
#         square_size = (h, w)
#     h1, w1 = square_size
#     # generate central elements' coordinate array of original hexagonal grid
#     imgcoor = np.mgrid[0:h:1., 0:w:1.]
#     # h方向
#     imgcoor[0] += 0.5
#     # h方向中心化
#     imgcoor[0] -= h / 2.
#     # w方向
#     imgcoor[1] += 0.5
#     imgcoor[1][(1 - offset)::2] += 0.5
#     # w方向中心化
#     imgcoor[1] -= (w + 0.5) / 2.
#     # 转齐次坐标
#     homogeneous_imgcoor = np.vstack([imgcoor, np.ones((1, imgcoor.shape[1], imgcoor.shape[2]))])
#     # 方格图像上界下界
#     left_top = [-(h / 2 - 0.5), -((w + 0.5) / 2 - 0.75), 1.]
#     right_top = [-(h / 2 - 0.5), (w + 0.5) / 2 - 0.75, 1.]
#     left_bottom = [h / 2 - 0.5, -((w + 0.5) / 2 - 0.75), 1.]
#     right_bottom = [h / 2 - 0.5, (w + 0.5) / 2 - 0.75, 1.]
#     # 图像角点坐标矩阵（把四个角点装载为一个矩阵）
#     imgcorner = np.array([
#         left_top, right_top, left_bottom, right_bottom
#     ], dtype=np.double).T
#
#     h_1_inf = np.min(imgcorner[0])
#     w_1_inf = np.min(imgcorner[1])
#     h_1_sup = np.max(imgcorner[0])
#     w_1_sup = np.max(imgcorner[1])
#
#     new_img_local_coor = np.array(
#         np.meshgrid(np.linspace(h_1_inf, h_1_sup, h1), np.linspace(w_1_inf, w_1_sup, w1), indexing='ij')
#     )
#     homogeneous_new_img_local_coor \
#         = np.vstack(
#         [new_img_local_coor,
#          np.ones((1, new_img_local_coor.shape[1], new_img_local_coor.shape[2]))]
#     )
#     h_1 = new_img_local_coor.shape[1]
#     w_1 = new_img_local_coor.shape[2]
#     # 反变换格心
#     # new_homogeneous_imgcoor_inverse \
#     #     = np.einsum(
#     #     'ij, jkl -> ikl',
#     #     np.linalg.inv(H),
#     #     homogeneous_new_img_local_coor
#     # )
#     new_image = np.empty((c, homogeneous_new_img_local_coor.shape[1], homogeneous_new_img_local_coor.shape[2]))
#     for i in range(c):
#         image_array = np.stack((imgcoor[0], imgcoor[1], hex_image[i]))
#         sample_coords = homogeneous_new_img_local_coor[0:2]  # 2 * h1 * w1
#         new_image_tmp = np.zeros(homogeneous_new_img_local_coor.shape[1:3])
#         torch.cuda.empty_cache()
#         cu_image_array = cuda.to_device(image_array)
#         cu_sample_coords = cuda.to_device(sample_coords)
#         cu_new_image = cuda.to_device(new_image_tmp)
#         threadsperblock = (32, 32)
#         blockspergrid_y = int(math.ceil(h_1 / threadsperblock[0]))
#         blockspergrid_x = int(math.ceil(w_1 / threadsperblock[1]))
#         blockspergrid = (blockspergrid_x, blockspergrid_y)
#         resample_on_hexagonal_grids[blockspergrid, threadsperblock](cu_sample_coords, cu_image_array, cu_new_image,
#                                                                     method)
#         new_image_tmp = cu_new_image.copy_to_host()
#         new_image[i] = new_image_tmp
#     return new_image
#
#
#

if __name__ == '__main__':
    from HexImage import HEXIMAGE
    from Image import IMAGE
    import mmcv
    # path = r"D:\mmsegmentation-0.29.1\Data\Hex_Potsdam\labels\test\1081_hex.tif"
    # path = r"D:\mmsegmentation-0.29.1\Data\Hex_Potsdam\images\test\52.jpg"
    path = r"D:\mmsegmentation-0.29.1\Data\Hex_FBP\images\test\GF2_PMS1__L1A0001064454-MSS1_0_0.tif"
    image = HEXIMAGE(path, heximagetype=2)
    print(image.shape)
    # image.Hex_imshow()

    # H = np.array([
    #     [np.cos(np.pi/6), -np.sin(np.pi/6), 0],
    #     [np.sin(np.pi/6), np.cos(np.pi/6), 0],
    #     [0, 0, 1]
    # ])
    h_scale = (5120 - 1) / (image.shape[1] - 1)
    w_scale = (5120 - 0.5) / (image.shape[2] - 0.5)
    H = np.array([
            [h_scale, 0, 0],
            [0, w_scale, 0],
            [0, 0, 1]
        ])

    # image1_Hexdata = image_geometric_transformation_gpu(image.HexagonImage, H, interpolation='linear')
    # image1_Hexdata = mmcv.imresize(
    #     image.HexagonImage.transpose(1, 2, 0),
    #     size=(600, 600),
    #     interpolation='bilinear',
    # ).transpose(2, 0, 1)
    # image1 = HEXIMAGE(data=image1_Hexdata)
    # image1.Hex_imshow()
    # print(image1.shape)
    image1_data = hex_to_square_resample(
        image.HexagonImage,
        (1024, 1024),
        'linear'
    )
    image1 = IMAGE(data=image1_data)
    image1.imshow()
    print(image1.shape)
