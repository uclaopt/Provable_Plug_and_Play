import math
import torch
import torch.nn as nn
import numpy as np
from skimage.measure import compare_psnr, compare_ssim
import torch.nn.parallel
from torch.nn.functional import normalize
import torch.optim
import torch.utils.data
from torch.autograd import Variable
import datetime
import os
from scipy.linalg import svd
from numpy import zeros
from scipy.sparse import lil_matrix

def weights_init_kaiming(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        nn.init.kaiming_normal(m.weight.data, a=0, mode='fan_in')
    elif classname.find('Linear') != -1:
        nn.init.kaiming_normal(m.weight.data, a=0, mode='fan_in')
    elif classname.find('BatchNorm') != -1:
        # nn.init.uniform(m.weight.data, 1.0, 0.02)
        m.weight.data.normal_(mean=0, std=math.sqrt(2./9./64.)).clamp_(-0.025,0.025)
        nn.init.constant(m.bias.data, 0.0)

def batch_PSNR(img, imclean, data_range):
    Img = img.data.cpu().numpy().astype(np.float32)
    Iclean = imclean.data.cpu().numpy().astype(np.float32)
    PSNR = 0
    for i in range(Img.shape[0]):
        PSNR += compare_psnr(Iclean[i,:,:,:], Img[i,:,:,:], data_range=data_range)
    return (PSNR/Img.shape[0])

def batch_SSIM(img, imclean, data_range):
    img = torch.transpose(img, 1, 3)
    imclean = torch.transpose(imclean, 1, 3)
    Img = img.data.cpu().numpy().astype(np.float32)
    Iclean = imclean.data.cpu().numpy().astype(np.float32)
    SSIM = 0
    for i in range(Img.shape[0]):
        SSIM += compare_ssim(Iclean[i,:,:,:], Img[i,:,:,:], data_range=data_range, multichannel=True)
    return (SSIM/Img.shape[0])

def data_augmentation(image, mode):
    out = np.transpose(image, (1,2,0))
    if mode == 0:
        # original
        out = out
    elif mode == 1:
        # flip up and down
        out = np.flipud(out)
    elif mode == 2:
        # rotate counterwise 90 degree
        out = np.rot90(out)
    elif mode == 3:
        # rotate 90 degree and flip up and down
        out = np.rot90(out)
        out = np.flipud(out)
    elif mode == 4:
        # rotate 180 degree
        out = np.rot90(out, k=2)
    elif mode == 5:
        # rotate 180 degree and flip
        out = np.rot90(out, k=2)
        out = np.flipud(out)
    elif mode == 6:
        # rotate 270 degree
        out = np.rot90(out, k=3)
    elif mode == 7:
        # rotate 270 degree and flip
        out = np.rot90(out, k=3)
        out = np.flipud(out)
    return np.transpose(out, (2,0,1))

# def l2_reg_ortho(mdl):
#     l2_reg = None
#     for W in mdl.parameters():
#         if W.ndimension() < 2:
#             continue
#         else:
#             cols = W[0].numel()
#             rows = W.shape[0]
#             w1 = W.view(-1, cols)
#             wt = torch.transpose(w1, 0, 1)
#             if (rows > cols):
#                 m = torch.matmul(wt, w1)
#                 ident = Variable(torch.eye(cols, cols), requires_grad=True)
#             else:
#                 m = torch.matmul(w1, wt)
#                 ident = Variable(torch.eye(rows, rows), requires_grad=True)
#
#             ident = ident.cuda()
#             w_tmp = (m - ident)
#             b_k = Variable(torch.rand(w_tmp.shape[1], 1))
#             b_k = b_k.cuda()
#
#             v1 = torch.matmul(w_tmp, b_k)
#             norm1 = torch.norm(v1, 2)
#             v2 = torch.div(v1, norm1)
#             v3 = torch.matmul(w_tmp, v2)
#
#             if l2_reg is None:
#                 l2_reg = (torch.norm(v3, 2)) ** 2
#             else:
#                 l2_reg = l2_reg + (torch.norm(v3, 2)) ** 2
#     return l2_reg

def l2_reg_normal_ortho(mdl):
    l2_reg = None
    for W in mdl.parameters():
        if W.ndimension() < 2:
            continue
        else:
            cols = W[0].numel()
            rows = W.shape[0]
            w1 = W.view(-1, cols)
            wt = torch.transpose(w1, 0, 1)
            m = torch.matmul(wt, w1)
            ident = Variable(torch.eye(cols, cols))
            ident = ident.cuda()

            w_tmp = (m - ident)
            height = w_tmp.size(0)
            u = normalize(w_tmp.new_empty(height).normal_(0, 1), dim=0, eps=1e-12)
            v = normalize(torch.matmul(w_tmp.t(), u), dim=0, eps=1e-12)
            u = normalize(torch.matmul(w_tmp, v), dim=0, eps=1e-12)
            sigma = torch.dot(u, torch.matmul(w_tmp, v))

            if l2_reg is None:
                l2_reg = (sigma) ** 2
            else:
                l2_reg = l2_reg + (sigma) ** 2
    # print(l2_reg)
    return l2_reg


def adjust_ortho_decay_rate(epoch, lamb_decay):
    o_d = lamb_decay

    if epoch > 40:
       o_d = 0.0
    elif epoch > 30:
       o_d = 1e-6 * o_d
    elif epoch > 20:
       o_d = 1e-4 * o_d
    elif epoch > 10:
       o_d = 1e-3 * o_d
    return o_d


def creat_SavePath(checkpoint, sigma, layer_num, mode,
                   LipConst = False, DecayOrth = False, Lambda_0 = 0.01, SN = False, RealSN = False, lip = 1):
    today = datetime.date.today()
    date_path = os.path.join(checkpoint, str(today))
    folder_name = 'DnCNN_Sigma{}_{}Layers_mode{}'.format(sigma, layer_num, mode)
    if LipConst:
        # bool1_name = [k for k, v in locals().items() if v is bool1][0]
        folder_name = folder_name + '_LipConst'
    if DecayOrth:
        # bool2_name = [k for k, v in locals().items() if v is bool1][0]
        folder_name = folder_name + '_DecayOrth_' + str(Lambda_0)

    if SN:
        folder_name = folder_name + '_SpecNorm_lip'  + str(lip)
    if RealSN:
        folder_name = folder_name + '_RealSpecNorm_lip' + str(lip)

    save_path = os.path.join(date_path, folder_name)

    print('Save To :', save_path)
    if not os.path.exists(date_path):
        os.mkdir(date_path)
    if not os.path.exists(save_path):
        os.mkdir(save_path)
    return save_path

# def creat_SavePath(checkpoint, sigma, layer_num, mode, lip=None, no_bn=False,
#                    adaptive=False):
#     today = datetime.date.today()
#     date_path = os.path.join(checkpoint, str(today))
#     folder_name = 'DnCNN_Sigma{}_{}Layers_mode{}'.format(sigma, layer_num, mode)

#     if lip > 0.0:
#         folder_name += "_full_sn_lip{}".format(lip)

#     if no_bn:
#         folder_name += "_no_bn"

#     if adaptive:
#         folder_name += "_adaptive"

#     save_path = os.path.join(date_path, folder_name)

#     print('Save trainied model to folder:', save_path)
#     if not os.path.exists(date_path):
#         os.mkdir(date_path)
#     if not os.path.exists(save_path):
#         os.mkdir(save_path)
#     return save_path


def unroll_kernel(kernel, n):  # n = input size, 4D kernal
    channel = kernel.shape[0]
    m = kernel.shape[2]  # [m for m in model.module.dncnn.children()][0].weight.shap
    unrolled_K = zeros((channel * ((n - m + 1) ** 2), n ** 2))
    # skipped = 0
    for c in range(channel):
        cth_kernel = kernel[c][0]
        skipped = 0
        for i in range(n ** 2):
            if (i % n) < n - m + 1 and ((i / n) % n) < n - m + 1:
                for j in range(m):
                    for l in range(m):
                        unrolled_K[(c * (n - m+1)**2) + i - skipped, i + j * n + l] = cth_kernel[j, l]
            else:
                skipped += 1
    return unrolled_K


def unroll_kernel_sparse(kernel, n, sparse=True):  # n = input size, 4D kernal
    channel = kernel.shape[0]
    print(kernel.shape)
    m = kernel.shape[2]  # [m for m in model.module.dncnn.children()][0].weight.shape
    if sparse:
        unrolled_K = lil_matrix((channel * ((n - m + 1) ** 2), n ** 2))
    else:
        unrolled_K = zeros((channel * ((n - m + 1) ** 2), n ** 2))
    # skipped = 0
    for c in range(channel):
        cth_kernel = kernel[c][0]
        skipped = 0
        for i in range(n ** 2):
            if (i % n) < n - m + 1 and ((i / n) % n) < n - m + 1:
                for j in range(m):
                    for l in range(m):
                        unrolled_K[(c * (n - m+1)**2) + i - skipped, i + j * n + l] = cth_kernel[j, l]
            else:
                skipped += 1
    return unrolled_K


def unroll_kernel_chen(kernel, n, sparse=True):  # n = input size, 4D kernal
    cout, cin = kernel.shape[:2]
    print(kernel.shape)
    m = kernel.shape[2]  # [m for m in model.module.dncnn.children()][0].weight.shape
    if sparse:
        unrolled_K = lil_matrix((cout * n**2, cin * (n+2)**2))
    else:
        unrolled_K = zeros((cout * n**2, cin * (n+2)**2))

    for c_in in range(cin):
        for c_out in range(cout):
            cur_kernel = kernel[c_out][c_in]

            for i in range(n ** 2):
               for j in range(m ** 2):
                   row = i // n + j // m
                   col = i % n  + j % m
                   if row == 0 or row == n+1 or col == 0 or col == n+1:
                       continue
                   pos = row * (n + 2) + col
                   unrolled_K[c_out * n**2 + i, c_in * (n+2)**2 + pos] = cur_kernel[j // m, j % m]

    return unrolled_K
