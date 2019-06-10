"""
    Plug and Play FBS for Compressive Sensing MRI
    Authors: Jialin Liu (danny19921123@gmail.com)
"""

import os
import numpy as np
from PIL import Image
import torch
import cv2
import argparse
import glob
import scipy.io as sio
import scipy.misc
from utils.utils import load_model
from utils.utils import psnr
from utils.config import analyze_parse

def pnp_fbs_csmri(denoise_func, im_orig, mask, noises, **opts):

    alpha    = opts.get('alpha', 0.4)
    maxitr = opts.get('maxitr', 100)
    verbose = opts.get('verbose',1)
    sigma = opts.get('sigma', 5)

    """ Initialization. """

    m, n = im_orig.shape
    index = np.nonzero(mask)

    y = np.fft.fft2(im_orig) * mask + noises # observed value
    x_init = np.fft.ifft2(y) # zero fill

    print(psnr(x_init,im_orig))

    x = np.copy(x_init)

    """ Main loop. """
    for i in range(maxitr):

        xold = np.copy(x)

        """ Update variables. """

        res = np.fft.fft2(x) * mask
        index = np.nonzero(mask)
        res[index] = res[index] - y[index]
        x = x - alpha * np.fft.ifft2(res)
        # x = np.real( x )
        x = np.absolute( x )

        """ Monitoring. """

        # psnr
        if verbose:
            print("i: {}, \t psnr: {}"\
                  .format(i+1, psnr(x,im_orig)))

        xout = np.copy(x)


        """ Denoising step. """

        xtilde = np.copy(x)
        mintmp = np.min(xtilde)
        maxtmp = np.max(xtilde)
        xtilde = (xtilde - mintmp) / (maxtmp - mintmp)
        
        # the reason for the following scaling:
        # our denoisers are trained with "normalized images + noise"
        # so the scale should be 1 + O(sigma)
        scale_range = 1.0 + sigma/255.0/2.0
        scale_shift = (1 - scale_range) / 2.0
        xtilde = xtilde * scale_range + scale_shift

        # pytorch denoising model
        xtilde_torch = np.reshape(xtilde, (1,1,m,n))
        xtilde_torch = torch.from_numpy(xtilde_torch).type(torch.FloatTensor).cuda()
        r = denoise_func(xtilde_torch).cpu().numpy()
        r = np.reshape(r, (m,n))
        x = xtilde - r

        # scale and shift the denoised image back
        x = (x - scale_shift) / scale_range
        x = x * (maxtmp - mintmp) + mintmp


    return xout

# ---- input arguments ----
opt = analyze_parse(5, 0.4, 100) # the arguments are default sigma, default alpha and default max iteration.

# ---- load the model ---- 
model = load_model(opt.model_type, opt.sigma)

with torch.no_grad():

    # ---- load the ground truth ----
    im_orig = cv2.imread('Demo_mat/CS_MRI/Brain.jpg', 0)/255.0

    # ---- load mask matrix ----
    mat = sio.loadmat('Demo_mat/CS_MRI/Q_Random30.mat')
    mask = mat.get('Q1').astype(np.float64)

    # ---- load noises -----
    noises = sio.loadmat('Demo_mat/CS_MRI/noises.mat')
    noises = noises.get('noises').astype(np.complex128) * 3.0

    # ---- set options -----
    opts = dict(sigma = opt.sigma, alpha=opt.alpha, maxitr=opt.maxitr, verbose=opt.verbose) 

    # ---- plug and play !!! -----
    out = pnp_fbs_csmri(model, im_orig, mask, noises, **opts)

    # ---- print result ----- 
    print('Plug-and-Play PNSR: ', psnr(out,im_orig))



