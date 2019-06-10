"""
    Plug and Play ADMM for Poisson denoising
    Authors: XXX
             XXX
             Jialin Liu (danny19921123@gmail.com)
"""

import os
import numpy as np
from PIL import Image
import torch
import cv2
import glob
from skimage.measure import compare_psnr
from numpy import linalg as LA
import scipy.io as sio
from utils.utils import load_model
from utils.utils import psnr
from utils.config import analyze_parse

def pnp_admm_poisson_denoising(noisy, denoiser, clean, **opts):
    """
    Parameters:
        :noisy - the noisy observation.
        :denoiser - the Gaussian denoiser used in Plug-and-Play ADMM.
        :clean - the clean image used to monitor PSNR.
        :opts - the kwargs for hyperparameters in Plug-and-Play ADMM.
            :lam - the value of 1/alpha.
            :beta - the prior weight parameter.
            :maxitr - the max number of iterations.
            :verbose - a flag that enables/disables info printing.
            :peak - the peak value of the original clean image before scaling.
            :maxval - the max pixel value of the original clean image.
                - NOTE: if `peak` and `M` options exist in `opts`, then the `clean` image is the scaled version of the original image.
    """

    """ Process parameters. """

    lam    = opts.get('lam', 10.0)
    beta   = opts.get('beta', 1.0)
    maxitr = opts.get('maxitr', 50)
    peak   = opts.get('peak', None)
    maxval = opts.get('maxval', None)
    data_range = opts.get('data_range', 1.0)
    verbose = opts.get('verbose', 1)
    outdir = opts.get('outdir', '.')


    """ Initialization. """

    m, n = noisy.shape


    noisy_flat = np.reshape(noisy, -1)
    x = np.copy(noisy_flat)
    v = np.zeros_like(noisy_flat, dtype=np.float64)
    u = np.zeros_like(noisy_flat, dtype=np.float64)


    """ Main loop. """

    for i in range(maxitr):

        # record the variables in the current iteration
        x_old = np.copy(x)
        v_old = np.copy(v)
        u_old = np.copy(u)

        """ proximal step. """

        vtilde = np.copy((lam * (x + u) - 1.0) / lam)
        v = np.copy((vtilde + np.sqrt(np.square(vtilde) + 4.0*noisy_flat/lam)) / 2.0)

        """ denoising step. """

        xtilde = np.copy(2*v - x_old - u_old)

        # scale xtilde to be in range of [0,1]
        mintmp = np.min(xtilde)
        maxtmp = np.max(xtilde)
        xtilde = (xtilde - mintmp) / (maxtmp - mintmp)

        # load to torch
        xtilde_torch = np.reshape(xtilde, (1,1,m,n))
        xtilde_torch = torch.from_numpy(xtilde_torch).type(torch.cuda.FloatTensor)

        # denoise 
        r = denoiser(xtilde_torch).cpu().numpy()
        r = np.reshape(r, -1)
        x = xtilde - r

        # rescale the denoised v back to original scale
        x = x * (maxtmp - mintmp) + mintmp

        """ dual update """

        u = np.copy(u_old + x_old - v)

        """ Monitors """

        fpr = np.sqrt(np.sum(np.square(x + u - x_old - u_old)))

        if peak is not None and maxval is not None:
            psnrs = compare_psnr(im_true=clean,
                                    im_test=np.reshape(x/peak*maxval, (m, n)),
                                    data_range=data_range)
            if verbose:
                print("i = {},\t psnr = {},\t fpr = {}".format(i+1, psnrs, fpr))


    """ Get restored image. """
    x = np.reshape((x) , (m, n))
    return x

# ---- input arguments ----
opt = analyze_parse(40, 0.1, 100) # the arguments are default sigma, default alpha and default max iteration.

# ---- load the model ---- 
model = load_model(opt.model_type, opt.sigma)

with torch.no_grad():
    # ---- load the problem ---- 
    mat = sio.loadmat('Demo_mat/poisson_demo.mat')
    clean_scaled = mat.get('clean_scaled').astype(np.float64)
    clean = mat.get('clean').astype(np.float64)
    noisy = mat.get('noisy').astype(np.float64)
    peak = float(mat.get('peak'))
    maxval = float(mat.get('maxval'))

    # ---- options ---- 
    opts = dict(lam=1/opt.alpha, beta=1.0, maxitr=opt.maxitr,
                peak=peak, maxval=maxval, data_range=255.0, verbose=opt.verbose)
    
    # ---- plug and play !!!! ----
    out = pnp_admm_poisson_denoising(noisy, model, clean, **opts)


