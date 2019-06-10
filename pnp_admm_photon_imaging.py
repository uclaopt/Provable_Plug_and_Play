"""
    Plug and Play ADMM for Single Photon Imaging
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


def inverse_step(u, v, K1, K, rho):

    """ proximal operator "prox_{alpha f}" for single photon imaging """

    xtilde = v - u
    x = np.copy(xtilde)
    K0 = np.square(K) - K1

    indices_0 = (K1 == 0)
    x[indices_0] = xtilde[indices_0] - K0[indices_0] / rho

    func = lambda y: K1 / (np.exp(y) - 1) - rho*y - K0 + rho*xtilde
    indices_1 = np.logical_not(indices_0)

    # binary search?
    bmin = 1e-5 * np.ones_like(x, dtype=np.float64)
    bmax = 100  * np.ones_like(x, dtype=np.float64)
    bave = (bmin + bmax) / 2.0
    for i in range(30):
        tmp = func(bave)
        indices_pos = np.logical_and(tmp > 0, indices_1)
        indices_neg = np.logical_and(tmp < 0, indices_1)
        indices_zero = np.logical_and(tmp == 0, indices_1)
        indices_0 = np.logical_or(indices_0, indices_zero)
        indices_1 = np.logical_not(indices_0)

        bmin[indices_pos] = bave[indices_pos]
        bmax[indices_neg] = bave[indices_neg]
        bave[indices_1] = (bmin[indices_1] + bmax[indices_1]) / 2.0

    x[K1 != 0] = bave[K1 != 0]
    # project back to image domain of range [0,1]
    return np.clip(x, 0.0, 1.0)

def blockfunc(ob, block_shape, func): # clear
    """
    Parameters:
        :ob: the observation of single photon imaging
        :block_shape: block shpae of this operation
        :func: the function that is applied to each block
    """

    # precompute some variables
    ob_m, ob_n = ob.shape

    # block shape
    b_m, b_n = block_shape

    # define the size of resulting image
    out_m, out_n = ob_m // b_m, ob_n // b_n

    # placeholder for the output
    out = np.zeros(shape=(out_m, out_n), dtype=np.float64)

    for i in range(out_m):
        for j in range(out_n):
            out[i][j] = func(ob[i*b_m:(i+1)*b_m, j*b_n:(j+1)*b_n])

    return out


def pnp_admm_photon_imaging(b, denoiser, im_true, **opts):
    """
    Parameters:
        :b - the observation in single photon imaging.
        :denoiser - the Gaussian denoiser used in Plug-and-Play ADMM.
        :im_true - the clean image used to monitor PSNR.
        :opts - the kwargs for hyperparameters in Plug-and-Play ADMM.
            :K - the parameter in single photo imaging.
            :lam - the value of 1/alpha.
            :rho - TODO
            :maxitr - the max number of iterations.
            :verbose - a flag that enables/disables info printing.
                - NOTE: if `peak` and `M` options exist in `opts`, then the
                  `clean` image is the scaled version of the original image.
            :beta - the prior weight parameter.
            :step - TODO
    """

    """ Process parameters. """

    K      = opts.get('K', 8)
    lam    = opts.get('lam', 15.0)
    rho    = opts.get('rho', 100.0) # rho = 1.0 / alpha
    maxitr = opts.get('maxitr', 50)
    data_range = opts.get('data_range', 1.0)
    verbose = opts.get('verbose', 1)
    step   = opts.get('step', 1.0)
    beta   = opts.get('beta', 1.0)


    """ Initialization. """

    K1 = blockfunc(b, (K, K), np.sum)
    x = K1 / K**2

    u = np.zeros_like(x, dtype=np.float64)
    v = np.copy(x)

    m, n = x.shape

    sigma = np.sqrt(lam / rho)

    """ Main loop. """
    for i in range(maxitr):

        x_old = x
        u_old = u
        v_old = v

        """ Inverse step. """

        x = inverse_step(u, v, K1, K, rho)

        """ Denoising step. """

        vtilde = x + u

        # scale vtilde to be in range of [0,1]
        mintmp = 0.0
        maxtmp = np.max(vtilde)
        vtilde = (vtilde - mintmp) / (maxtmp - mintmp)
        trans_sigma = sigma / (maxtmp - mintmp)

        # then set data range to [0.15, 0.85] to avoid clipping of extreme values
        scale_range = 0.4
        scale_shift = (1 - scale_range) / 2.0
        vtilde = vtilde * scale_range + scale_shift
        trans_sigma = trans_sigma * scale_range

        # pytorch denoising model
        vtilde_torch = np.reshape(vtilde, (1,1,m,n))
        vtilde_torch = torch.from_numpy(vtilde_torch).type(torch.FloatTensor).cuda()
        r = denoiser(vtilde_torch).cpu().numpy()
        r = np.reshape(r, (m,n))
        v = vtilde - r

        # scale and shift the denoised image back
        v = (v - scale_shift) / scale_range
        v = v * (maxtmp - mintmp) + mintmp


        """ Update variables. """
        u = u + x - v

        """ Monitoring. """
        # successive difference
        dif = np.sqrt(np.sum(np.square( x - x_old )))
        dif_denom = np.sqrt(np.sum(np.square( x_old )))

        # psnr

        if verbose:
            print("i: {}, \t successive difference: {}, \t psnr: {} \t"\
                  .format(i+1, dif/dif_denom, psnr(im_true, x)))

    return x



# ---- input arguments ----
opt = analyze_parse(15, 0.01, 15) # the arguments are default sigma, default alpha and default max iteration.

# ---- load the model ---- 
model = load_model(opt.model_type, opt.sigma)


with torch.no_grad():
    # ---- load problem ------ 
    mat = sio.loadmat('./Demo_mat/single_photon_imaging_demo.mat')
    im = mat.get('im').astype(np.float64)
    ob = mat.get('ob').astype(np.float64)

    # ---- options ----
    opts = dict(K=8, rho=1.0 / opt.alpha, lam=15.0, beta=1.0, maxitr=opt.maxitr, data_range=255.0, verbose=opt.verbose)

    # ---- plug and play !!! -----
    out = pnp_admm_photon_imaging(ob, model, im, **opts)





