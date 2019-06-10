# config file.
# Authors: Jialin Liu (UCLA math, danny19921123@gmail.com)

import argparse

# ---- analyze the parse arguments -----
def analyze_parse(default_sigma, default_alpha, default_itr):
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_type", type=str, default="RealSN_DnCNN", help='DnCNN/ SimpleCNN / RealSN_DnCNN / RealSN_SimpleCNN')
    parser.add_argument("--sigma", type=int, default=default_sigma, help="Noise level for the denoising model")
    parser.add_argument("--alpha", type=float, default=default_alpha, help="Step size in Plug-and Play")
    parser.add_argument("--maxitr", type=int, default=default_itr, help="Number of iterations")
    parser.add_argument("--verbose", type=int, default=1, help="Whether printing the info out")
    opt = parser.parse_args()
    return opt
