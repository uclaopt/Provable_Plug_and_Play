# Provable_Plug_and_Play

The implement of the following paper: 

E. K. Ryu, J. Liu, S. Wang, X. Chen, Z. Wang, and W. Yin. "Plug-and-Play Methods Provably Converge with Properly Trained Denoisers." ICML, 2019.

Scripts
========================
1. pnp_admm_csmri.py  (CS-MRI solved with Plug-and-Play ADMM)
2. pnp_fbs_csmri.py  (CS-MRI solved with Plug-and-Play FBS)
3. pnp_admm_poisson_denoise.py (Poisson Denoising solved with Plug-and-Play ADMM)
4. pnp_fbs_poisson_denoise.py (Poisson Denoising solved with Plug-and-Play FBS)
5. pnp_admm_photon_imaging.py (Single Photon Imaging solved with Plug-and-Play ADMM)
6. pnp-fbs_photon_imaging.py (to appear soon)

How to run the scripts
========================

## Run with default settings
```
$ python3 pnp_admm_csmri.py
```

## Run with costmized settings
```
$ python3 pnp_admm_csmri.py --model_type DnCNN --sigma 15 --alpha 2.0 --maxitr 100 --verbose 1
```
All the arguments are explained in the file "utils/config.py".

Citation
=========================
If you find our code helpful in your resarch or work, please cite our paper.
```
@InProceedings{pmlr-v97-ryu19a,
  title = 	 {Plug-and-Play Methods Provably Converge with Properly Trained Denoisers},
  author = 	 {Ryu, Ernest and Liu, Jialin and Wang, Sicheng and Chen, Xiaohan and Wang, Zhangyang and Yin, Wotao},
  booktitle = 	 {Proceedings of the 36th International Conference on Machine Learning},
  pages = 	 {5546--5557},
  year = 	 {2019},
  editor = 	 {Chaudhuri, Kamalika and Salakhutdinov, Ruslan},
  volume = 	 {97},
  series = 	 {Proceedings of Machine Learning Research},
  address = 	 {Long Beach, California, USA},
  month = 	 {09--15 Jun},
  publisher = 	 {PMLR},
  pdf = 	 {http://proceedings.mlr.press/v97/ryu19a/ryu19a.pdf},
  url = 	 {http://proceedings.mlr.press/v97/ryu19a.html},
```
