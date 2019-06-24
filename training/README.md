# Provable_Plug_and_Play

The implement of the training method in the following paper: 

E. K. Ryu, J. Liu, S. Wang, X. Chen, Z. Wang, and W. Yin. "Plug-and-Play Methods Provably Converge with Properly Trained Denoisers." ICML, 2019.


How to run the script
========================

## First-time running (with default settings)
```
$ python3 train_full_realsn.py --preprocess True
```

## Running with default settings after the first time
```
$ python3 train_full_realsn.py
```

## Running with costmized settings

DnCNN (default)
```
$ python3 train_full_realsn.py
```
RealSN-DnCNN
```
$ python3 train_full_realsn.py --lip 1.0
```
SimpleCNN
```
$ python3 train_full_realsn.py --no_bn True
```
RealSN-SimpleCNN
```
$ python3 train_full_realsn.py --no_bn True --lip 1.0
```
All the arguments are explained in the file "train_full_realsn.py".


Acknowledgment
=========================
We use the same dataset and loading method as the following repository:
https://github.com/SaoYan/DnCNN-PyTorch


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
  url = 	 {http://proceedings.mlr.press/v97/ryu19a.html}
}
```

