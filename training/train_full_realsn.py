import os
import datetime
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.utils as utils

from torch.autograd import Variable
from torch.utils.data import DataLoader
from tensorboardX import SummaryWriter
from model.full_realsn_models import DnCNN
from utilities.dataset import prepare_data, Dataset
from utilities.utils import *
from tqdm import tqdm

def str2bool(x):
    return x.lower() in ['true', '1', 'y', 'yes']

parser = argparse.ArgumentParser(description="DnCNN")
parser.add_argument("--preprocess", type=bool, default=False, help='run prepare_data or not')
parser.add_argument("--batchSize", type=int, default=128, help="Training batch size")
parser.add_argument("--num_of_layers", type=int, default=17, help="Number of total layers")
parser.add_argument("--lip", type=float, default=0.0, help="Lipschitz constraint. Default not using.")
parser.add_argument("--adaptive", type=str2bool, default=False, help="adaptive sn for different layers.")
parser.add_argument("--no_bn", type=str2bool, default=False, help="Whether use BN or not.")
parser.add_argument("--epochs", type=int, default=50, help="Number of training epochs")
parser.add_argument("--milestone", type=int, default=25, help="When to decay learning rate; should be less than epochs")
parser.add_argument("--lr", type=float, default=1e-3, help="Initial learning rate")
parser.add_argument("--outf", type=str, default="logs", help='path of log files')
parser.add_argument("--mode", type=str, default="S", help='with known noise level (S) or blind training (B)')
parser.add_argument("--noiseL", type=int, default=40, help='noise level; ignored when mode=B')
parser.add_argument("--val_noiseL", type=float, default=40, help='noise level used on validation set')
parser.add_argument("--gpu", type=int, default=0, help='GPU index')
opt = parser.parse_args()

print("Lipschitz constant: {}".format(opt.lip))

if opt.gpu is None:
    raise ValueError("please provide gpu id")
os.environ["CUDA_VISIBLE_DEVICES"] = "%d" % opt.gpu

def Load_Model(model, arg1, arg2, arg3):
    return model.load_state_dict(torch.load(os.path.join('logs', arg1, arg2, arg3)))

def main():
    noiseL_B = [15,45]  # ingnored when opt.mode=='S'
    if opt.mode == 'B':
        sigma = noiseL_B
    else:
        sigma = opt.noiseL
    save_path = creat_SavePath(opt.outf, sigma, opt.num_of_layers, opt.mode,
                               opt.lip, opt.no_bn, opt.adaptive)

    print('Loading dataset ...\n')
    dataset_train = Dataset(train=True)
    dataset_val = Dataset(train=False)
    loader_train = DataLoader(dataset=dataset_train, num_workers=4, batch_size=opt.batchSize, shuffle=True)
    print("# of training samples: %d\n" % int(len(dataset_train)))

    # Build model
    print(opt.no_bn)
    model = DnCNN(channels=1, num_of_layers=opt.num_of_layers,
                  lip=opt.lip, no_bn=opt.no_bn).cuda()
    print(model)
    model.apply(weights_init_kaiming)

    # Loss function
    criterion = nn.MSELoss(size_average=False)
    criterion.cuda()

    # Optimizer
    optimizer = optim.Adam(model.parameters(), lr=opt.lr)
    # training
    writer = SummaryWriter(save_path)
    step = 0

    # for epoch in range(opt.epochs):
    for epoch in range(opt.epochs):
        if epoch < opt.milestone:
            current_lr = opt.lr
        else:
            current_lr = opt.lr / 10.
        # set learning rate
        for param_group in optimizer.param_groups:
            param_group["lr"] = current_lr
        print('learning rate %f' % current_lr)
        # train
        for i, data in enumerate(tqdm(loader_train), 0):
            # training step
            model.train()
            model.zero_grad()
            optimizer.zero_grad()
            img_train = data
            if opt.mode == 'S':
                noise = torch.FloatTensor(img_train.size()).normal_(mean=0, std=opt.noiseL/255.)
            if opt.mode == 'B':
                noise = torch.zeros(img_train.size())
                stdN = np.random.uniform(noiseL_B[0], noiseL_B[1], size=noise.size()[0])
                for n in range(noise.size()[0]):
                    sizeN = noise[0,:,:,:].size()
                    noise[n,:,:,:] = torch.FloatTensor(sizeN).normal_(mean=0, std=stdN[n]/255.)
            imgn_train = img_train + noise
            img_train, imgn_train = Variable(img_train.cuda()), Variable(imgn_train.cuda())
            noise = Variable(noise.cuda())
            out_train = model(imgn_train)

            loss = criterion(out_train, noise) / (imgn_train.size()[0]*2)
            loss.backward()
            optimizer.step()
            # results
            model.eval()
            out_train = torch.clamp(imgn_train-model(imgn_train), 0., 1.)
            psnr_train = batch_PSNR(out_train, img_train, 1.)
            # if you are using older version of PyTorch, you may need to change loss.item() to loss.data[0]
            if (i + 1) % 200 == 0:
                print("[epoch %d][%d/%d] loss: %.4f PSNR_train: %.4f" %
                      (epoch + 1, i + 1, len(loader_train), loss.item(), psnr_train))
            if step % 10 == 0:
                # Log the scalar values
                writer.add_scalar('loss', loss.item(), step)
                writer.add_scalar('PSNR on training data', psnr_train, step)
            step += 1
            torch.save(model.state_dict(), os.path.join(save_path, 'Latest_Model.pth'))
        ## the end of each epoch
        model.eval()
        # validate
        psnr_val = 0
        ssim_val = 0
        for k in range(len(dataset_val)):
            img_val = torch.unsqueeze(dataset_val[k], 0)
            if opt.mode == 'S':
                noise = torch.FloatTensor(img_val.size()).normal_(mean=0, std=opt.noiseL/255.)
            if opt.mode == 'B':
                noise = torch.zeros(img_val.size())
                stdN = np.random.uniform(noiseL_B[0], noiseL_B[1], size=noise.size()[0])
                for n in range(noise.size()[0]):
                    sizeN = noise[0,:,:,:].size()
                    noise[n,:,:,:] = torch.FloatTensor(sizeN).normal_(mean=0, std=stdN[n]/255.)
            # noise = torch.FloatTensor(img_val.size()).normal_(mean=0, std=opt.val_noiseL/255.)
            imgn_val = img_val + noise
            img_val, imgn_val = Variable(img_val.cuda()), Variable(imgn_val.cuda())
            out_val = torch.clamp(imgn_val-model(imgn_val), 0., 1.)
            psnr_val += batch_PSNR(out_val, img_val, 1.)
            ssim_val += batch_SSIM(out_val, img_val, 1.)
        psnr_val /= len(dataset_val)
        ssim_val /= len(dataset_val)
        print("\n[epoch %d]  PSNR_val: %.4f  SSIM_val: %.4f" % (epoch+1, psnr_val, ssim_val))
        print("-" * 40)
        # writer.add_scalar('PSNR on validation data', psnr_val, epoch)
        # log the images
        writer.add_scalar('Test PSNR', psnr_val, epoch)
        writer.add_scalar('Test SSIM', ssim_val, epoch)

        out_train = torch.clamp(imgn_train-model(imgn_train), 0., 1.)
        Img = utils.make_grid(img_train.data, nrow=8, normalize=True, scale_each=True)
        Imgn = utils.make_grid(imgn_train.data, nrow=8, normalize=True, scale_each=True)
        Irecon = utils.make_grid(out_train.data, nrow=8, normalize=True, scale_each=True)
        writer.add_image('clean image', Img, epoch)
        writer.add_image('noisy image', Imgn, epoch)
        writer.add_image('reconstructed image', Irecon, epoch)
        # save model
        # model_name = 'epoch{:d}_noise{:d}_PSNR{:.2f}_SSIM{:.2f}.pth'.format(epoch + 1, opt.noiseL, psnr_val, ssim_val)
        # torch.save(model.state_dict(), os.path.join(save_path, model_name))

        if opt.mode == 'B':
            model_name = 'epoch{:d}_noise{}_PSNR{:.2f}_SSIM{:.2f}.pth'.format(epoch+1, noiseL_B, psnr_val, ssim_val)
        else:
            model_name = 'epoch{:d}_noise{:d}_PSNR{:.2f}_SSIM{:.2f}.pth'.format(epoch+1, opt.noiseL, psnr_val, ssim_val)
        torch.save(model.state_dict(), os.path.join(save_path, model_name))

if __name__ == "__main__":
    if opt.preprocess:
        if opt.mode == 'S':
            prepare_data(data_path='data', patch_size=40, stride=10, aug_times=1)
        if opt.mode == 'B':
            prepare_data(data_path='data', patch_size=50, stride=10, aug_times=2)
    main()

