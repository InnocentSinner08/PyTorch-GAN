from __future__ import print_function
import argparse
import os
import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data
import torchvision.datasets as dset
import torchvision.transforms as transforms
import torchvision.utils as vutils
from torchmetrics.image.fid import FrechetInceptionDistance
from model import ResDiscriminator32, ResGenerator32
from regan import Regan_training
import numpy as np
import warnings

warnings.filterwarnings("ignore")

def save_generated_images(netG, epoch, device, noise_size=128, num_images=16):
    with torch.no_grad():
        fixed_noise = torch.randn(num_images, noise_size, device=device)
        fake_images = netG(fixed_noise).cpu()
        os.makedirs("generated_images", exist_ok=True)
        vutils.save_image(fake_images, f'generated_images/generated_epoch_{epoch}.png', normalize=True, nrow=4)
        print(f"Generated images saved for epoch {epoch}")

def main():
    dataset = dset.CIFAR10(root=args.dataroot,
                           transform=transforms.Compose([
                               transforms.Resize(args.image_size),
                               transforms.CenterCrop(args.image_size),
                               transforms.ToTensor(),
                               transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                           ]), download=True, train=True)

    subset = torch.utils.data.Subset(dataset, np.arange(int(len(dataset) * args.data_ratio)))
    dataloader = torch.utils.data.DataLoader(subset, batch_size=args.batch_size, shuffle=True, num_workers=args.workers)

    netD = ResDiscriminator32().to(device)
    netG = Regan_training(ResGenerator32(args.noise_size).to(device), sparsity=args.sparsity)

    optimizerD = optim.Adam(netD.parameters(), args.lr, (0, 0.9))
    optimizerG = optim.Adam(netG.parameters(), args.lr, (0, 0.9))

    print("Starting Training Loop...")

    prev_loss = float('inf')
    loss_plateau_counter = 0
    patience = 5  # Number of epochs to wait before updating masks
    loss_threshold = 0.01  # Minimum loss change to consider improvement

    for epoch in range(1, args.epoch + 1):
        running_loss_G = 0.0

        for i, data in enumerate(dataloader, 0):
            netD.zero_grad()
            real_cpu = data[0].to(device)
            b_size = real_cpu.size(0)
            output = netD(real_cpu).view(-1)
            errD_real = torch.mean(nn.ReLU(inplace=True)(1.0 - output))
            errD_real.backward()
            
            noise = torch.randn(b_size, args.noise_size, device=device)
            fake = netG(noise)
            output = netD(fake.detach()).view(-1)
            errD_fake = torch.mean(nn.ReLU(inplace=True)(1 + output))
            errD_fake.backward()
            optimizerD.step()

            if i % args.n_critic == 0:
                netG.zero_grad()
                fake = netG(noise)
                output = netD(fake).view(-1)
                errG = -torch.mean(output)
                errG.backward()
                optimizerG.step()
                running_loss_G += errG.item()

        avg_loss_G = running_loss_G / len(dataloader)
        print(f'Epoch [{epoch}/{args.epoch}] - Generator Loss: {avg_loss_G:.4f}')

        if args.regan:
            if abs(prev_loss - avg_loss_G) < loss_threshold:
                loss_plateau_counter += 1
            else:
                loss_plateau_counter = 0
            
            if loss_plateau_counter >= patience:
                print("Loss plateau detected! Updating masks...")
                netG.update_masks()
                loss_plateau_counter = 0

            prev_loss = avg_loss_G

        if epoch % 10 == 0:
            save_generated_images(netG, epoch, device, args.noise_size)

if __name__ == '__main__':
    argparser = argparse.ArgumentParser()
    argparser.add_argument('--epoch', type=int, default=20)
    argparser.add_argument('--batch_size', type=int, default=64)
    argparser.add_argument('--lr', type=float, default=2e-4)
    argparser.add_argument('--workers', type=int, default=4)
    argparser.add_argument('--image_size', type=int, default=32)
    argparser.add_argument('--noise_size', type=int, default=128)
    argparser.add_argument('--dataroot', type=str, default='../dataset')
    argparser.add_argument('--n_critic', type=int, default=5)
    argparser.add_argument('--sparsity', type=float, default=0.3)
    argparser.add_argument('--data_ratio', type=float, default=1.0)
    argparser.add_argument('--regan', action="store_true")
    args = argparser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    main()
