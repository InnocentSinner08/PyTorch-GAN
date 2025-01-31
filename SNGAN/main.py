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
    """Generate and save images from the generator."""
    with torch.no_grad():
        fixed_noise = torch.randn(num_images, noise_size, device=device)
        fake_images = netG(fixed_noise).cpu()
        os.makedirs("generated_images", exist_ok=True)
        vutils.save_image(fake_images, f'generated_images/generated_epoch_{epoch}.png', normalize=True, nrow=4)
        print(f"Generated images saved for epoch {epoch}")

def compute_fid_score(netG, real_images, device, noise_size=128, num_samples=1024):
    """Compute FID score between real and generated images."""
    fid = FrechetInceptionDistance(feature=2048).to(device)

    # Convert real images to uint8
    real_images = ((real_images * 0.5 + 0.5) * 255).clamp(0, 255).byte()
    real_images = real_images.to(device)
    fid.update(real_images, real=True)

    # Generate fake images and convert to uint8
    with torch.no_grad():
        noise = torch.randn(num_samples, noise_size, device=device)
        fake_images = netG(noise)
        fake_images = ((fake_images * 0.5 + 0.5) * 255).clamp(0, 255).byte()
        fid.update(fake_images, real=False)

    fid_score = fid.compute().item()
    print(f"FID Score: {fid_score}")
    return fid_score

def main():
    dataset = dset.CIFAR10(root=args.dataroot,
                           transform=transforms.Compose([
                               transforms.Resize(args.image_size),
                               transforms.CenterCrop(args.image_size),
                               transforms.ToTensor(),
                               transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                           ]), download=True, train=True)

    subset = torch.utils.data.Subset(dataset, np.arange(int(len(dataset) * args.data_ratio)))
    dataloader = torch.utils.data.DataLoader(subset, batch_size=args.batch_size,
                                             shuffle=True, num_workers=args.workers)

    netD = ResDiscriminator32().to(device)
    netG = Regan_training(ResGenerator32(args.noise_size).to(device), sparsity=args.sparsity)

    optimizerD = optim.Adam(netD.parameters(), args.lr, (0, 0.9))
    optimizerG = optim.Adam(netG.parameters(), args.lr, (0, 0.9))

    print("Starting Training Loop...")

    flag_g = 1
    fid_scores = []

    for epoch in range(1, args.epoch + 1):

        if args.regan:
            if epoch < args.warmup_epoch + 1:
                print('Warmup training...')
                netG.train_on_sparse = False
            elif epoch > args.warmup_epoch and flag_g < args.g + 1:
                print(f'Epoch {epoch}, Sparse training...')
                netG.turn_training_mode(mode='sparse')
                if flag_g == 1:
                    for params in optimizerG.param_groups:
                        params['lr'] = args.lr
                flag_g += 1
            elif epoch > args.warmup_epoch and flag_g < 2 * args.g + 1:
                print(f'Epoch {epoch}, Dense training...')
                netG.turn_training_mode(mode='dense')
                if flag_g == args.g + 1:
                    for params in optimizerG.param_groups:
                        params['lr'] = args.lr * 0.1
                flag_g += 1
                if flag_g == 2 * args.g + 1:
                    flag_g = 1

        for i, data in enumerate(dataloader, 0):

            netD.zero_grad()
            real_cpu = data[0].to(device)
            b_size = real_cpu.size(0)
            output = netD(real_cpu).view(-1)
            errD_real = torch.mean(nn.ReLU(inplace=True)(1.0 - output))
            errD_real.backward()
            D_x = output.mean().item()

            noise = torch.randn(b_size, args.noise_size, device=device)
            fake = netG(noise)

            output = netD(fake.detach()).view(-1)
            errD_fake = torch.mean(nn.ReLU(inplace=True)(1 + output))
            errD_fake.backward()
            D_G_z1 = output.mean().item()
            errD = errD_real + errD_fake
            optimizerD.step()

            if i % args.n_critic == 0:
                netG.zero_grad()
                noise = torch.randn(b_size, args.noise_size, device=device)
                fake = netG(noise)
                output = netD(fake).view(-1)
                errG = -torch.mean(output)
                errG.backward()
                D_G_z2 = output.mean().item()

                if args.regan and netG.train_on_sparse:
                    netG.apply_masks()

                optimizerG.step()

            if i % 50 == 0:
                print('[%4d/%4d][%3d/%3d]\tLoss_D: %.4f\tLoss_G: %.4f\tD(x): %.4f\tD(G(z)): %.4f / %.4f'
                      % (epoch, args.epoch, i, len(dataloader),
                         errD.item(), errG.item(), D_x, D_G_z1, D_G_z2))

        # Save generated images every 10 epochs
        if epoch % 10 == 0:
            save_generated_images(netG, epoch, device, args.noise_size)

        # Compute FID score every 10 epochs
        if epoch % 10 == 0:
            fid_score = compute_fid_score(netG, real_cpu, device, args.noise_size)
            fid_scores.append((epoch, fid_score))

    # Save FID scores to file
    with open("fid_scores.txt", "w") as f:
        for epoch, score in fid_scores:
            f.write(f"Epoch {epoch}: FID {score}\n")

if __name__ == '__main__':
    model_name = 'SNGAN'
    argparser = argparse.ArgumentParser()
    argparser.add_argument('--epoch', type=int, default=20)
    argparser.add_argument('--batch_size', type=int, default=64)
    argparser.add_argument('--lr', type=float, default=2e-4)
    argparser.add_argument('--workers', type=int, default=4)
    argparser.add_argument('--image_size', type=int, default=32)
    argparser.add_argument('--noise_size', type=int, default=128)
    argparser.add_argument('--dataroot', type=str, default='../dataset')
    argparser.add_argument('--clip_value', type=float, default=0.01)
    argparser.add_argument('--n_critic', type=int, default=5)
    argparser.add_argument('--sparsity', type=float, default=0.3)
    argparser.add_argument('--g', type=int, default=5)
    argparser.add_argument('--warmup_epoch', type=int, default=100)
    argparser.add_argument('--data_ratio', type=float, default=1.0)
    argparser.add_argument('--regan', action="store_true")
    args = argparser.parse_args()

    if not os.path.exists(args.dataroot):
        os.makedirs(args.dataroot)

    device = "cuda" if torch.cuda.is_available() else "cpu"

    main()
