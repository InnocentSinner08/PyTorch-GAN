import argparse
import os
import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data
import torchvision.datasets as dset
import torchvision.transforms as transforms
import torchvision.utils as vutils
from model import ResDiscriminator32, ResGenerator32
from regan import Regan_training
import numpy as np
import warnings
import subprocess

warnings.filterwarnings("ignore")

# Directory to save images
save_dir = "./generated_images/"
os.makedirs(save_dir, exist_ok=True)

def save_generated_images(G, epoch, device, num_samples=64):
    """Generate and save images every 10 epochs"""
    G.eval()
    with torch.no_grad():
        noise = torch.randn(num_samples, args.noise_size, 1, 1, device=device)
        fake_images = G(noise)
        vutils.save_image(fake_images, f"{save_dir}/epoch_{epoch}.png", normalize=True)
    G.train()

def compute_fid(real_images_path, generated_images_path):
    """Compute FID score between real and generated images"""
    fid_score = subprocess.run(["pytorch-fid", real_images_path, generated_images_path], capture_output=True, text=True)
    return fid_score.stdout.strip()

def main(args):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    os.makedirs(args.dataroot, exist_ok=True)

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

    prev_loss_D = float('inf')
    prev_loss_G = float('inf')
    loss_plateau_count = 0

    for epoch in range(1, args.epoch + 1):
        for i, data in enumerate(dataloader, 0):
            netD.zero_grad()
            real_cpu = data[0].to(device, dtype=torch.float32)
            b_size = real_cpu.size(0)

            output = netD(real_cpu).view(-1)
            errD_real = torch.mean(nn.ReLU(inplace=True)(1.0 - output))
            errD_real.backward()
            D_x = output.mean().item()

            noise = torch.randn(b_size, args.noise_size, device=device, dtype=torch.float32)
            fake = netG(noise)

            output = netD(fake.detach()).view(-1)
            errD_fake = torch.mean(nn.ReLU(inplace=True)(1 + output))
            errD_fake.backward()
            D_G_z1 = output.mean().item()
            errD = errD_real + errD_fake
            optimizerD.step()

            if i % args.n_critic == 0:
                netG.zero_grad()
                noise = torch.randn(b_size, args.noise_size, device=device, dtype=torch.float32)
                fake = netG(noise)
                output = netD(fake).view(-1)
                errG = -torch.mean(output)
                errG.backward()
                D_G_z2 = output.mean().item()

                if args.regan and netG.train_on_sparse:
                    netG.apply_masks()

                optimizerG.step()

            if abs(errD.item() - prev_loss_D) < args.loss_threshold and abs(errG.item() - prev_loss_G) < args.loss_threshold:
                loss_plateau_count += 1
            else:
                loss_plateau_count = 0

            prev_loss_D = errD.item()
            prev_loss_G = errG.item()

            if loss_plateau_count >= args.loss_patience:
                print(f"Loss plateau detected at epoch {epoch}, applying mask update...")
                netG.turn_training_mode(mode='sparse')
                netG.apply_masks()
                loss_plateau_count = 0

            if i % 50 == 0:
                print(f'[{epoch}/{args.epoch}][{i}/{len(dataloader)}] '
                      f'Loss_D: {errD.item():.4f} Loss_G: {errG.item():.4f} '
                      f'D(x): {D_x:.4f} D(G(z)): {D_G_z1:.4f} / {D_G_z2:.4f}')

        # Save images & compute FID every 10 epochs
        if epoch % 10 == 0:
            save_generated_images(netG, epoch, device)
            fid_score = compute_fid(args.real_images_path, save_dir)
            print(f"Epoch {epoch} - FID Score: {fid_score}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--epoch', type=int, default=1000, help="Number of training epochs")
    parser.add_argument('--batch_size', type=int, default=64, help="Batch size")
    parser.add_argument('--lr', type=float, default=2e-4, help="Learning rate")
    parser.add_argument('--workers', type=int, default=4, help="Number of data loader workers")
    parser.add_argument('--image_size', type=int, default=32, help="Size of input images")
    parser.add_argument('--noise_size', type=int, default=128, help="Noise vector size")
    parser.add_argument('--dataroot', type=str, default='../dataset', help="Dataset path")
    parser.add_argument('--clip_value', type=float, default=0.01, help="Weight clipping value")
    parser.add_argument('--n_critic', type=int, default=5, help="Number of critic updates per generator update")
    parser.add_argument('--sparsity', type=float, default=0.3, help="Sparsity level")
    parser.add_argument('--loss_threshold', type=float, default=0.001, help="Minimum loss change to avoid plateau")
    parser.add_argument('--loss_patience', type=int, default=5, help="Number of consecutive plateau epochs before pruning")
    parser.add_argument('--warmup_epoch', type=int, default=100, help="Warm-up epochs before pruning")
    parser.add_argument('--data_ratio', type=float, default=1.0, help="Fraction of dataset to use")
    parser.add_argument('--regan', action="store_true", help="Enable ReGAN pruning & growing")
    parser.add_argument('--real_images_path', type=str, required=True, help="Path to real images for FID computation")

    args = parser.parse_args()
    os.makedirs(args.dataroot, exist_ok=True)

    main(args)
