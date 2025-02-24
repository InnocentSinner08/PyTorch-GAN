import numpy as np
import torch
import argparse
import os
import torch.nn as nn
import torch.optim as optim
import torch.utils.data
import torchvision.datasets as dset
import torchvision.transforms as transforms
import torchvision.utils as vutils
from model import ResDiscriminator32, ResGenerator32
from regan import Regan_training
import warnings

warnings.filterwarnings("ignore")
from torchmetrics.image.fid import FrechetInceptionDistance
import torch.nn.functional as F

def compute_fid_score(netG, dataloader, device, noise_size=128, num_images=1000):
    """Computes FID score between real and generated images."""
    
    fid = FrechetInceptionDistance(feature=2048).to(device)
    
    netG.eval()
    real_images_collected = 0
    
    # Collect real images
    for real_batch in dataloader:
        real_images = real_batch[0].to(device)
        real_images = F.interpolate(real_images, size=(299, 299), mode='bilinear', align_corners=False)  # Resize for Inception
        fid.update(real_images, real=True)
        real_images_collected += real_images.shape[0]
        if real_images_collected >= num_images:
            break

    # Generate fake images
    with torch.no_grad():
        for _ in range(num_images // dataloader.batch_size + 1):
            noise = torch.randn(dataloader.batch_size, noise_size, device=device)
            fake_images = netG(noise)
            fake_images = F.interpolate(fake_images, size=(299, 299), mode='bilinear', align_corners=False)  # Resize for Inception
            fid.update(fake_images, real=False)

    fid_score = fid.compute().item()
    print(f"FID Score: {fid_score:.4f}")
    return fid_score

def detect_plateau(loss_history, patience=5, loss_threshold=0.01):
    """Detects if the loss has plateaued by checking if the recent loss values have minimal change."""
    if len(loss_history) < patience:
        return False  # Not enough history to decide
    recent_losses = loss_history[-patience:]
    loss_change = np.max(recent_losses) - np.min(recent_losses)
    return loss_change < loss_threshold

def save_generated_images(netG, epoch, device, noise_size=128, num_images=16):
    """Generate and save images from the generator."""
    with torch.no_grad():
        fixed_noise = torch.randn(num_images, noise_size, device=device)
        fake_images = netG(fixed_noise).cpu()
        os.makedirs("generated_images", exist_ok=True)
        vutils.save_image(fake_images, f'generated_images/generated_epoch_{epoch}.png', normalize=True, nrow=4)
        print(f"Generated images saved for epoch {epoch}")

def train_with_dynamic_pruning(netG, netD, optimizerG, optimizerD, dataloader, device, args):
    """Train function with dynamic loss-based pruning updates."""
    loss_history = []  # Store generator loss history
    
    for epoch in range(1, args.epoch + 1):
        epoch_loss_G = []
        
        for i, data in enumerate(dataloader, 0):
            netD.zero_grad()
            real_cpu = data[0].to(device)
            b_size = real_cpu.size(0)
            
            output = netD(real_cpu).view(-1)
            errD_real = torch.mean(torch.nn.ReLU()(1.0 - output))
            errD_real.backward()
            
            noise = torch.randn(b_size, args.noise_size, device=device)
            fake = netG(noise)
            output = netD(fake.detach()).view(-1)
            errD_fake = torch.mean(torch.nn.ReLU()(1 + output))
            errD_fake.backward()
            optimizerD.step()
            
            if i % args.n_critic == 0:
                netG.zero_grad()
                noise = torch.randn(b_size, args.noise_size, device=device)
                fake = netG(noise)
                output = netD(fake).view(-1)
                errG = -torch.mean(output)
                errG.backward()
                optimizerG.step()
                
                epoch_loss_G.append(errG.item())
        
        avg_loss_G = np.mean(epoch_loss_G)
        loss_history.append(avg_loss_G)
        print(f"Epoch [{epoch}/{args.epoch}] - Generator Loss: {avg_loss_G:.4f}")
        
        # Check if pruning/regrowth should occur
        if detect_plateau(loss_history, patience=5, loss_threshold=0.01):
            print(f"Loss plateau detected at epoch {epoch}. Updating masks.")
            netG.update_masks()
            loss_history = []  # Reset history after mask update
        
        # Save images every 10 epochs
                # Save images and compute FID score every 10 epochs
        if epoch % 10 == 0:
            save_generated_images(netG, epoch, device, args.noise_size)
            fid_score = compute_fid_score(netG, dataloader, device, args.noise_size)
            print(f"Epoch {epoch} - FID Score: {fid_score:.4f}")


if __name__ == '__main__':
    argparser = argparse.ArgumentParser()
    argparser.add_argument('--epoch', type=int, default=1000)
    argparser.add_argument('--batch_size', type=int, default=64)
    argparser.add_argument('--lr', type=float, default=2e-4)
    argparser.add_argument('--workers', type=int, default=4)
    argparser.add_argument('--image_size', type=int, default=32)
    argparser.add_argument('--noise_size', type=int, default=128)
    argparser.add_argument('--dataroot', type=str, default='../dataset')
    argparser.add_argument('--clip_value', type=float, default=0.01)
    argparser.add_argument('--n_critic', type=int, default=5)
    argparser.add_argument('--sparsity', type=float, default=0.3)
    argparser.add_argument('--data_ratio', type=float, default=1.0)
    argparser.add_argument('--regan', action="store_true")
    args = argparser.parse_args()
    
    if not os.path.exists(args.dataroot):
        os.makedirs(args.dataroot)
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
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
    train_with_dynamic_pruning(netG, netD, optimizerG, optimizerD, dataloader, device, args)
