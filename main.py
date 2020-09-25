import torch
import torch.nn as nn
from torchvision import transforms
import os
import argparse
from utils import split_dataset, train, validate
from grayscale import Grayscale
from colorize import Colorize

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

ap = argparse.ArgumentParser()
ap.add_argument("-i", "--image", required=True,
                help="path to input image")
ap.add_argument("-e", "--epochs", type=int, default=100,
                help="# of epochs")
args = vars(ap.parse_args())
split_dataset(args["image"])

# Training
train_transforms = transforms.Compose([transforms.RandomResizedCrop(224), transforms.RandomHorizontalFlip()])
train_folder = Grayscale('images/train', train_transforms)
train_loader = torch.utils.data.DataLoader(train_folder, batch_size=64, shuffle=True)

# Validation
val_transforms = transforms.Compose([transforms.Resize(256), transforms.CenterCrop(224)])
val_folder = Grayscale('images/val', val_transforms)
val_loader = torch.utils.data.DataLoader(val_folder, batch_size=64, shuffle=False)

# Model
model = Colorize()
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-2, weight_decay=0.0)

os.makedirs('outputs/color', exist_ok=True)
os.makedirs('outputs/gray', exist_ok=True)
os.makedirs('checkpoints', exist_ok=True)
save_images = True
best_losses = 1e10
epochs = args["epochs"]

for epoch in range(epochs):
    train(train_loader, model, criterion, optimizer, epoch, device)
    with torch.no_grad():
        losses = validate(val_loader, model, criterion, save_images, epoch, device)
    if losses < best_losses:
        best_losses = losses
        torch.save(model.state_dict(), 'checkpoints/model-epoch-{}-losses-{:.3f}.pth'.format(epoch + 1, losses))
