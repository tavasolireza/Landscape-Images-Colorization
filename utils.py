import os, time
from skimage.color import lab2rgb
import torch
import matplotlib.pyplot as plt
import numpy as np
from average_meter import AverageMeter


def split_dataset():
    os.makedirs('/content/images/train/class/', exist_ok=True)
    os.makedirs('/content/images/val/class/', exist_ok=True)
    for i, file in enumerate(os.listdir('/content/landscape')):
        if i < 1000:
            os.rename('/content/landscape/' + file, '/content/images/val/class/' + file)
        else:
            os.rename('/content/landscape/' + file, '/content/images/train/class/' + file)


def convert_to_rgb(grayscale_input, ab_input, save_path=None, save_name=None):
    plt.clf()
    color_image = torch.cat((grayscale_input, ab_input), 0).numpy()
    color_image = color_image.transpose((1, 2, 0))
    color_image[:, :, 0:1] = color_image[:, :, 0:1] * 100
    color_image[:, :, 1:3] = color_image[:, :, 1:3] * 255 - 128
    color_image = lab2rgb(color_image.astype(np.float64))
    grayscale_input = grayscale_input.squeeze().numpy()
    if save_path is not None and save_name is not None:
        plt.imsave(arr=grayscale_input, fname='{}{}'.format(save_path['grayscale'], save_name), cmap='gray')
        plt.imsave(arr=color_image, fname='{}{}'.format(save_path['colorized'], save_name))


def validate(val_loader, model, criterion, save_images, epoch, device):
    model.eval()

    batch_time, data_time, losses = AverageMeter(), AverageMeter(), AverageMeter()

    end = time.time()
    already_saved_images = False
    for i, (input_gray, input_ab, target) in enumerate(val_loader):
        data_time.update(time.time() - end)

        # Use GPU
        input_gray, input_ab, target, model = input_gray.to(device), input_ab.to(device), target.to(device), model.to(
            device)

        output_ab = model(input_gray)
        loss = criterion(output_ab, input_ab)
        losses.update(loss.item(), input_gray.size(0))

        if save_images and not already_saved_images:
            already_saved_images = True
            for j in range(min(len(output_ab), 10)):
                save_path = {'grayscale': '/content/outputs/gray/', 'colorized': '/content/outputs/color/'}
                save_name = 'img-{}-epoch-{}.jpg'.format(i * val_loader.batch_size + j, epoch)
                convert_to_rgb(input_gray[j].cpu(), ab_input=output_ab[j].detach().cpu(), save_path=save_path,
                               save_name=save_name)

        batch_time.update(time.time() - end)
        end = time.time()

        print('Validate: [{0}/{1}]\t'
              'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
              'Loss {loss.val:.4f} ({loss.avg:.4f})\t'.format(
            i, len(val_loader), batch_time=batch_time, loss=losses))

    print('Finished validation.')
    return losses.avg


def train(train_loader, model, criterion, optimizer, epoch, device):
    print('Starting training epoch {}'.format(epoch))
    model.train()

    batch_time, data_time, losses = AverageMeter(), AverageMeter(), AverageMeter()

    end = time.time()
    for i, (input_gray, input_ab, target) in enumerate(train_loader):
        input_gray, input_ab, target, model = input_gray.to(device), input_ab.to(device), target.to(device), model.to(
            device)

        data_time.update(time.time() - end)

        output_ab = model(input_gray)
        loss = criterion(output_ab, input_ab)
        losses.update(loss.item(), input_gray.size(0))

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        batch_time.update(time.time() - end)
        end = time.time()

        print('Epoch: [{0}][{1}/{2}]\t'
              'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
              'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
              'Loss {loss.val:.4f} ({loss.avg:.4f})\t'.format(
            epoch, i, len(train_loader), batch_time=batch_time,
            data_time=data_time, loss=losses))

    print('Finished training epoch {}'.format(epoch))
