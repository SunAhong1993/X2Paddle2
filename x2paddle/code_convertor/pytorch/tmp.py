from x2paddle import torch2paddle
import argparse
import logging
import os
import sys
import numpy as np
import paddle as torch
import paddle.nn as nn
from paddle import optimizer as optim
from tqdm import tqdm
import time
from eval import eval_net
from unet import UNet
from utils.dataset import BasicDataset
from x2paddle.torch2paddle import random_split
from x2paddle.torch2paddle import BaseDataLoader as DataLoader
dir_img = '../Pytorch-UNet/data/train/'
dir_mask = '../Pytorch-UNet/data/train_masks/'
dir_checkpoint = 'checkpoints/'


def train_net(net, device, epochs=5, batch_size=1, lr=0.001, val_percent=
    0.1, save_cp=True, img_scale=0.5):
    dataset = BasicDataset(dir_img, dir_mask, img_scale)
    n_val = int(len(dataset) * val_percent)
    n_train = len(dataset) - n_val
    train, val = random_split(dataset, [n_train, n_val])
    train_loader = BaseDataLoader(train, batch_size=batch_size, shuffle=
        False, num_workers=1)
    val_loader = BaseDataLoader(val, batch_size=batch_size, shuffle=False,
        num_workers=8, drop_last=True)
    global_step = 0
    logging.info(
        f"""Starting training:
        Epochs:          {epochs}
        Batch size:      {batch_size}
        Learning rate:   {lr}
        Training size:   {n_train}
        Validation size: {n_val}
        Checkpoints:     {save_cp}
        Device:          {device.type}
        Images scaling:  {img_scale}
    """
        )
    optimizer = optimizer.Adam(parameters=net.parameters(), learning_rate=
        lr, beta1=0.9, beta2=0.999)
    scheduler = optimizer.lr.ReduceOnPlateau(patience=2, mode='min' if net.
        n_classes > 1 else 'max', learning_rate=0.01)
    optimizer._learning_rate = scheduler
    if net.n_classes > 1:
        criterion = nn.CrossEntropyLoss()
    else:
        criterion = nn.BCEWithLogitsLoss()
    for epoch in range(epochs):
        net.train()
        epoch_loss = 0
        with tqdm(total=n_train, desc=f'Epoch {epoch + 1}/{epochs}', unit='img'
            ) as pbar:
            start = 0
            all_cost = 0
            for i, batch in enumerate(train_loader):
                c = time.time() - start
                imgs = batch['image']
                true_masks = batch['mask']
                assert imgs.shape[1
                    ] == net.n_channels, f'Network has been defined with {net.n_channels} input channels, but loaded images have {imgs.shape[1]} channels. Please check that the images are loaded correctly.'
                imgs = imgs.to(device=device, dtype='float32')
                mask_type = 'float32' if net.n_classes == 1 else 'int64'
                true_masks = true_masks.to(device=device, dtype=mask_type)
                masks_pred = net(imgs)
                loss = criterion(masks_pred, true_masks)
                epoch_loss += loss.item()
                pbar.set_postfix(**{'loss (batch)': loss.item()})
                start = time.time()
                optimizer.zero_grad()
                loss.backward()
                torch2paddle.clip_grad_value_(net.parameters(), 0.1)
                optimizer.step()
                end = time.time()
                pbar.update(imgs.shape[0])
                global_step += 1
                if i >= 10 and i < 20:
                    all_cost += end - start
                if i == 20:
                    print(all_cost / 10.0)
                    break
        if save_cp:
            try:
                os.mkdir(dir_checkpoint)
                logging.info('Created checkpoint directory')
            except OSError:
                pass
            paddle.save(net.state_dict(), dir_checkpoint +
                f'CP_epoch{epoch + 1}.pth')
            logging.info(f'Checkpoint {epoch + 1} saved !')


def get_args():
    parser = argparse.ArgumentParser(description=
        'Train the UNet on images and target masks', formatter_class=
        argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('-e', '--epochs', metavar='E', type=int, default=5,
        help='Number of epochs', dest='epochs')
    parser.add_argument('-b', '--batch-size', metavar='B', type=int, nargs=
        '?', default=1, help='Batch size', dest='batchsize')
    parser.add_argument('-l', '--learning-rate', metavar='LR', type=float,
        nargs='?', default=0.0001, help='Learning rate', dest='lr')
    parser.add_argument('-f', '--load', dest='load', type=str, default=
        False, help='Load model from a .pth file')
    parser.add_argument('-s', '--scale', dest='scale', type=float, default=
        0.5, help='Downscaling factor of the images')
    parser.add_argument('-v', '--validation', dest='val', type=float,
        default=10.0, help=
        'Percent of the data that is used as validation (0-100)')
    return parser.parse_args()


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s'
        )
    args = get_args()
    device = 'cuda' if paddle.is_compiled_with_cuda() else 'cpu'
    device = device.replace('cuda', 'gpu')
    device = paddle.set_device(device)
    logging.info(f'Using device {device}')
    net = UNet(n_channels=3, n_classes=1, bilinear=True)
    logging.info(
        f"""Network:
	{net.n_channels} input channels
	{net.n_classes} output channels (classes)
	{'Bilinear' if net.bilinear else 'Transposed conv'} upscaling"""
        )
    if args.load:
        net.load_state_dict(paddle.load(args.load))
        logging.info(f'Model loaded from {args.load}')
    net.to(device=device)
    try:
        train_net(net=net, epochs=args.epochs, batch_size=args.batchsize,
            lr=args.lr, device=device, img_scale=args.scale, val_percent=
            args.val / 100)
    except KeyboardInterrupt:
        paddle.save(net.state_dict(), 'INTERRUPTED.pth')
        logging.info('Saved interrupt')
        try:
            sys.exit(0)
        except SystemExit:
            os._exit(0)
