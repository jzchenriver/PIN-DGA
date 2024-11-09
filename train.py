import argparse
import torch
import os
import numpy as np
from tqdm import tqdm
import torch.backends.cudnn
import torch.utils.data
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter

from model.model_HoLoCoNet import Y_HoLoCoNet
from utils.data import YNet_loader
from utils.loss import CBSLoss
from utils.pytorchtools import EarlyStopping
from utils.sam import FSAM

parser = argparse.ArgumentParser()
parser.add_argument('--train_dataset', type=str, default='IRSTD-1k')
parser.add_argument('--base_size', type=int, default=512, help='image size')
parser.add_argument('--save_path', type=str, default='result/PIN-DGA/')
parser.add_argument('--batch_size', type=int, default=4)
parser.add_argument('--n_epochs', type=int, default=200)
parser.add_argument('--lr', type=float, default=0.05)
args = parser.parse_args()


def train(net, train_label_loader, optimizer, det_loss, rec_loss, epoch, writer):

    label_loss_list, unlabel_loss_list, loss_list = [], [], []
    net.train()
    learning_rate = optimizer.param_groups[0]['lr']
    pbar = tqdm(total=len(train_label_loader), position=0, colour='blue', leave=True, ncols=80)
    for batch, data in enumerate(train_label_loader):
        pbar.set_description(f"Epoch {epoch + 1}")
        # label_data
        label = data['label'].to(device, dtype=torch.float32)
        input = data['input'].to(device, dtype=torch.float32)
        mask = data['mask'].to(device)
        seg_label = data['seg_label'].to(device)

        seg_pred, output = net(input)
        # Recovery loss
        unlabel_loss = rec_loss(output * (1 - mask), label * (1 - mask))
        # Detction loss
        label_loss = det_loss(seg_pred, seg_label)
        loss = unlabel_loss + label_loss
        loss.backward()
        optimizer.first_step(zero_grad=True)

        seg_pred, output = net(input)
        # Recovery loss
        unlabel_loss = rec_loss(output * (1 - mask), label * (1 - mask))
        # Detection loss
        label_loss = det_loss(seg_pred, seg_label)
        loss = unlabel_loss + label_loss
        loss.backward()
        optimizer.second_step(zero_grad=True)

        loss_list.append(loss.item())
        label_loss_list.append(label_loss.item())
        unlabel_loss_list.append(unlabel_loss.item())

        pbar.set_postfix(loss=loss.item())
        pbar.update(1)

    pbar.close()
    writer.add_scalar('train/loss', np.mean(loss_list), global_step=epoch)
    writer.add_scalar('train/label_loss', np.mean(label_loss_list), global_step=epoch)
    writer.add_scalar('train/unlabel_loss', np.mean(unlabel_loss_list), global_step=epoch)
    writer.add_scalar('train/lr', learning_rate, global_step=epoch)

    return np.mean(loss_list)


def validate(net, val_loader, det_loss, epoch, writer):
    net.eval()
    with torch.no_grad():
        val_loss_list = []
        pbar = tqdm(total=len(val_loader), position=0, colour='blue', leave=True, ncols=80)
        for batch, data in enumerate(val_loader):
            pbar.set_description(f"Epoch {epoch + 1}")

            input = data['input'].to(device, dtype=torch.float32)
            seg_label = data['seg_label'].to(device)
            seg_pred, _ = net(input)
            # Detection loss
            loss = det_loss(seg_pred, seg_label)

            val_loss_list.append(loss.item())

            pbar.set_postfix(loss=loss.item())
            pbar.update(1)

        pbar.close()
    # write to tensorboard
    writer.add_scalar('val/loss', np.mean(val_loss_list), global_step=epoch)

    return np.mean(val_loss_list)


def train_net(net, device, base_size, train_dataset, save_path, epochs, batch_size, lr):
    save_path = save_path + train_dataset + '/'
    os.makedirs(save_path, exist_ok=True)
    # Training set
    train_label_dataset = YNet_loader(base_size, train_dataset)
    val_dataset = YNet_loader(base_size, train_dataset, val_mode=True)

    print("The num of labeled training data：", train_label_dataset.__len__())
    train_label_loader = torch.utils.data.DataLoader(dataset=train_label_dataset,
                                                     batch_size=batch_size,
                                                     shuffle=True,
                                                     num_workers=4,
                                                     pin_memory=True)

    print("The num of validating data：", val_dataset.__len__())
    val_loader = torch.utils.data.DataLoader(dataset=val_dataset,
                                             batch_size=batch_size,
                                             shuffle=False,
                                             num_workers=4,
                                             pin_memory=False)

    base_optimizer = torch.optim.SGD
    optimizer = FSAM(net.named_parameters(), net.parameters(), base_optimizer, adaptive=False, lr=lr, momentum=0.9)

    early_stopping = EarlyStopping(patience=20, verbose=True)
    rec_loss = nn.L1Loss().to(device)
    det_loss = CBSLoss().to(device)
    best_loss = float('inf')
    writer = SummaryWriter(log_dir=save_path + 'logs')
    for epoch in range(epochs):
        loss = train(net, train_label_loader, optimizer, det_loss, rec_loss, epoch, writer)
        print('loss=%.5f' % (loss))
        # Validating
        val_loss = validate(net, val_loader, det_loss, epoch, writer)
        print('val_loss=%.5f' % (val_loss))

        os.makedirs(save_path + 'every_model', exist_ok=True)
        torch.save(net.state_dict(), save_path + 'every_model/epoch_%d.pth' % epoch)

        if val_loss < best_loss:
            best_loss = val_loss
            os.makedirs(save_path + 'best_model', exist_ok=True)
            torch.save(net.state_dict(), save_path + 'best_model/best.pth')

        early_stopping(val_loss, net)
        if early_stopping.early_stop:
            print("Early stopping")
            break

    print('\nbest_loss=%.5f' % (best_loss))


if __name__ == "__main__":
    torch.cuda.empty_cache()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    model = Y_HoLoCoNet()
    model.to(device=device)

    train_net(model, device, args.base_size, args.train_dataset, args.save_path, epochs=args.n_epochs, batch_size=args.batch_size, lr=args.lr)
