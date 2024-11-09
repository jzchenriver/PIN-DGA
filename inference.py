import argparse
import torch
import os
import numpy as np
import copy
import cv2
from tqdm import tqdm
import torch.nn as nn
import torch.utils.data
import torch.backends.cudnn

from model.model_HoLoCoNet import Y_HoLoCoNet
from utils.data import Test_Loader

parser = argparse.ArgumentParser()
parser.add_argument('--base_size', type=int, default=512, help='image size')
parser.add_argument('--model', type=str, default='result/PIN-DGA/IRSTD-1k/best_model/best.pth')
args = parser.parse_args()


def adjust_momentum(model, motum):
    for nm, m in model.named_modules():
        if isinstance(m, nn.BatchNorm2d):
            m.track_running_stats = True
            m.momentum = motum
    return model


def collect_meanvar(model):
    name_list = []
    mean_list = []
    var_list = []
    for nm, m in model.named_modules():
        if isinstance(m, nn.BatchNorm2d):
            name_list.append(nm)
            mean_list.append(m.running_mean)
            var_list.append(m.running_var)
    return name_list, mean_list, var_list


def kldivergence(p, q):
    mean_1, var_1 = p[0], p[1]
    mean_2, var_2 = q[0], q[1]
    kl_pq = torch.log(torch.sqrt(var_2) / torch.sqrt(var_1)) + \
            (var_1 + torch.pow((mean_1 - mean_2), 2)) / (2 * var_2) - 0.5
    return kl_pq


def JS(mean_1, mean_2, var_1, var_2, name):
    layer_number = len(mean_1)
    js_list = []
    for l in range(layer_number):
        require_l = 'mask' in name[l]
        if require_l:
            mean_1_l = mean_1[l]
            var_1_l = var_1[l]
            mean_2_l = mean_2[l]
            var_2_l = var_2[l]
            channel_number = len(mean_1_l)
            js_l_list = []

            for ch in range(channel_number):
                t1 = torch.cat((mean_1_l[ch].unsqueeze(0), var_1_l[ch].unsqueeze(0)), dim=0)  # p
                t2 = torch.cat((mean_2_l[ch].unsqueeze(0), var_2_l[ch].unsqueeze(0)), dim=0)  # q
                tmean = (t1 + t2) / 2
                js = (kldivergence(t2, tmean) + kldivergence(t1, tmean)) / 2
                js_l_list.append(js.item())
            js_l = np.mean(js_l_list)
            js_list.append(js_l)
    if not js_list:
        js = 0
    else:
        js = np.mean(js_list)
    return js


def predict(adaptnet, data, show_mask_path):
    adaptnet.eval()
    label = data['label'].to(device, dtype=torch.float32)
    name = data['name']
    # Predict
    with torch.no_grad():
        try:
            pred_label, _ = adaptnet(label)
        except:
            pred_label = adaptnet(label)
    pred_label = torch.sigmoid(pred_label)
    pred_masks = pred_label
    for i in range(pred_masks.shape[0]):
        pred_mask = np.array(pred_masks.data.cpu()[i])[0]
        pred_max = np.max(pred_mask)
        pred_mask[pred_mask >= 0.5] = 255
        pred_mask[pred_mask < 0.5] = 0

        cv2.imwrite(os.path.join(show_mask_path, name[i][5:]), pred_mask)

    print("Predict complete!")
    adaptnet.train()
    return adaptnet


class Infer:

    def inference(self, original_path, test_dataset, show_mask_path, net, batch_size, momentum_k):
        # Load the model
        net.load_state_dict(torch.load(original_path))
        adaptnet = copy.deepcopy(net)
        tempnet = copy.deepcopy(net)
        print("The num of the target data：", test_dataset.__len__())
        train_test_loader = torch.utils.data.DataLoader(dataset=test_dataset,
                                                        batch_size=batch_size,
                                                        shuffle=False,
                                                        num_workers=4,
                                                        pin_memory=True,
                                                        drop_last=False)

        for param in adaptnet.parameters():
            param.requires_grad = False

        name_oracle, mean_oracle, var_oracle = collect_meanvar(adaptnet)  

        for _ in range(1):
            adaptnet.train()
            tempnet.train()
            momentum_list = []
            with tqdm(train_test_loader, desc='train %s' % (test_path.split('/')[-1]), unit="batch") as tepoch:
                for data in tepoch:
                    img_input = data['label'].to(device, dtype=torch.float32)
                    tempnet = adjust_momentum(tempnet, motum=1)
                    with torch.no_grad():
                        try:
                            _, _ = tempnet(img_input)
                        except:
                            _ = tempnet(img_input)
                    name_current, mean_current, var_current = collect_meanvar(tempnet)  
                    del tempnet
                    torch.cuda.empty_cache()
                    momentum_t = momentum_k * JS(mean_current, mean_oracle, var_current, var_oracle,
                                                 name_current)  
                    momentum_t = min(momentum_t, 1)
                    momentum_list.append(momentum_t)
                    adaptnet = adjust_momentum(adaptnet, motum=momentum_t)

                    with torch.no_grad():
                        try:
                            _, _ = adaptnet(img_input)
                        except:
                            _ = adaptnet(img_input)
                    adaptnet = predict(adaptnet, data, show_mask_path)
                    tempnet = copy.deepcopy(adaptnet)
                    name_oracle, mean_oracle, var_oracle = collect_meanvar(adaptnet)
        return momentum_list


if __name__ == "__main__":

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    original_path = args.model
    train_dataset = original_path.split('/')[2]
    base_size = args.base_size

    net = Y_HoLoCoNet()
    net.to(device=device)

    if train_dataset == 'IRST640':
        test_datasets = ['IRSTD-1k',  'KTB', 'a1', 'b1', 'c1', 'c2', 'cl1', 'cl2',
                         'f1', 'f2', 'm1', 'm2', 'r1', 'r2']

    if train_dataset == 'IRSTD-1k':
        test_datasets = ['IRST640',  'KTB', 'a1', 'b1', 'c1', 'c2', 'cl1', 'cl2',
                         'f1', 'f2', 'm1', 'm2', 'r1', 'r2']

    for test_dataset in test_datasets:
        print(test_dataset)
        test_path = 'test_img_%s' % test_dataset
        shows_path = 'datasets/test_img/' + test_path
        show_mask_path = shows_path + '/PIN-DGA-' + train_dataset
        os.makedirs(show_mask_path, exist_ok=True)
        mydata = Infer()
        test_dataset = Test_Loader(base_size, test_path)
        momentum = mydata.inference(original_path, test_dataset, show_mask_path, net, batch_size=6, momentum_k=1)

