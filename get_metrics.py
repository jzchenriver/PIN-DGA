import torch.utils.data as data
from tqdm import tqdm
import numpy as np
from argparse import ArgumentParser
from utils.image import PredictImage
from utils.metrics import SigmoidMetric, SamplewiseSigmoidMetric, PD_FA


def parse_args():
    parser = ArgumentParser(description='Implement of ACM model')

    parser.add_argument('--mode', type=str, default='val', help='val')
    parser.add_argument('--train_dataset', type=str, default='IRSTD-1k', help='train dataset')
    parser.add_argument('--meth', type=str, default='PIN-DGA', help='save folder')

    parser.add_argument('--base-size', type=int, default=512, help='base image size')
    parser.add_argument('--batch-size', type=int, default=1, help='batch_size for val')

    args = parser.parse_args()
    return args


class Infer(object):
    def __init__(self, args, inferset):
        # dataset
        self.infer_data_loader = data.DataLoader(inferset, batch_size=args.batch_size)
        self.infer_data_len = len(self.infer_data_loader)

        # evaluation metrics
        self.BINS = 20
        self.iou_metric = SigmoidMetric()
        self.nIoU_metric = SamplewiseSigmoidMetric(1, score_thresh=1/(1+np.exp(-0.5)))
        self.PD_FA = PD_FA(1, self.BINS)

    def val(self, test_dataset):
        self.iou_metric.reset()
        self.nIoU_metric.reset()
        self.PD_FA.reset()
        IoU = 0
        nIoU = 0

        tbar = tqdm(self.infer_data_loader)
        for i, (msk, res, w, h) in enumerate(tbar):
            self.iou_metric.update(res, msk)
            self.nIoU_metric.update(res, msk)
            res = res[:, 0, :, :]
            msk = msk[:, 0, :, :]
            self.PD_FA.update(res, msk, w, h)
            _, IoU = self.iou_metric.get()  
            _, nIoU = self.nIoU_metric.get()  

        FA, PD, FAT = self.PD_FA.get(len(self.infer_data_loader), w, h)

        if test_dataset in ['IRST640', 'IRSTD-1k', 'KTB']:
            print('Metrics on {}'.format(test_dataset))
            print('IoU: ')
            print(IoU)
            print('nIoU: ')
            print(nIoU)
            print('PD: ')
            print(PD[0])
            print('FA: ')
            print(FA[0])
            print('FAT: ')
            print(FAT[0])

        return IoU, nIoU, PD, FA, FAT


if __name__ == '__main__':
    args = parse_args()
    train_dataset = args.train_dataset

    if train_dataset == 'IRST640':
        test_datasets = ['IRSTD-1k', 'KTB', 'a1', 'b1', 'c1', 'c2', 'cl1', 'cl2',
                         'f1', 'f2', 'm1', 'm2', 'r1', 'r2']

    if train_dataset == 'IRSTD-1k':
        test_datasets = ['IRST640', 'KTB', 'a1', 'b1', 'c1', 'c2', 'cl1', 'cl2',
                         'f1', 'f2', 'm1', 'm2', 'r1', 'r2']

    IoU_list, nIoU_list, FA_array, PD_array, FAT_array = [], [], np.zeros((14, 21)), np.zeros((14, 21)), np.zeros((14, 21))

    for i in range(len(test_datasets)):
        test_dataset = test_datasets[i]
        inferset = PredictImage(args, test_dataset)
        infer = Infer(args, inferset)
        IoU, nIoU, PD, FA, FAT = infer.val(test_dataset)

        IoU_list.append(IoU)
        nIoU_list.append(nIoU)
        PD_array[i, :] = PD
        FA_array[i, :] = FA
        FAT_array[i, :] = FAT

    IoU_IRSTScenes = np.mean(IoU_list[2:])
    nIoU_IRSTScenes = np.mean(nIoU_list[2:])
    PD_IRSTScenes = np.mean(PD_array[2:, :], axis=0)
    FA_IRSTScenes = np.mean(FA_array[2:, :], axis=0)
    FAT_IRSTScenes = np.mean(FAT_array[2:, :], axis=0)

    print('Metrics on IRSTScenes datasets')
    print('IoU: ')
    print(IoU_IRSTScenes)
    print('nIoU: ')
    print(nIoU_IRSTScenes)
    print('PD: ')
    print(PD_IRSTScenes[0])
    print('FA: ')
    print(FA_IRSTScenes[0])
    print('FAT: ')
    print(FAT_IRSTScenes[0])

    PD_array[2, :] = PD_IRSTScenes
    FA_array[2, :] = FA_IRSTScenes
    FAT_array[2, :] = FAT_IRSTScenes

    IoU_avg = np.mean([IoU_list[0], IoU_list[1], np.mean(IoU_list[2:])])
    nIoU_avg = np.mean([nIoU_list[0], nIoU_list[1], np.mean(nIoU_list[2:])])
    PD_avg = np.mean(PD_array[0:3, :], axis=0)
    FA_avg = np.mean(FA_array[0:3, :], axis=0)
    FAT_avg = np.mean(FAT_array[0:3, :], axis=0)

    print('Metrics on all datasets')
    print('IoU: ')
    print(IoU_avg)
    print('nIoU: ')
    print(nIoU_avg)
    print('PD: ')
    print(PD_avg[0])
    print('FA: ')
    print(FA_avg[0])
    print('FAT: ')
    print(FAT_avg[0])