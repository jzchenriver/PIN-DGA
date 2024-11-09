import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np
from skimage import measure


class SigmoidMetric():
    def __init__(self):
        self.reset()

    def update(self, pred, labels):
        correct, labeled = self.batch_pix_accuracy(pred, labels)
        inter, union = self.batch_intersection_union(pred, labels)

        self.total_correct += correct
        self.total_label += labeled
        self.total_inter += inter
        self.total_union += union
        self.correct = correct
        self.label = labeled
        self.inter = inter
        self.union = union

    def now(self):
        pixAcc = 1.0 * self.correct / (np.spacing(1) + self.label)
        IoU = 1.0 * self.inter / (np.spacing(1) + self.union)
        mIoU = IoU.mean()
        return pixAcc, mIoU

    def get(self):
        """Gets the current evaluation result."""
        pixAcc = 1.0 * self.total_correct / (np.spacing(1) + self.total_label)
        IoU = 1.0 * self.total_inter / (np.spacing(1) + self.total_union)
        mIoU = IoU.mean()
        return pixAcc, mIoU

    def reset(self):
        """Resets the internal evaluation result to initial state."""
        self.total_inter = 0
        self.total_union = 0
        self.total_correct = 0
        self.total_label = 0
        self.inter = 0
        self.union = 0
        self.correct = 0
        self.label = 0

    def batch_pix_accuracy(self, output, target):
        assert output.shape == target.shape
        output = output.detach().numpy()
        target = target.detach().numpy()

        predict = (output > 0.5).astype('int64')  # P
        pixel_labeled = np.sum(target > 0)  # T
        pixel_correct = np.sum((predict == target) * (target > 0))  # TP
        assert pixel_correct <= pixel_labeled
        return pixel_correct, pixel_labeled

    def batch_intersection_union(self, output, target):
        mini = 1
        maxi = 1  # nclass
        nbins = 1  # nclass
        predict = (output.detach().numpy() > 0.5).astype('int64')  # P
        target = target.numpy().astype('int64')  # T
        intersection = predict * (predict == target)  # TP

        # areas of intersection and union
        area_inter, _ = np.histogram(intersection, bins=nbins, range=(mini, maxi))
        area_pred, _ = np.histogram(predict, bins=nbins, range=(mini, maxi))
        area_lab, _ = np.histogram(target, bins=nbins, range=(mini, maxi))
        area_union = area_pred + area_lab - area_inter
        assert (area_inter <= area_union).all()
        return area_inter, area_union


class SamplewiseSigmoidMetric():
    def __init__(self, nclass, score_thresh=0.5):
        self.nclass = nclass
        self.score_thresh = score_thresh
        self.reset()

    def update(self, preds, labels):
        """Updates the internal evaluation result."""
        inter_arr, union_arr = self.batch_intersection_union(preds, labels,
                                                             self.nclass, self.score_thresh)
        self.total_inter = np.append(self.total_inter, inter_arr)
        self.total_union = np.append(self.total_union, union_arr)

    def get(self):
        """Gets the current evaluation result."""
        IoU = 1.0 * self.total_inter / (np.spacing(1) + self.total_union)
        mIoU = IoU.mean()

        return IoU, mIoU

    def reset(self):
        """Resets the internal evaluation result to initial state."""
        self.total_inter = np.array([])
        self.total_union = np.array([])
        self.total_correct = np.array([])
        self.total_label = np.array([])

    def batch_intersection_union(self, output, target, nclass, score_thresh):
        """mIoU"""
        # inputs are tensor
        # the category 0 is ignored class, typically for background / boundary
        mini = 1
        maxi = 1  # nclass
        nbins = 1  # nclass

        predict = (torch.sigmoid(output).detach().numpy() > score_thresh).astype('int64')  # P
        target = target.detach().numpy().astype('int64')  # T
        intersection = predict * (predict == target)  # TP

        num_sample = intersection.shape[0]
        area_inter_arr = np.zeros(num_sample)
        area_pred_arr = np.zeros(num_sample)
        area_lab_arr = np.zeros(num_sample)
        area_union_arr = np.zeros(num_sample)

        for b in range(num_sample):
            # areas of intersection and union
            area_inter, _ = np.histogram(intersection[b], bins=nbins, range=(mini, maxi))
            area_inter_arr[b] = area_inter

            area_pred, _ = np.histogram(predict[b], bins=nbins, range=(mini, maxi))
            area_pred_arr[b] = area_pred

            area_lab, _ = np.histogram(target[b], bins=nbins, range=(mini, maxi))
            area_lab_arr[b] = area_lab

            area_union = area_pred + area_lab - area_inter
            area_union_arr[b] = area_union

            assert (area_inter <= area_union).all()

        return area_inter_arr, area_union_arr


class ROCMetric():
    def __init__(self, nclass, bins):
        self.nclass = nclass
        self.bins = bins
        self.tp_arr = np.zeros(self.bins + 1)
        self.pos_arr = np.zeros(self.bins + 1)
        self.fp_arr = np.zeros(self.bins + 1)
        self.neg_arr = np.zeros(self.bins + 1)
        self.target_arr = np.zeros(self.bins + 1)
        self.t_arr = np.zeros(self.bins + 1)

    def update(self, preds, labels):
        for iBin in range(self.bins + 1):
            score_thresh = (iBin + 0.0) / self.bins
            i_tp, i_pos, i_fp, i_neg, i_target, i_t = cal_tp_pos_fp_neg_target(preds, labels, self.nclass, score_thresh)

            self.tp_arr[iBin] += i_tp
            self.pos_arr[iBin] += i_pos
            self.fp_arr[iBin] += i_fp
            self.neg_arr[iBin] += i_neg
            self.target_arr[iBin] += i_target
            self.t_arr[iBin] += i_t

    def get(self):
        tp_rates = self.tp_arr / (self.pos_arr + 0.001)
        fp_rates = self.fp_arr / (self.neg_arr + 0.001)
        tar = self.target_arr / (self.t_arr)

        return tp_rates, fp_rates, tar


def cal_tp_pos_fp_neg_target(output, target, nclass, score_thresh):
    mini = 1
    maxi = 1  # nclass
    nbins = 1  # nclass

    predict = (torch.sigmoid(output).detach().numpy() > score_thresh).astype('int64').squeeze()  # P
    target = target.detach().numpy().astype('int64').squeeze()  # T
    intersection = predict * (predict == target)  # TP
    tp = intersection.sum()
    fp = (predict * (predict != target)).sum()  # FP
    tn = ((1 - predict) * (predict == target)).sum()  # TN
    fn = ((predict != target) * (1 - predict)).sum()  # FN
    pos = tp + fn
    neg = fp + tn

    labels = measure.label(target, connectivity=2)  #
    # stats
    properties = measure.regionprops(labels)

    t = labels.max()
    tar = 0
    m, n = target.shape
    for prop in properties:
        center = prop.centroid

        r = round(center[1])
        c = round(center[0])

        distanceMap = np.zeros((m, n))
        distanceMap[c - 2: c, r - 2: r] = 1

        if sum(sum(predict * predict * distanceMap)) > 0:
            tar += 1

    return tp, pos, fp, neg, tar, t


class T_ROCMetric():
    def __init__(self, nclass, bins):
        self.nclass = nclass
        self.bins = bins
        self.tp_arr = np.zeros(self.bins + 1)
        self.pos_arr = np.zeros(self.bins + 1)
        self.fp_arr = np.zeros(self.bins + 1)
        self.neg_arr = np.zeros(self.bins + 1)

    def update(self, preds, labels):
        for iBin in range(self.bins + 1):
            score_thresh = (iBin + 0.0) / self.bins
            i_tp, i_fp, i_neg = cal_tp_fp_neg(preds, labels, self.nclass, score_thresh)

            self.tp_arr[iBin] += i_tp
            self.fp_arr[iBin] += i_fp
            self.neg_arr[iBin] += i_neg

    def get(self):
        tp_rates = self.tp_arr
        fp_rates = self.fp_arr / (self.neg_arr + 0.001)

        return tp_rates, fp_rates


def cal_tp_fp_neg(output, target, nclass, score_thresh):
    mini = 1
    maxi = 1  # nclass
    nbins = 1  # nclass

    predict = (torch.sigmoid(output).detach().numpy() > score_thresh).astype('int64').squeeze()  # P
    target = target.detach().numpy().astype('int64').squeeze()  # T

    labels = measure.label(target, connectivity=2)  
    properties = measure.regionprops(labels)
    tp = 0
    fp = (predict * (predict != target)).sum()  # FP
    tn = ((1 - predict) * (predict == target)).sum()  # TN
    neg = fp + tn

    m, n = target.shape

    for prop in properties:
        center = prop.centroid

        r = round(center[1])
        c = round(center[0])

        distanceMap = np.zeros((m, n));
        distanceMap[r - 2: r, c - 2: c] = 1;

        if sum(sum(predict * predict * distanceMap)) > 0:
            tp = 1
        else:
            tp = 0

    return tp, fp, neg


class PD_FA:
    def __init__(self, nclass, bins):
        super(PD_FA, self).__init__()
        self.nclass = nclass
        self.bins = bins
        self.image_area_total = []
        self.image_area_match = []
        self.FA = np.zeros(self.bins + 1)
        self.PD = np.zeros(self.bins + 1)
        self.target = np.zeros(self.bins + 1)
        self.predict = np.zeros(self.bins + 1)

    def update(self, preds, labels, w, h):

        for iBin in range(self.bins + 1):
            score_thresh = iBin * (1 / self.bins)
            # preds = torch.sigmoid(preds)
            predits = np.array((preds > score_thresh).cpu()).astype('int64')
            predits = np.reshape(predits, (w, h))
            labelss = np.array((labels).cpu()).astype('int64')  # P
            labelss = np.reshape(labelss, (w, h))

            image = measure.label(predits, connectivity=2)
            coord_image = measure.regionprops(image)
            label = measure.label(labelss, connectivity=2)
            coord_label = measure.regionprops(label)

            self.target[iBin] += len(coord_label)
            self.predict[iBin] += len(coord_image)
            self.image_area_total = []
            self.image_area_match = []
            self.distance_match = []
            self.dismatch = []

            for K in range(len(coord_image)):
                area_image = np.array(coord_image[K].area)
                self.image_area_total.append(area_image)

            for i in range(len(coord_label)):
                centroid_label = np.array(list(coord_label[i].centroid))
                for m in range(len(coord_image)):
                    centroid_image = np.array(list(coord_image[m].centroid))
                    distance = np.linalg.norm(centroid_image - centroid_label)
                    area_image = np.array(coord_image[m].area)
                    if distance < 3:
                        self.distance_match.append(distance)
                        self.image_area_match.append(area_image)

                        del coord_image[m]
                        break

            self.dismatch = [x for x in self.image_area_total if x not in self.image_area_match]
            self.FA[iBin] += np.sum(self.dismatch)
            self.PD[iBin] += len(self.distance_match)

    def get(self, img_num, w, h):

        Final_FA = self.FA / ((w * h) * img_num)
        Final_FA = Final_FA.numpy()
        # if Final_FA.dtype == torch.float32:
        #     Final_FA = Final_FA.cpu().numpy()
        Final_PD = self.PD / self.target  
        Final_FAT = self.predict - self.PD

        return Final_FA, Final_PD, Final_FAT

    def reset(self):
        self.FA = np.zeros([self.bins + 1])
        self.PD = np.zeros([self.bins + 1])


class PD_FA_output:
    def __init__(self, nclass, bins):
        super(PD_FA_output, self).__init__()
        self.nclass = nclass
        self.bins = bins
        self.image_area_total = []
        self.image_area_match = []
        self.FA = np.zeros(self.bins + 1)
        self.PD = np.zeros(self.bins + 1)
        self.target = np.zeros(self.bins + 1)
        self.predict = np.zeros(self.bins + 1)

    def update(self, preds, labels, w, h, txt_path, name):

        for iBin in range(self.bins + 1):
            score_thresh = 0
            # preds = torch.sigmoid(preds)
            predits = np.array((preds > score_thresh).cpu()).astype('int64')
            predits = np.reshape(predits, (w, h))
            labelss = np.array((labels).cpu()).astype('int64')  # P
            labelss = np.reshape(labelss, (w, h))

            image = measure.label(predits, connectivity=2)
            coord_image = measure.regionprops(image)
            label = measure.label(labelss, connectivity=2)
            coord_label = measure.regionprops(label)

            self.target[iBin] += len(coord_label)
            self.predict[iBin] += len(coord_image)
            self.image_area_total = []
            self.image_area_match = []
            self.distance_match = []
            self.dismatch = []

            for K in range(len(coord_image)):
                area_image = np.array(coord_image[K].area)
                self.image_area_total.append(area_image)

            for i in range(len(coord_label)):
                centroid_label = np.array(list(coord_label[i].centroid))
                for m in range(len(coord_image)):
                    centroid_image = np.array(list(coord_image[m].centroid))
                    distance = np.linalg.norm(centroid_image - centroid_label)
                    area_image = np.array(coord_image[m].area)
                    if distance < 3:
                        self.distance_match.append(distance)
                        self.image_area_match.append(area_image)

                        del coord_image[m]
                        coord_label[i] = 0
                        break

            if any(coord_label):
                with open(txt_path, 'a') as txt_file:
                    txt_file.write(name[0] + '\n')

            self.dismatch = [x for x in self.image_area_total if x not in self.image_area_match]
            self.FA[iBin] += np.sum(self.dismatch)
            self.PD[iBin] += len(self.distance_match)

    def get(self, img_num, w, h):

        Final_FA = self.FA / ((w * h) * img_num)
        # if Final_FA.dtype == torch.float32:
        #     Final_FA = Final_FA.cpu().numpy()
        Final_PD = self.PD / self.target 
        Final_FAT = self.predict - self.PD

        return Final_FA, Final_PD, Final_FAT

    def reset(self, test_dataset):
        self.FA = np.zeros([self.bins + 1])
        self.PD = np.zeros([self.bins + 1])

        txt_path = '/home/piton/JPIN/datasets/MD_list/missed_img_on_%s.txt' % test_dataset

        return txt_path


def calculateF1Measure(output_image, gt_image, thre=0):
    output_image = output_image.cpu().numpy()
    gt_image = gt_image.cpu().numpy()
    output_image = np.squeeze(output_image)
    gt_image = np.squeeze(gt_image)
    out_bin = (output_image > thre)
    gt_bin = (gt_image > thre)

    recall = np.sum(gt_bin * out_bin) / np.maximum(1, np.sum(gt_bin))
    prec = np.sum(gt_bin * out_bin) / np.maximum(1, np.sum(out_bin))
    F1 = 2 * recall * prec / np.maximum(0.001, recall + prec)
    return F1
