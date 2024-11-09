import copy
import os
import numpy as np
from torch.utils.data import Dataset
from PIL import Image


class YNet_loader(Dataset):
    def __init__(self, base_size, train_dataset, val_mode=False, ratio=0.7, size_window=(9, 9)):

        self.base_size = base_size
        self.data_path = 'datasets/' + train_dataset
        self.val_mode = val_mode
        self.ratio = ratio
        self.size_data = (1, base_size, base_size)
        self.size_window = size_window

        if self.val_mode:
            self.idx_list = np.loadtxt(os.path.join(self.data_path, 'test.txt'),
                                       dtype=str)
        else:
            self.idx_list = np.loadtxt(os.path.join(self.data_path, 'trainval.txt'),
                                       dtype=str)
        self.idx_list = ['images/' + i for i in self.idx_list]

        if 'IRST640' in self.data_path:
            self.ext = '.bmp'
        else:
            self.ext = '.png'

    def __getitem__(self, index):

        image = Image.open(os.path.join(self.data_path, self.idx_list[index] + self.ext)).convert('L')
        image = image.resize((self.base_size, self.base_size), Image.BILINEAR)
        image = (image - np.min(image)) / (np.max(image) - np.min(image))
        image = np.array(image)
        image = image.reshape(1, image.shape[0], image.shape[1])

        label = image
        input, mask = self.generate_mask(copy.deepcopy(label))

        seg_label = Image.open(os.path.join(self.data_path, self.idx_list[index].replace('images', 'masks') + self.ext)).convert('L')
        seg_label = seg_label.resize((self.base_size, self.base_size), Image.NEAREST)
        seg_label = (seg_label - np.min(seg_label)) / (np.max(seg_label) - np.min(seg_label))
        seg_label = np.array(seg_label)
        seg_label = seg_label.reshape(1, seg_label.shape[0], seg_label.shape[1])

        data = {'label': label, 'input': input, 'mask': mask, 'seg_label': seg_label, 'name': self.idx_list[index]}

        return data

    def __len__(self):
        return len(self.idx_list)

    def generate_mask(self, input):

        ratio = self.ratio
        size_window = self.size_window
        size_data = self.size_data
        num_sample = int(size_data[2] * size_data[1] * (1 - ratio))

        mask = np.ones(size_data)
        output = input

        for ich in range(size_data[0]):
            idy_msk = np.random.randint(0, size_data[2], num_sample)
            idx_msk = np.random.randint(0, size_data[1], num_sample)

            idy_neigh = np.random.randint(-size_window[0] // 2 + size_window[0] % 2,
                                          size_window[0] // 2 + size_window[0] % 2, num_sample)
            idx_neigh = np.random.randint(-size_window[1] // 2 + size_window[1] % 2,
                                          size_window[1] // 2 + size_window[1] % 2, num_sample)

            idy_msk_neigh = idy_msk + idy_neigh
            idx_msk_neigh = idx_msk + idx_neigh

            idy_msk_neigh = idy_msk_neigh + (idy_msk_neigh < 0) * size_data[2] - (idy_msk_neigh >= size_data[2]) * \
                            size_data[2]
            idx_msk_neigh = idx_msk_neigh + (idx_msk_neigh < 0) * size_data[1] - (idx_msk_neigh >= size_data[1]) * \
                            size_data[1]

            id_msk = (ich, idy_msk, idx_msk)
            id_msk_neigh = (ich, idy_msk_neigh, idx_msk_neigh)

            output[id_msk] = input[id_msk_neigh]
            mask[id_msk] = 0.0

        return output, mask


class Test_Loader(Dataset):
    def __init__(self, base_size, data_path, ratio=0.7, size_window=(9, 9)):

        self.base_size = base_size
        self.data_path = 'datasets/test_img/' + data_path
        self.ratio = ratio
        self.size_window = size_window
        self.idx_list = []

        self.idx_list = np.loadtxt(os.path.join(self.data_path, 'imgs.txt'), dtype=str)
        self.idx_list = ['imgs/' + i for i in self.idx_list]
        pass

    def __getitem__(self, index):
        image = Image.open(os.path.join(self.data_path, self.idx_list[index])).convert('L')
        image = image.resize((self.base_size, self.base_size), Image.BILINEAR)
        image = (image - np.min(image)) / (np.max(image) - np.min(image))
        image = np.array(image)
        image = image.reshape(1, image.shape[0], image.shape[1])

        label = image
        input, mask = self.generate_mask(copy.deepcopy(label), image.shape)

        data = {'name': self.idx_list[index], 'image': image, 'label': label, 'input': input, 'mask': mask}

        return data

    def __len__(self):
        return len(self.idx_list)

    def generate_mask(self, input, shape):

        ratio = self.ratio
        size_window = self.size_window
        size_data = shape
        num_sample = int(size_data[1] * size_data[2] * (1 - ratio))

        mask = np.ones(size_data)
        output = input

        for ich in range(size_data[0]):
            idy_msk = np.random.randint(0, size_data[1], num_sample)
            idx_msk = np.random.randint(0, size_data[2], num_sample)

            idy_neigh = np.random.randint(-size_window[0] // 2 + size_window[0] % 2,
                                          size_window[0] // 2 + size_window[0] % 2, num_sample)
            idx_neigh = np.random.randint(-size_window[1] // 2 + size_window[1] % 2,
                                          size_window[1] // 2 + size_window[1] % 2, num_sample)

            idy_msk_neigh = idy_msk + idy_neigh
            idx_msk_neigh = idx_msk + idx_neigh

            idy_msk_neigh = idy_msk_neigh + (idy_msk_neigh < 0) * size_data[1] - (idy_msk_neigh >= size_data[1]) * \
                            size_data[1]
            idx_msk_neigh = idx_msk_neigh + (idx_msk_neigh < 0) * size_data[2] - (idx_msk_neigh >= size_data[2]) * \
                            size_data[2]

            id_msk = (ich, idy_msk, idx_msk)
            id_msk_neigh = (ich, idy_msk_neigh, idx_msk_neigh)

            output[id_msk] = input[id_msk_neigh]
            mask[id_msk] = 0.0

        return output, mask
