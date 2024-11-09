import torch.utils.data as Data
import torchvision.transforms as transforms

from PIL import Image
import os.path as osp


class PredictImage(Data.Dataset):
    def __init__(self, args, test_dataset):
        base_dir = 'datasets/test_img/test_img_%s/' % test_dataset
        meth = args.meth + '-' + args.train_dataset

        self.dataset = test_dataset
        self.msk_dir = base_dir + 'labels/'
        if test_dataset == 'IRSTD-1k':
            self.ext = '.png'
        else:
            self.ext = '.bmp'
        self.res_dir = base_dir + meth + '/'
        self.list_dir = base_dir + 'imgs.txt'

        self.names = []
        with open(self.list_dir, 'r') as f:
            self.names += [line.strip() for line in f.readlines()]

        self.base_size = args.base_size

    def __getitem__(self, i):
        test_dataset = self.dataset
        name = self.names[i]
        msk_path = osp.join(self.msk_dir, name)
        res_path = osp.join(self.res_dir, name)

        msk = Image.open(msk_path).convert('L')
        w, h = msk.size
        res = Image.open(res_path).convert('L')
        msk, res = self._val_sync_transform(msk, res, w, h)

        msk, res = transforms.ToTensor()(msk), transforms.ToTensor()(res)
        return msk, res, w, h

    def __len__(self):
        return len(self.names)

    def _val_sync_transform(self, mask, result, w, h):
        mask = mask.resize((w, h), Image.NEAREST)
        result = result.resize((w, h), Image.NEAREST)
        return mask, result
