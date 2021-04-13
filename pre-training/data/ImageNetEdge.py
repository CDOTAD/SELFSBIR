import torch.utils.data as data
import torchvision.transforms as transforms
import numpy as np
from PIL import Image
import random
from PIL import ImageFilter
import data.dataset_util as datautil
import torch
import os
import cv2


class GaussianBlur(object):
    def __init__(self, sigma=[.1, 2.]):
        self.sigma = sigma

    def __call__(self, x):
        sigma = random.uniform(self.sigma[0], self.sigma[1])
        x = x.filter(ImageFilter.GaussianBlur(radius=sigma))
        return x


class ImageEdgeMoCo(data.Dataset):
    def __init__(self, image_root, edge_root):
        normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                         std=[0.229, 0.224, 0.225])
        super(ImageEdgeMoCo, self).__init__()

        self.transform = transforms.Compose([
            transforms.RandomResizedCrop(224, scale=(.2, 1.)),
            transforms.RandomApply([
                transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)
            ], p=0.8),
            transforms.RandomGrayscale(p=0.2),
            transforms.RandomApply([GaussianBlur([.1, 2])], p=0.5),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize
        ])
        '''
        self.transform = transforms.Compose([
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor()
        ])
        '''
        classes, class2idx = datautil.find_class(image_root)

        self.image_root = image_root
        self.edge_root = edge_root

        self.img_paths = sorted(datautil.make_dataset(self.image_root))
        # self.edge_paths = sorted(datautil.make_dataset(self.edge_root))
        self.classes = classes
        self.class2idx = class2idx

        self.len = len(self.img_paths)

    def __getitem__(self, index):
        img_path = self.img_paths[index]
        img = Image.open(img_path).convert('RGB')
        
        '''random edge and mixed'''
        # q = self._random_edge_or_mix(mixed_img)
        # k = self._random_edge_or_mix(mixed_img)

        '''random edge'''
        # q = self._random_edge(mixed_img)
        # k = self._random_edge(mixed_img)
        
        '''random mixed'''
        q = self._random_mixed(img)
        k = self._random_mixed(img)

        q = self.transform(img)
        k = self.transform(img)
        qk = torch.cat([q, k], dim=0)
        return qk

    def __len__(self):
        return self.len

    def _random_edge_or_mix(self, img, p=0.5):
        if random.random() > p:
            return img
        
        if random.random() > p:
            return self._canny_dec(img)
        
        return self._random_mixed(img)

    def _canny_dec(self, img):
        img = np.asarray(img)
        img_gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        img_np = cv2.GaussianBlur(img_gray, (5, 5), 0)
        canny = cv2.Canny(img_np, 50, 150)
        canny = 255 - canny
        img_edge = np.stack([canny, canny, canny], axis=2)
        img_edge = img_edge.astype(dtype=np.uint8)
        img_edge = Image.fromarray(img_edge, mode='RGB')
        return img_edge

    def _random_edge(self, img, p=0.5):
        if random.random() > p:
            return img
        return self._canny_dec(img)

    def _random_mixed(self, img, p=0.5):
        h, w = np.shape(img)[:2]
        img_edge = np.asarray(self._canny_dec(img))
        img = np.asarray(img)
        step_h = h // 3
        step_w = w // 3
        mixed_img = np.zeros_like(img, dtype=np.uint8)
        for i in range(3):
            start_x = i * step_h
            end_x = start_x + step_h
            if i == 2:
                end_x = h
            for j in range(3):
                start_y = j * step_w
                end_y = start_y + step_w
                if j == 2:
                    end_y = w

                if random.random() >= p:
                    mixed_img[start_x:end_x, start_y:end_y, :] = img[start_x:end_x, start_y:end_y, :]
                else:
                    mixed_img[start_x:end_x, start_y:end_y, :] = img_edge[start_x:end_x, start_y:end_y, :]

        mixed_img = Image.fromarray(mixed_img, mode='RGB')
        return mixed_img


    '''
    def _random_mixed(self, img_path, p):
        img_paths = img_path.split('/')
        img_name = img_paths[-1].split('.')[0]
        img_class = img_paths[-2]

        img = Image.open(img_path).convert('RGB')
        h, w = np.shape(img)[:2]

        ratio = h / w
        if h < w:
            img = img.resize((int(256. / ratio), 256), Image.ANTIALIAS)
        else:
            img = img.resize((256, int(256 * ratio)), Image.ANTIALIAS)
        img = np.asarray(img)

        candidate_hed_edge = os.path.join(self.edge_root, img_class, img_name + '.mat')

        if os.path.exists(candidate_hed_edge) and random.random() > p:
            img_edge = sio.loadmat(candidate_hed_edge)['edge_predict']
            img_edge = np.round(img_edge)
            img_edge = 1 - img_edge
            img_edge = np.stack([img_edge, img_edge, img_edge], axis=2)
            img_edge = img_edge * 255
            img_edge = img_edge.astype(dtype=np.uint8)
        else:
            img_gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
            img_np = cv2.GaussianBlur(img_gray, (5, 5), 0)
            canny = cv2.Canny(img_np, 50, 150)
            canny = 255 - canny
            img_edge = np.stack([canny, canny, canny], axis=2)
            img_edge = img_edge.astype(dtype=np.uint8)

        h, w = np.shape(img)[:2]

        step_h = h // 3
        step_w = w // 3
        mixed_img = np.zeros_like(img, dtype=np.uint8)

        for i in range(3):
            start_x = i * step_h
            end_x = start_x + step_h
            if i == 2:
                end_x = h
            for j in range(3):
                start_y = j * step_w
                end_y = start_y + step_w
                if j == 2:
                    end_y = w

                if random.random() >= 0.2:
                    mixed_img[start_x:end_x, start_y:end_y, :] = img[start_x:end_x, start_y:end_y, :]
                else:
                    mixed_img[start_x:end_x, start_y:end_y, :] = img_edge[start_x:end_x, start_y:end_y, :]

        mixed_img = Image.fromarray(mixed_img)
        return mixed_img
    '''
















