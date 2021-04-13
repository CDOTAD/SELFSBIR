import torch.utils.data as data
import torchvision.transforms as transforms
from PIL import Image
from data.dataset_util import find_class
from data.dataset_util import make_dataset
import random
import torchvision.transforms.functional as F


class TripletDataset(data.Dataset):
    def __init__(self, photo_root, sketch_root):
        super(TripletDataset, self).__init__()
        self.photo_paths = sorted(make_dataset(photo_root))
        self.sketch_paths = sorted(make_dataset(sketch_root))
        self.len = len(self.photo_paths)
        self.photo2sketch = self.get_photo_sketch_pair(separator=self.separator)

    def __len__(self):
        return self.len

    def get_photo_sketch_pair(self, separator):
        photo2sketch = dict()
        sk_idx = 0
        sk_len = len(self.sketch_paths)
        for i, photo_name in enumerate(self.photo_paths):
            p_name = photo_name.split('/')[-1]
            p_name = p_name.split('.')[0]

            photo2sketch[i] = []
            while sk_idx < sk_len:
                sk_name = self.sketch_paths[sk_idx].split('/')[-1]
                if sk_name.split(separator)[0] == p_name:
                    photo2sketch[i].append(sk_idx)
                    sk_idx += 1
                else:
                    break
        return photo2sketch

    def get_triplet(self, index):
        return random.choice(self.photo2sketch[index])

    def __getitem__(self, index):
        photo_path = self.photo_paths[index]
        sketch_path = self.sketch_paths[self.get_triplet(index)]

        photo = Image.open(photo_path).convert('RGB')
        sketch = Image.open(sketch_path).convert('RGB')

        if self.transform:
            P = self.transform(photo)
            S = self.transform(sketch)
        else:
            P = self.transform_first(photo)
            S = self.transform_first(sketch)
            if random.random() < 0.5:
                P = F.hflip(P)
                S = F.hflip(S)
            P = self.transform_second(P)
            S = self.transform_second(S)

        cname = photo_path.split('/')[-2]
        if self.class2idx:
            label = self.class2idx[cname]
        else:
            label = index
        return {'P': P, 'S': S, 'L': label}


class SketchyDatabase(TripletDataset):
    def __init__(self, photo_root, sketch_root, transform=None):
        self.separator = '-'
        super(SketchyDatabase, self).__init__(photo_root, sketch_root)
        if transform:
            self.transform = transforms
        else:
            self.transform = transforms.Compose([
                transforms.Resize(224),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
            ])
        classes, class2idx = find_class(photo_root)
        self.classes = classes
        self.class2idx = class2idx


class QUMLDataV1(TripletDataset):
    def __init__(self, photo_root, sketch_root, transform=None):
        self.separator = '.'
        super(QUMLDataV1, self).__init__(photo_root, sketch_root)
        if transform:
            self.transform = transforms
        else:
            self.transform = None
            self.transform_first = transforms.Compose([
                transforms.Resize(256),
                transforms.RandomCrop(224)
            ])

            self.transform_second = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
            ])

        self.class2idx = None


class QUMLDataV2(TripletDataset):
    def __init__(self, photo_root, sketch_root, transform=None):
        self.separator = '_'
        super(QUMLDataV2, self).__init__(photo_root, sketch_root)
        if transform:
            self.transform = transforms
        else:
            self.transform = None
            self.transform_first = transforms.Compose([
                transforms.Resize(256),
                transforms.RandomCrop(224)
                # transforms.RandomResizedCrop(224, scale=(0.2, 1.0))
            ])

            self.transform_second = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
            ])

        self.class2idx = None
