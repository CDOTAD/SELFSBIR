import scipy.io as sio
import sys
import numpy as np
from PIL import Image
import os

os.environ['CUDA_VISIBLE_DEVICES'] = '0'

print('================================')
print('load caffe')
caffe_root =  '/home/zzl/caffe'
sys.path.insert(0, caffe_root+'/python')
import caffe
print('load caffe successfully')

HED_NET = '/home/zzl/project/hed/examples/hed/deploy.prototxt'
HED_MODEL = '/home/zzl/project/hed/examples/hed/hed_pretrained_bsds.caffemodel'

caffe.set_mode_gpu()
caffe.set_device(0)
print('===============================')
print('load caffe mode')
net = caffe.Net(HED_NET, HED_MODEL, caffe.TEST)
print('load caffe mode successfully')
IMG_DIR = '/home/zzl/ImageNet/ImageNet/ILSVRC2012'

class_name = sorted(os.listdir(IMG_DIR))

border = 112
print('==============================')
print('start extract edge maps')
SAVE_PATH = '/home/zzl/ImageNet/ImageNet/image_hed'
with open('./abandon.txt', 'w') as f:
    for i, cname in enumerate(class_name):
        print(i, cname)
        CLASS_PATH = os.path.join(IMG_DIR, cname)
        CLASS_SAVE_PATH = os.path.join(SAVE_PATH, cname)
        if not os.path.exists(CLASS_SAVE_PATH):
            os.mkdir(CLASS_SAVE_PATH)

        photo_lists = sorted(os.listdir(CLASS_PATH))
        for photo_name in photo_lists:
            #print(photo_name)
            save_name = photo_name.split('.')[0]
            save_path = os.path.join(CLASS_SAVE_PATH, save_name+'.mat')
            if os.path.exists(save_path):
                continue
            IMG_PATH = os.path.join(CLASS_PATH, photo_name)
            im = Image.open(IMG_PATH)
            if len(np.shape(im)) == 2:
                continue
            h, w, c = np.shape(im)
            ratio = float(h)/float(w)
            if ratio > 5 or ratio < 0.2:
                f.writelines(IMG_PATH+'\n')
                continue
            if h < w:
                im = im.resize((int(256./ratio), 256), Image.ANTIALIAS)
            else:
                im = im.resize((256, int(256.*ratio)), Image.ANTIALIAS)
            print(photo_name)
            print(np.shape(im))

            in_ = np.array(im, dtype=np.float32)
            if len(np.shape(in_)) == 2:
                print('gray')
            in_ = np.pad(in_, ((border, border), (border, border), (0,0)), 'reflect')

            in_ = in_[:,:,0:3]
            in_ = in_[:,:,::-1]
            in_ -= np.array((104.00698793, 116.66876762, 122.67891434))
            in_ = in_.transpose((2, 0, 1))

            net.blobs['data'].reshape(1, *in_.shape)
            net.blobs['data'].data[...]=in_

            net.forward()

            fuse = net.blobs['sigmoid-fuse'].data[0][0,:,:]
            fuse = fuse[(border+35):(-border+35), (border+35):(-border+35)]
            sio.savemat(save_path, {'edge_predict':fuse})

