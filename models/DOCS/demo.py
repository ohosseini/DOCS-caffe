import sys
import os
import numpy as np
import matplotlib.pylab as plt
import cv2
import argparse

demo_path = os.path.split(os.path.realpath(__file__))[0]
cdir = os.path.abspath(os.curdir)
os.chdir(demo_path)
sys.path.insert(0,os.path.abspath('../../python/caffe/'))
sys.path.insert(0,os.path.abspath('../../python/'))
os.chdir(cdir)

import caffe

def load_image(filename, bgr_means, input_size=512):
    img = cv2.imread(filename)
    h, w = img.shape[:2]

    if h>=w and h>input_size:
        img= cv2.resize(img,(w * input_size / h, input_size))
        h, w = img.shape[:2]
    elif w>=h and w>input_size:
        img= cv2.resize(img,(input_size, h * input_size / w))
        h, w = img.shape[:2]

    pad_top = (input_size - h)/2
    pad_left = (input_size - w )/2
    pad_bottom = input_size - h - pad_top
    pad_right  = input_size - w  - pad_left

    img_padded = cv2.copyMakeBorder(img,pad_top,pad_bottom,pad_left,pad_right,cv2.BORDER_CONSTANT,value=[0,0,0])
    img_padded = img_padded.astype(np.float32)
    img_padded -= bgr_means

    return img, img_padded, (pad_top,pad_left,pad_bottom,pad_right)

def remove_pad(a, pad):
    return a[pad[0]:a.shape[0]-pad[2], pad[1]:a.shape[1]-pad[3]]

def parse_args():
    parser = argparse.ArgumentParser(description='Deep Object Co-Segmentation (DOCS) Demo: '
						 'Given two input images, segments the common objects within two images.')
    parser.add_argument('gpu', metavar='GPU', type=int, help='gpu-id')
    parser.add_argument('image_a_path', metavar='IMG_A_PATH', help='path to first image.')
    parser.add_argument('image_b_path', metavar='IMG_B_PATH', help='path to second image.')
    parser.add_argument('snapshot', metavar='SNAPSHOT_PATH', help='paht to model\'s snapshot.')
    return parser.parse_args()

def main():
    args = parse_args()

    bgr_means = np.array([104.00699, 116.66877, 122.67892], dtype=np.float32) 

    caffe.set_mode_gpu()
    caffe.set_device(args.gpu)

    net = caffe.Net(demo_path+'/test.prototxt', args.snapshot, caffe.TEST) 

    # load img_a 
    img_a, img_a_padded, pad_a= load_image(args.image_a_path, bgr_means)
        
    # load img_b
    img_b, img_b_padded, pad_b= load_image(args.image_b_path, bgr_means)

    net.blobs['image_a'].data[...] = img_a_padded.transpose((2,0,1))
    net.blobs['image_b'].data[...] = img_b_padded.transpose((2,0,1))

    out = net.forward()

    result_a = remove_pad(out['out_a'][0,1], pad_a)>0.5
    result_b = remove_pad(out['out_b'][0,1], pad_b)>0.5

    filtered_img_a = img_a * np.tile(result_a,(3,1,1)).transpose((1,2,0))
    filtered_img_b = img_b * np.tile(result_b,(3,1,1)).transpose((1,2,0))

    plt.subplot(2,2,1)
    plt.imshow(img_a[:,:,::-1])
    plt.subplot(2,2,2)
    plt.imshow(img_b[:,:,::-1])
    plt.subplot(2,2,3)
    plt.imshow(filtered_img_a[:,:,::-1])
    plt.subplot(2,2,4)
    plt.imshow(filtered_img_b[:,:,::-1])
    plt.show()

if __name__ == '__main__':
    main()
