import pdb

import cv2 as cv
import numpy as np
import json
import argparse
import glob
import os

parser = argparse.ArgumentParser(description='rectification of images datasets')
parser.add_argument('--weights', type=str, default='h20.json', help='the weights file of cameraMatrix and distortion coefficients')
parser.add_argument('--datasets', type=str, default='m2ea-v', help='the datasets need rectification')
parser.add_argument('--output', type=str, default='m2ea-trans', help='the datasets need rectification')
args = parser.parse_args()

if not os.path.exists(args.output):
    os.mkdir(args.output)

with open(args.weights, 'r', encoding='utf-8-sig') as f:
    weights = json.load(f)
    h = np.array(weights['homography'])


#
# im_src = cv.imread('DJI_0111.JPG')
# im_dst = cv.imread('DJI_0112.JPG')
# im_out = cv.warpPerspective(im_src, h, (im_dst.shape[1], im_dst.shape[0]))
# images = glob.glob(args.datasets + '/*.JPG')
#
# for fname in images:
#     basename = os.path.basename(fname)
#     im_src = cv.imread(fname)
#     im_out = cv.warpPerspective(im_src, h, (610, 478))
#     cv.imwrite(args.output + '/' + basename, im_out)
#



# with open('data/raw_rect_h20t_tl.json', 'r', encoding='utf-8-sig') as f:
#     data = json.load(f)
#     data_in = np.array(data["000003"])
#
# print(data_in[0])
#
#
#
# with open('data/dst_rect_h20t_tl.json', 'r', encoding='utf-8-sig') as f:
#     data = json.load(f)
#     data_out = np.array(data["000003"])

# print(data_out[0])

h = np.linalg.inv(h)
print(h)
k = np.array([9, 57, 1])
a = np.float32([[[9, 57]]])

# a = np.array([[9, 57]]).reshape(-1, 1, 2)
print(a)
#
dst = cv.perspectiveTransform(a, h)
print(dst)
#
# c = [[13], [58], [1]]
#
# b = h * k
# d = np.sum(b, axis=1)
#
# x = d[0] / d[2]
# y = d[1] / d[2]
# print(x, y)
# #
# print(d)
# print(b)
# print(b[0])


