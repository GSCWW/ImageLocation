import os
import pdb
import cv2 as cv
from sensed_info import Roi
import numpy as np
import refer_info
import math
import matplotlib.pyplot as plt
from utils import *
from TransParm_Gen import *
# file = './HZ1.JPG'
# data = cv.imread(file)
# print(data)

# a = os.listdir('pic/')
# print(a)

test = glob.glob('./' + '*.JPG')
for image in test:
    print(image)
    # print(os.path.abspath(imag))
    # pdb.set_trace()
    img = cv.imread(image)
    # print(img)
    # pdb.set_trace()
    img = cv.resize(img, (640, 512))
    cv.imwrite(image, img)
    # pdb.set_trace()
print('a')
pdb.set_trace()
# #
#
# #


# img = cv.imread('HZ1.JPG')
# img_r = cv.imread('HZ1-ref.JPG')
# h_sift = detect('SIFT', img, img_r)

H = np.array([[-3.56726153e+00, -5.07909654e+00,  1.10230614e+03],
              [-1.42517930e-01, -9.62185463e+00,  1.02466341e+03],
              [1.59568096e-04, -1.23628729e-02,  1.00000000e+00]])

rect = {}

rect['1'] = dict(pt1=(210, 200),
                 pt2=(225, 224),
                 color=(255, 0, 0),
                 thickness=2)
rect['2'] = dict(pt1=(614, 477),
                 pt2=(630, 498),
                 color=(0, 255, 0),
                 thickness=2)

rect['3'] = dict(pt1=(593, 133),
                 pt2=(610, 155),
                 color=(0, 0, 255),
                 thickness=2)

rect['4'] = dict(pt1=(300, 100),
                 pt2=(350, 155),
                 color=(255, 0, 255),
                 thickness=2)

# print(rect)


img = cv.imread('sensed5.JPG')
for i in rect.values():
    img = cv.rectangle(img, **i)
    cv.imwrite('sensed5.JPG', img)

for defect in rect.values():
    src = np.array((defect['pt1'], defect['pt2']), dtype=float).reshape(-1, 1, 2)
    # print(src)
    dst = cv.perspectiveTransform(src, H).reshape(-1, 2)
    dst = np.around(dst, 0).astype(int)
    # print(dst)
    defect['pt1'] = tuple(dst[0])  # [202.44872647 284.65494903]
    defect['pt2'] = tuple(dst[1])
    # print(defect['pt1'])
    # pdb.set_trace()

print(rect)
# pdb.set_trace()
# src = [[[210, 200], [225, 224]], [[614, 477], [630, 498]], [[593, 133], [610, 155]], [[300, 100], [350, 155]]]


img_r = cv.imread('refer5.JPG')
# img_r = cv.resize(img_r, (640, 512))
for i in rect.values():
    img_r = cv.rectangle(img_r, **i)
    cv.imwrite('refer5.JPG', img_r)

