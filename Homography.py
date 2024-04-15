# -*- coding: utf-8 -*-
"""
    @Created on 2021/11/23
    @function: Calculate the homography matrix from the rectified UAV image to the cropped DOM image.
    @versions: 1.4
    @author: GSCWW
"""
import pdb
from glob import glob

# os.chdir("..")
from copy import deepcopy

# import os
# os.environ["CUDA_VISIBLE_DEVICES"] = "0"

import torch
import cv2 as cv
import numpy as np
from src.loftr import LoFTR, default_cfg

# The default config uses dual-softmax.
# The outdoor and indoor models share the same config.
# You can change the default values like thr and coarse_match_type.


def load_networks(weights):
    matcher = LoFTR(config=default_cfg)
    matcher.load_state_dict(torch.load(weights)['state_dict'])
    matcher = matcher.eval().cuda()
    return matcher

def homography(sensed_image, refer_image, matcher):
    imgsensed_raw = cv.imread(sensed_image, cv.IMREAD_GRAYSCALE)
    imgref_or = cv.imread(refer_image, cv.IMREAD_GRAYSCALE)

    imgsensed_raw = cv.resize(imgsensed_raw, (640, 512))
    imgref_raw = cv.resize(imgref_or, (640, 512))

    imgsensed_raw = cv.resize(imgsensed_raw,
                              (imgsensed_raw.shape[1] // 8 * 8,
                               imgsensed_raw.shape[0] // 8 * 8))  # input size shuold be divisible by 8
    imgref_raw = cv.resize(imgref_raw,
                           (imgref_raw.shape[1] // 8 * 8, imgref_raw.shape[0] // 8 * 8))

    imgsensed = torch.from_numpy(imgsensed_raw)[None][None].cuda() / 255.
    imgref = torch.from_numpy(imgref_raw)[None][None].cuda() / 255.

    batch = {'image0': imgsensed, 'image1': imgref}

    # Inference with LoFTR and get prediction
    with torch.no_grad():
        matcher(batch)
        mkpts0 = batch['mkpts0_f'].cpu().numpy()
        mkpts1 = batch['mkpts1_f'].cpu().numpy()
        mconf = batch['mconf'].cpu().numpy()

    # mkpts0 = mkpts0[np.where(mconf >= 0.8)]
    # mkpts1 = mkpts1[np.where(mconf >= 0.8)]

    mkpts1 = coordconvert(mkpts1, imgref_raw, imgref_or)

    if len(mkpts0) > 4:
        H, _ = cv.findHomography(mkpts0, mkpts1, cv.RANSAC, ransacReprojThreshold=3.0)
        return H
    else:
        raise AttributeError('not enough match points a found')


def coordconvert(src, img_src, img_dst):
    '''
    convert the coordinates in the DOM with size (640, 512) to the initial DOM size
    '''
    h_dst, w_dst = img_dst.shape[:2]
    h_src, w_src = img_src.shape[:2]
    ratio_w = w_dst / w_src
    ratio_h = h_dst / h_src
    ratio = np.array([ratio_w, ratio_h])
    dst = src * ratio
    return dst


if __name__ == '__main__':

    WEIGHTS = "/home/gscww/Desktop/HomographyNet/LoFTR/weights/outdoor_ds.ckpt"
    weights = WEIGHTS
    sensed_dir = "./"
    refer_dir = "./"
    matcher = load_networks(weights)
    images = glob(sensed_dir + '/*.JPG')
    for image in images:
        # fname = os.path.basename(image)
        # refer_image = os.path.join(refer_dir, fname)
        sensed_image = "TongLiao1-rect/DJI_20210414123147_0014_Z.JPG"
        refer_image = "initmatch/TongLiao1/DJI_20210414123147_0014_Z.JPG"
        H = homography(sensed_image, refer_image, matcher)
        print(H)
        pdb.set_trace()

        imgsensed_raw = cv.imread(sensed_image)
        imgsensed_raw = cv.resize(imgsensed_raw, (640, 512))
        img = cv.warpPerspective(imgsensed_raw, H, (640, 512))
        cv.imshow('img', img)
        # cv.imwrite('test.jpg', img)
        if cv.waitKey(0) & 0xff == 27:
            cv.destroyAllWindows()
        # pdb.set_trace()