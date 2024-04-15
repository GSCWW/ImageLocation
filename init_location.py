# -*- coding: utf-8 -*-
"""
    @Created on 2021/11/23
    @function: initial location of UAV images in the DOM.
    @versions: 1.4
    @author: GSCWW
"""

import pdb
import cv2 as cv
import sensed_info
import argparse
import refer_info
import logging
import traceback
import os

import utils
from utils import *
import sys
import numpy as np

logger = logger('location')


def init_match(dom_file, images, roi_dom_path, cfg):
    """
    crop the initial roi in the DOM according to the info extracted from UAV images and DOM
    :param dom_file: DOM complete path
    :param images: UAV images list
    :param roi_dom_path: path to save the cropped dom images
    :param cfg: rectification parameters for UAV cameras
    :return: GeoTransform of the DOM and top_left corner world coordinates(utm) of cropped DOM images
    """

    roi_info = {}

    # load basic parameters from DOM
    dataset, GeoTransform, XSize, YSize, bands = refer_info.tiff_read(dom_file)

    for image in images:
        try:
            fname = os.path.basename(image)
            outfile = os.path.join(roi_dom_path, fname)

            # output utm coordinates of the sensed image
            roi = sensed_info.Roi(image, cfg)
            roi, utm_zone, utm_band = roi.init_locate()
            x_cord, y_cord = roi[:, 0], roi[:, 1]
            coord = (x_cord, y_cord)

            # the top-left world coordinates of the cropped DOM
            roi_info[fname] = np.min(x_cord), np.max(y_cord), utm_zone, utm_band  # tl_x, tl_y

            # find roi in the reference image
            refer_roi = refer_info.InitRoi(dataset, GeoTransform, bands, coord)

            # save the roi and top-left coordinates in the reference image
            img = refer_roi.tiff2img()
            cv.imwrite(outfile, img)

            logger.info('%s initial region of interests found!' % image)

        except Exception:
            logger.info('%s initial region of interests not found!' % image)
            logger.error(traceback.format_exc())
    return GeoTransform, roi_info


if __name__ == '__main__':
    # from glob import glob
    # test = ['HuZhou/20210421170605000Z.JPG', 'HuZhou/20210421170622000Z.JPG', 'HuZhou/20210421170634000Z.JPG']
    test = data_read('TongLiao1')
    # pdb.set_trace()
    cfg = utils.data_read('cfg/rectification.yaml')
    roi_info = init_match('result.tif', test, 'initmatch/TongLiao1', cfg)
    print(roi_info)

    # roi_info:
    # ((226190.94737, 0.03586, 0.0, 3408561.2751800003, 0.0, -0.03586),
    # {'20210421170605000Z.JPG': (226863.0811476989, 3407758.7457681694, 51, 'R'),
    #  '20210421170622000Z.JPG': (226826.239324108, 3407759.029107784, 51, 'R'),
    #  '20210421170634000Z.JPG': (226786.77407198498, 3407759.7816543262, 51, 'R')})
