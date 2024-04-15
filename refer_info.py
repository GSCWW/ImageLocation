# -*- coding: utf-8 -*-
"""
    @Created on 2021/11/23
    @function: Locate the the ROI region according to world coordinates(utm).
    @versions: 1.4
    @author: GSCWW
"""
import pdb

import gdal
from osgeo import gdal_array
import numpy as np
# import pdb
# import os
import sys
import cv2 as cv
# os.environ['PROJ_LIB'] = '/home/gscww/anaconda3/envs/rectify/share/proj'


def tiff_read(in_file):
    dataset = gdal.Open(in_file)
    if dataset is None:
        print('Could not open ' + in_file)
        sys.exit(1)
    XSize = dataset.RasterXSize  # 网格的X轴像素数量
    YSize = dataset.RasterYSize  # 网格的Y轴像素数量
    bands = dataset.RasterCount  # 波段数
    GeoTransform = dataset.GetGeoTransform()  # 投影转换信息
    return dataset, GeoTransform, XSize, YSize, bands

# def get_lon_lat(tl_x, tl_y, coord, gtf):
#     """
#     get the longitude and latitude
#     """
#     XSize, YSize = coord
#     x_range = range(0, XSize)
#     y_range = range(0, YSize)
#     x, y = np.meshgrid(x_range, y_range)
#     lon = tl_x + x * gtf[1] + y * gtf[2]
#     lat = tl_y + x * gtf[4] + y * gtf[5]
#     return lon, lat

class InitRoi:
    """
    crop the roi in the DOM according to the roi_dom corners world coordinates(utm)
    """
    def __init__(self, dataset, GeoTransform, bands, coord):
        self.dataset = dataset  # Tiff或者ENVI文件
        self.GeoTransform = GeoTransform
        self.bands = bands
        self.coord = coord

    def get_pixel_offset(self):
        # get the pixel coordinates according to the utm coordinates
        x, y = self.coord
        gtf = self.GeoTransform
        # xOffset = int((x-gtf[0]) / gtf[1])
        xOffset = (x * gtf[5] - y * gtf[2] + gtf[2] * gtf[3] - gtf[0] * gtf[5])\
                  / (gtf[1] * gtf[5] - gtf[2] * gtf[4])
        # yOffset = int((y-gtf[3]) / gtf[5])
        yOffset = (x * gtf[4] - y * gtf[1] + gtf[1] * gtf[3] - gtf[0] * gtf[4])\
                  / (gtf[2] * gtf[4] - gtf[1] * gtf[5])

        return xOffset, yOffset

    def tiff2img(self):
        x, y = self.get_pixel_offset()
        # original region:
        tl_x_or, br_x_or = int(round(np.min(x))), int(round(np.max(x)))
        tl_y_or, br_y_or = int(round(np.min(y))), int(round(np.max(y)))
        width_or = br_x_or - tl_x_or
        height_or = br_y_or - tl_y_or

        # expand region
        # ratio = 0.05
        # width_add = int(width_or * ratio)
        # height_add = int(height_or * ratio)
        # tl_x, br_x = tl_x_or - width_add, br_x_or + width_add
        # tl_y, br_y = tl_y_or - height_add, br_y_or + height_add
        # width = br_x - tl_x
        # height = br_y - tl_y

        # according to the top_left corner pixel coordinates and pixel size to crop the roi in the DOM
        image = np.zeros((height_or, width_or, 3))
        # cv.namedWindow('img', cv.WINDOW_NORMAL | cv.WINDOW_KEEPRATIO)
        for band in range(self.bands-1):
            data = self.dataset.GetRasterBand(band+1)
            image[:, :, band] = data.ReadAsArray(tl_x_or, tl_y_or, width_or, height_or)
        image = cv.cvtColor(image.astype(np.uint8), cv.COLOR_BGR2RGB)

        return image


if __name__ == '__main__':
    import utils
    import sensed_info
    image = './HuZhou/20210421170605000Z.JPG'
    # image = 'HZ1-ref.JPG'
    test = cv.imread(image)
    # print(test.shape)
    # pdb.set_trace()
    in_file = "trans.tif"

    cfg = utils.data_read('cfg/rectification.yaml')
    roi = sensed_info.Roi(image, cfg)
    roi = roi.init_locate()
    print(roi)
    # pdb.set_trace()
    # x_cord, y_cord = roi[:, 0], roi[:, 1]
    # roi = np.array([[226859.31525625, 3407726.81057009],
    #       [226860.15581088, 3407762.10779697],
    #       [226907.21803267, 3407760.95762105],
    #       [226906.33295282, 3407725.66148234]])
    print('roi:', roi)
    x_cord, y_cord = roi[0][:, 0], roi[0][:, 1]
    print('x_cord:', x_cord)
    print('y_cord:', y_cord)
    coord = (x_cord, y_cord)
    # pdb.set_trace()
    dataset, GeoTransform, XSize, YSize, bands = tiff_read(in_file)
    image = InitRoi(dataset, GeoTransform, bands, coord).tiff2img()
    cv.namedWindow('test', cv.WINDOW_NORMAL | cv.WINDOW_KEEPRATIO)
    cv.imshow('test', image)
    if cv.waitKey(0) & 0xff == 27:
        cv.destroyAllWindows()



