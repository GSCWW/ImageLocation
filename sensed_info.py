# -*- coding: utf-8 -*-
"""
    @Created on 2021/11/23
    @function: transform from picture pixel coordinates to world coordinates.
    @versions: 1.4
    @author: GSCWW
"""


import pdb
import re
import exifread
import requests
import utm
import argparse
import math
import glob
import os
import numpy as np
import sys
import utils

#
# class ExifInfo():
#
#     def __init__(self, photo_path):
#         self.photo_path = photo_path
#
#     def get_tags(self):
#         """获取照片信息"""
#         image_content = open(self.photo_path, 'rb')
#         tags = exifread.process_file(image_content)
#
#         if tags is None:
#             print('Could not open ' + self.photo_path)
#             sys.exit(1)
#
#         image_content.close()
#         return tags
#
#     def get_lng_lat(self):
#         """经纬度转换"""
#         tags = self.get_tags()
#         try:
#             # 纬度
#             LatRef = tags["GPS GPSLatitudeRef"].printable
#             Lat = tags["GPS GPSLatitude"].printable[1:-1].replace(" ", "").replace("/", ",").split(",")
#             Lat = float(Lat[0]) + float(Lat[1]) / 60 + float(Lat[2]) / 36000000
#             if LatRef != "N":
#                 Lat = Lat * (-1)
#             # 经度
#             LonRef = tags["GPS GPSLongitudeRef"].printable
#             Lon = tags["GPS GPSLongitude"].printable[1:-1].replace(" ", "").replace("/", ",").split(",")
#             Lon = float(Lon[0]) + float(Lon[1]) / 60 + float(Lon[2]) / 36000000
#             if LonRef != "E":
#                 Lon = Lon * (-1)
#             return Lat, Lon
#         except:
#             print('Unable to get')
#

class CoordConvert:
    """
    transform of world coordinate between utm and wgs84
    """

    def __init__(self, coord):
        self.coord = coord

    def wgs842utm(self):
        import utm
        lat, lon = self.coord
        utm_ = utm.from_latlon(lat, lon)
        utm_x = utm_[0]
        utm_y = utm_[1]
        utm_zone = utm_[2]
        utm_band = utm_[3]
        return utm_x, utm_y, utm_zone, utm_band

    def utm2wgs84(self):
        utm_x, utm_y, utm_zone, utm_band = self.coord
        lat, lon = utm.to_latlon(utm_x, utm_y, utm_zone, utm_band)
        return lat, lon


class Roi:

    def __init__(self, image, config, dfov=None):
        self.image = image
        self.config = config
        # self.dfov = dfov
        self.model, self.yawdegree, self.rolldegree, self.pitchdegree, self.lat, self.lon, self.altitude, self.xdimension, self.ydimension = self.xmp_info()

    def xmp_info(self):
        """
        read the relevant information
        """
        import re
        from libxmp import XMPFiles, consts
        try:
            xmpfile = XMPFiles(file_path=self.image, open_forupdate=True)
            xmp = xmpfile.get_xmp()
            YawDegree = float(re.findall(r'<drone-dji:GimbalYawDegree>(.*)</drone-dji:GimbalYawDegree>', str(xmp))[0])
            RollDegree = float(
                re.findall(r'<drone-dji:GimbalRollDegree>(.*)</drone-dji:GimbalRollDegree>', str(xmp))[0])
            PitchDegree = float(
                re.findall(r'<drone-dji:GimbalPitchDegree>(.*)</drone-dji:GimbalPitchDegree', str(xmp))[0])
            Lat = float(re.findall(r'<drone-dji:GpsLatitude>(.*)</drone-dji:GpsLatitude>', str(xmp))[0])
            Lon = float(re.findall(r'<drone-dji:GpsLongitude>(.*)</drone-dji:GpsLongitude>', str(xmp))[0])
            Alt = float(re.findall(r'<drone-dji:RelativeAltitude>(.*)</drone-dji:RelativeAltitude>', str(xmp))[0])
            XDimension = float(re.findall(r'<exif:PixelXDimension>(.*)</exif:PixelXDimension>', str(xmp))[0])
            YDimension = float(re.findall(r'<exif:PixelYDimension>(.*)</exif:PixelYDimension>', str(xmp))[0])
            model = re.findall(r'<tiff:Model>(.*)</tiff:Model>', str(xmp))[0]
            # print('euler', YawDegree, RollDegree, PitchDegree)
            return model, YawDegree, RollDegree, PitchDegree, Lat, Lon, Alt, XDimension, YDimension

        except:
            print('Unable to get xmp information')

    def pixel_coord(self):
        """
        pixel_coordinates ,x_z(E_N)plane
        """
        # get the retification roi coordinates
        model = self.model
        roi = self.config[model]['visual']['roi']

        tl = [roi[0], self.ydimension-roi[1]]
        br = [roi[0]+roi[2], self.ydimension-roi[1]-roi[3]]
        tr = [roi[0]+roi[2], self.ydimension-roi[1]]
        bl = [roi[0], self.ydimension-roi[1]-roi[3]]

        # get the orginal roi coordinates
        # bl = [0, 0]
        # tl = [0, self.ydimension]
        # tr = [self.xdimension, self.ydimension]
        # br = [self.xdimension, 0]
        # print('pixel_original_coordinates:', bl, tl, tr, br)

        return bl, tl, tr, br

    def camera_matrix(self):

        """
        load the camera matrix from camera-coordinates to pixel-coordinates
        """

        model = self.model
        mtx = np.array(self.config[model]['visual']['mtx'])
        dist = np.array(self.config[model]['visual']['dist'])

        # transform camera_matrix from x_y coordinates to x_z coordinates
        mtx[[1, 2], :] = mtx[[2, 1], :]
        mtx[:, [1, 2]] = mtx[:, [2, 1]]

        return mtx, dist

    def cal_rotatran(self):

        """
        Calculation the rotation and translation matrix from world-coordinates to camera-coordinates
        """

        # euler = YawDegree, PitchDegree, RollDegree = 1.4（北东), -89.9(北地）, 0
        # 东北天坐标系（x,y,z),  YawDegree-- -z, PitchDegree-- x, RollDegree-- y
        # 云台姿态角即使用大地坐标系（NED，北东地坐标系）描述云台上负载设备的角度，该角度也称为欧拉角。

        euler = list(map(math.radians, (self.yawdegree, self.pitchdegree, self.rolldegree)))

        R_x = np.array([[1, 0, 0],
                        [0, math.cos(euler[1]), math.sin(euler[1])],
                        [0, -math.sin(euler[1]), math.cos(euler[1])]
                        ])

        R_y = np.array([[math.cos(euler[2]), 0, math.sin(euler[2])],
                        [0, 1, 0],
                        [-math.sin(euler[2]), 0, math.cos(euler[2])]
                        ])

        R_z = np.array([[math.cos(euler[0]), -math.sin(euler[0]), 0],
                        [math.sin(euler[0]), math.cos(euler[0]), 0],
                        [0, 0, 1]
                        ])

        R = np.dot(R_y, np.dot(R_x, R_z))
        Rt = R.T

        center = np.array([0, 0, self.altitude])
        # pc - R*pw = t, and here pc = [0, 0, 0]
        t = -np.dot(R, center)
        t = np.array(t)

        return R, Rt, t

    # def cam_area(self):
    #     '''
    #     calculate the coordinates of init ROI corners in the camera coordinate system
    #     '''
    #
    #     diagonal = 2 * math.tan(math.radians(self.dfov / 2)) * self.altitude
    #     ratio = self.xdimension / self.ydimension
    #     height = (diagonal ** 2 / (ratio ** 2 + 1)) ** 0.5
    #     width = ratio * height
    #
    #     # coordinates of corners in the X-Z plane of camera coordinates
    #
    #     tl = [-width / 2, height / 2]
    #     tr = [width / 2, height / 2]
    #     bl = [-width / 2, -height / 2]
    #     br = [width / 2, -height / 2]
    #
    #     # caluculate the Y values(altitudes) of corners in camera coordinates
    #     # pw = Rt *（pc - t)
    #
    #     R, Rt, t = self.cal_rotatran()
    #     tl_alt = -(Rt[2, 0] * (tl[0] - t[0]) + Rt[2, 2] * (tl[1] - t[2])) / Rt[2, 1] + t[1]
    #     tr_alt = -(Rt[2, 0] * (tr[0] - t[0]) + Rt[2, 2] * (tr[1] - t[2])) / Rt[2, 1] + t[1]
    #     bl_alt = -(Rt[2, 0] * (bl[0] - t[0]) + Rt[2, 2] * (bl[1] - t[2])) / Rt[2, 1] + t[1]
    #     br_alt = -(Rt[2, 0] * (br[0] - t[0]) + Rt[2, 2] * (br[1] - t[2])) / Rt[2, 1] + t[1]
    #
    #     tl.insert(1, tl_alt)
    #     tr.insert(1, tr_alt)
    #     bl.insert(1, bl_alt)
    #     br.insert(1, br_alt)
    #
    #     # print('center,tl, tr, bl, br:',center,tl, tr, bl, br)
    #     # return the camera-relative coordinates of four corners
    #     return tl, tr, bl, br

    def pixel2world(self):
        """

        Transform from pixel-coordinates to world-coordinates

        alt*P = K*(R*Pw+t)

        P: pixel coordinates
        K:camera matrix
        alt: altitude
        R: rotaton matrix
        t: translation matrix
        altitude need to to calculated firstly, in the x_z plane, it's y value in camera coordinates

        according to:
        Yc*p = K*Pc = K*(R*Pw + t)
        Pw = Rt *（pc - t) = Rt *（pc - t) = Rt *（Yc*k_1*p - t)

        thereby:
        Yc = (Pw + Rt*t)/Rt*k_1*p(Rt = R_1) and for altitude, Pw = 0
        """

        # altitude calculation
        bl, tl, tr, br = self.pixel_coord()
        bl.insert(1, 1), tl.insert(1, 1), tr.insert(1, 1), br.insert(1, 1)
        p = np.array([bl, tl, tr, br]).T
        R, Rt, t = self.cal_rotatran()
        k, _ = self.camera_matrix()
        K_1 = np.linalg.inv(k)
        mat1 = np.dot(Rt, t)
        mat2 = np.dot(np.dot(Rt, K_1), p)
        alt = mat1[2] / mat2[2]

        # world-coordinates transform
        Pc = alt*np.matmul(K_1, p)

        Pw = np.dot(Rt, (Pc.T - t).T)
        return Pw


    def init_locate(self):

        # utm coordinates of camera center in the world coordinates

        coord = self.lat, self.lon
        utm_x, utm_y, utm_zone, utm_band = CoordConvert(coord).wgs842utm()

        center = np.array([utm_x, utm_y])

        tl, br, tr, bl = self.pixel_coord()
        roi = self.pixel2world().T  # (longitude, latitude, height) relative to center
        roi = np.add(roi[:, :-1], center)
        # print("roi:", roi)

        # return roi, center
        return roi, utm_zone, utm_band

    # def init_match_noeuler(self):
    #     coord = self.lat, self.lon
    #     utm_x, utm_y, utm_zone, utm_band = CoordConvert(coord).wgs842utm()
    #     tl, tr, bl, br = self.cam_area()
    #     tl = tl[0] + utm_x, tl[2] + utm_y
    #     br = br[0] + utm_x, br[2] + utm_y
    #     print(tl, br)
    #
    #     return tl, br

    # def init_match_or(self):
    #     # utm coordinates of camera center in the world coordinates
    #     coord = self.lat, self.lon
    #     utm_x, utm_y, utm_zone, utm_band = CoordConvert(coord).wgs842utm()
    #     print('center:', utm_x, utm_y)
    #     R, Rt, t = self.cal_rotatran()
    #     tl, tr, bl, br = self.cam_area()
    #
    #     # pw = Rt *（pc - t)
    #     area = np.array((tl - t, tr - t, bl - t, br - t))
    #     roi = np.dot(Rt, area.T).T
    #     tl = np.min(roi[:, 0]) + utm_x, np.max(roi[:, 1]) + utm_y
    #     br = np.max(roi[:, 0]) + utm_x, np.min(roi[:, 1]) + utm_y
    #
    #     # return the top-left corner coordinates(tl) and bottom-right corner coordinates(br)
    #     return tl, br


if __name__ == '__main__':
    import utils

    # model = 'ZH20T'
    image = 'HuZhou/20210421170605000Z.JPG'
    cfg = utils.data_read('cfg/rectification.yaml')
    roi = Roi(image, cfg)
    roi = roi.init_locate()
    print('*' * 50)
    print('roi:', roi)
    x_cord, y_cord = roi[0][:, 0], roi[0][:, 1]
    tl_x, tl_y = np.min(x_cord), np.max(y_cord)

    coord = (x_cord, y_cord)
    print(coord)


