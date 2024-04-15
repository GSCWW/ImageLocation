""""opencv template match to match the image and DOM region"""
import pdb

import cv2 as cv
import numpy as np


def rotation(img):
    img = cv.imread(img)
    (h, w) = img.shape[:2]
    center = (w //2, h //2)
    M = cv.getRotationMatrix2D(center, -3.0, 1)
    dst = cv.warpAffine(img, M, (w, h))
    dst = dst[30:480, 20:600]
    # cv.imwrite('rotated.jpg', dst)
    # cv.imshow('rotate', dst)
    # if cv.waitKey(0) & 0xff(27):
        # cv.destroyAllWindows()
    return dst


def template_demo(tpl):
    # tpl = cv.imread("HZ1.JPG")  #模板图像
    tpl = cv.resize(tpl, (360, 270))
    target = cv.imread("InitMatch_visual.jpg")#原图像
    # cv.imshow("template image", tpl)
    # cv.imshow("target image", target)
    methods = [cv.TM_SQDIFF_NORMED, cv.TM_CCORR_NORMED, cv.TM_CCOEFF_NORMED] #标准平方差匹配 ，标准相关匹配，标准相关系数匹配
    th, tw = tpl.shape[:2]  #模板的高宽
    for md in methods:
       # print(md)
        result = cv.matchTemplate(target, tpl, md)   #像素点的相关度量值
        min_val, max_val, min_loc, max_loc = cv.minMaxLoc(result) #寻找匹配最值（大小和位置）
        if md == cv.TM_SQDIFF_NORMED:
            tl = min_loc
        else:
            tl = max_loc
        br = (tl[0]+tw, tl[1]+th);  #确定匹配区域
        cv.rectangle(target, tl, br, (0, 0, 255), 2)#将匹配区域绘制到原图上
        img = cv.resize(target, (620, 514))
        cv.imwrite('temp.jpg', img)
        cv.imshow("match-"+np.str_(md), target)
        cv.waitKey(0)
        # pdb.set_trace()
       # cv.imshow("match-" + np.str(md), result)

if __name__ == '__main__':
    img = "HZ1.JPG"
    dst = rotation(img)
    template_demo(dst)
# src = cv.imread("HZ1-ref.JPG")
# cv.namedWindow("input image", cv.WINDOW_AUTOSIZE)
# cv.imshow("input image", src)
template_demo()
# cv.waitKey(0)

# cv.destroyAllWindows()
