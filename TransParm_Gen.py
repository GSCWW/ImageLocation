import pdb
import cv2 as cv
import numpy as np
import glob
import json
# from tqdm import tqdm
import time
import argparse
import os, shutil, sys

parser = argparse.ArgumentParser(description='Generation of intrinsic matrix and distortion coefficients')
parser.add_argument('--ImgAlign', type=str, default='HZ1.JPG', help='the datasets of calibration target used for rectification')
parser.add_argument('--ImgRef', type=str, default='HZ1-ref.JPG', help='the datasets of calibration target used for rectification')
args = parser.parse_args()

def Read2Gray(img):
    img = cv.imread(img)
    img = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    img = cv.resize(img, (1280, 1040))
    return img

def detect(key, Imga, Imgr):
    # descriptor.type=float, cv.FlannBasedMatcher and cv.BFMatcher can be used
    ImgA = cv.cvtColor(Imga, cv.COLOR_BGR2GRAY)
    ImgR = cv.cvtColor(Imgr, cv.COLOR_BGR2GRAY)
    MAX_MATCHES = 200
    GOOD_MATCH_PERCENT = 0.5

    if key == 'SIFT':
        sift = cv.SIFT_create(MAX_MATCHES)
        kts1, descriptors1 = sift.detectAndCompute(ImgA, None)
        kts2, descriptors2 = sift.detectAndCompute(ImgR, None)

        matcher = cv.BFMatcher()
        matcher = matcher.knnMatch(descriptors1, descriptors2, k=2)
        matches = []
        for m, n in matcher:
            if m.distance < 0.9 * n.distance:
                matches.append(m)

    if key == 'ORB':
        orb = cv.ORB_create(MAX_MATCHES)
        kts1, descriptors1 = orb.detectAndCompute(ImgA, None)
        kts2, descriptors2 = orb.detectAndCompute(ImgR, None)

        matcher = cv.DescriptorMatcher_create(cv.DESCRIPTOR_MATCHER_BRUTEFORCE_HAMMING)
        matches = matcher.match(descriptors1, descriptors2, None)
        matches.sort(key=lambda x: x.distance, reverse=False)
        numGoodMatches = int(len(matches) * GOOD_MATCH_PERCENT)
        matches = matches[:numGoodMatches]

    # imMatches = cv.drawMatches(ImgA, kts1, ImgR, kts2, matches, None)
    imMatches = cv.drawMatches(Imgr, kts2, Imga, kts1, matches, None)

    # if len(matches) >= 4:
        # src_pts = np.float32([kts1[m.queryIdx].pt for m in matches]).reshape(-1, 1, 2)
        # print(src_pts)
        # pdb.set_trace()
        # dst_pts = np.float32([kts2[m.trainIdx].pt for m in matches]).reshape(-1, 1, 2)
    # H, _ = cv.findHomography(src_pts, dst_pts, cv.RANSAC, 2)
    cv.namedWindow('match', cv.WINDOW_NORMAL | cv.WINDOW_KEEPRATIO)
    cv.imshow("match", imMatches)
    if cv.waitKey(0) & 0xff == 27:
        cv.destroyAllWindows()
    # cv.imwrite("matches.jpg", imMatches)
    # print(H)
    # return H

if __name__ == '__main__' :
    img = cv.imread(args.ImgAlign)
    ImgR = cv.imread(args.ImgRef)
    H = detect('SIFT', img, ImgR)
    print(H)



