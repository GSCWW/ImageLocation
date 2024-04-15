# -*- coding: utf-8 -*-
"""
    @Created on 2021/11/23
    @function: Process of locate the precise defect corner world coordinates(wsg84) in the DOM.
    @versions: 1.0
    @author: GSCWW
"""

from final_match import *

class DefectLocat:

    def __init__(self, infile, outfile, dom, defects_path):
        self.infile = infile
        self.outfile = outfile
        self.dom = dom
        self.defects_path = defects_path

    def prepare(self):

        # load .json UAV images
        images_info = data_read(self.infile)
        images_path = images_info['images']
        images = [image['visual'] for image in images_path]

        # load defects_path with defects relative information
        defects_info = data_read(self.defects_path)
        defects = defects_info['defects']
        defects_pixel_coords = {}

        # extract the images name and defects information from the defects_result .json file
        for image_info in defects:
            defects_pixel_coords[image_info['images']] = image_info['defect']

        # load configure file of UAV camera
        cfg = config_path('rectification.yaml')
        cfg = data_read(cfg)

        # preload model for finding homography
        weights = config_path('outdoor_ds.ckpt')
        matcher = load_networks(weights)

        return images, defects_pixel_coords, cfg, matcher

    def locate(self):

        images, defects_pixel_coords, cfg, matcher = self.prepare()

        roi_info = init_match(self.dom, images, self.outfile, cfg)

        images_coords = final_match(defects_pixel_coords, roi_info, matcher, self.outfile)

        return images_coords



if __name__ == '__main__':

    infile = './test/input.json'
    outfile = './test/initmatch'
    dom = 'trans.tif'
    defects_path ='./test/output.json'

    defects_location = DefectLocat(infile, outfile, dom, defects_path)
    defects = defects_location.locate()
    for i, j in defects.items():
        print(i)
        print(j)
        print('*'*30)





