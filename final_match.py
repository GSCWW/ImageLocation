# -*- coding: utf-8 -*-
"""
    @Created on 2021/11/23
    @function: Locate the precise defect corner world coordinates(wsg84) in the DOM.
    @versions: 1.0
    @author: GSCWW
"""

from Homography import *
from sensed_info import *
from init_location import *


def final_match(defects_input, roi_info, matcher, dom_path):

    # cropped DOM image and top left world coordinates(utm) generated from init_match
    reference, tl_coord = [], []
    for image, coord in roi_info[1].items():
        reference.append(image)
        tl_coord.append(coord)

    # GeoTransform(gtf) from DOM
    gtf = roi_info[0]
    images_coords = {}

    for image, defect in defects_input.items():
        fname = os.path.basename(image)
        if fname in reference:
            pts = {}
            tl_x, tl_y, utm_zone, utm_band = tl_coord[reference.index(fname)]
            refer_image = os.path.join(dom_path, fname)
            H = homography(image, refer_image, matcher)
            for i in defect:
                tl = [i['tlx'], i['tly']]
                br = [i['brx'], i['bry']]
                tr = [i['brx'], i['tly']]
                bl = [i['tlx'], i['bry']]
                src = [tl, br, tr, bl]

                # transform of defect from UAV pixel coordinates to crop_dom pixel_coordinates
                dst = transform(src, H)

                # from pixel_coordinates to world_coordinates(utm)
                utm = get_lon_lat(gtf, dst, tl_x, tl_y)
                world_coords = []

                # utm to wgs84 coordinates transform for four-corner coordinates of every defect
                for coord in utm:
                    utm_coord = coord[0], coord[1], utm_zone, utm_band
                    world_coord = CoordConvert(utm_coord).utm2wgs84()
                    world_coords.append(world_coord)

                # add defect type info
                pts[i['class']] = world_coords
            # add the image info(images basename)
            images_coords[fname] = pts

    return images_coords


def transform(src, H):
    '''
    coordinates of defects in the cropped DOM
    '''

    src = np.array(src, dtype=float).reshape(-1, 1, 2)
    dst = cv.perspectiveTransform(src, H).reshape(-1, 2)

    return dst


def get_lon_lat(gtf, dst, tl_x, tl_y):
    """
    get the longitude and latitude
    """
    XSize, YSize = dst[:, 0], dst[:, 1]

    lon = tl_x + XSize * gtf[1] + YSize * gtf[2]
    lat = tl_y + XSize * gtf[4] + YSize * gtf[5]

    utm = np.hstack((lon[:, np.newaxis], lat[:, np.newaxis]))

    return utm
    # return lon, lat


if __name__ == '__main__':

    matcher = load_networks(weights)

    defects_pixel_coords = {'./HuZhou-rect/20210421170605000Z.JPG': [
        {'brx': 409, 'bry': 268, 'class': 0, 'conf': 0.9241147041320801, 'rot': 0, 'rotx': 390, 'roty': 259, 'tlx': 372,
         'tly': 251},
        {'brx': 409, 'bry': 268, 'class': 1, 'conf': 0.9241147041320801, 'rot': 0, 'rotx': 390, 'roty': 259, 'tlx': 371,
         'tly': 251}],
        './HuZhou-rect/20210421170622000Z.JPG': [
        {'brx': 409, 'bry': 268, 'class': 0, 'conf': 0.9241147041320801, 'rot': 0, 'rotx': 390, 'roty': 259, 'tlx': 372,
         'tly': 251},
        {'brx': 409, 'bry': 268, 'class': 0, 'conf': 0.9241147041320801, 'rot': 0, 'rotx': 390, 'roty': 259, 'tlx': 371,
         'tly': 251}],
        './HuZhou-rect/20210421170634000Z.JPG': [
        {'brx': 409, 'bry': 268, 'class': 0, 'conf': 0.9241147041320801, 'rot': 0, 'rotx': 390, 'roty': 259, 'tlx': 372,
         'tly': 251},
        {'brx': 409, 'bry': 268, 'class': 0, 'conf': 0.9241147041320801, 'rot': 0, 'rotx': 390, 'roty': 259, 'tlx': 372,
         'tly': 251}],
        './HuZhou-rect/202104230.JPG': [
        {'brx': 409, 'bry': 268, 'class': 0, 'conf': 0.9241147041320801, 'rot': 0, 'rotx': 390, 'roty': 259, 'tlx': 372,
         'tly': 251},
        {'brx': 409, 'bry': 268, 'class': 0, 'conf': 0.9241147041320801, 'rot': 0, 'rotx': 390, 'roty': 259, 'tlx': 372,
         'tly': 251}]}

    roi_info = ((226190.94737, 0.03586, 0.0, 3408561.2751800003, 0.0, -0.03586),
                {'20210421170605000Z.JPG': (226863.0811476989, 3407758.7457681694, 51, 'R'),
                 '20210421170622000Z.JPG': (226826.239324108, 3407759.029107784, 51, 'R'),
                 '20210421170634000Z.JPG': (226786.77407198498, 3407759.7816543262, 51, 'R')})

    outfile = './initmatch/HuZhou'
    
    images_defects = final_match(defects_pixel_coords, roi_info, matcher, outfile)
    print(images_defects)

    # images_defects:
    # {'20210421170605000Z.JPG': {0: [(30.771302439603172, 120.14649238122442), (30.771293257075254, 120.14651483355337),
    #                                 (30.77130230111551, 120.14651481151157), (30.771293366307212, 120.14649233357116)],
    #                             1: [(30.771302443358614, 120.14649177296293), (30.771293257075254, 120.14651483355337),
    #                                 (30.77130230111551, 120.14651481151157), (30.771293369269312, 120.14649172341335)]},
    #  '20210421170622000Z.JPG': {0: [(30.771296529223864, 120.14610717707905), (30.771287343011213, 120.1461302376989),
    #                                 (30.77129638704915, 120.14613021562113), (30.77128745513667, 120.1461071275656)]},
    #  '20210421170634000Z.JPG': {0: [(30.771294235745376, 120.14569568380644), (30.771285053360195, 120.14571813619773),
    #                                 (30.771294097395685, 120.14571811408138), (30.77128516245375, 120.14569563622808)]},
    #  '202104230.JPG': {0: [(30.771294235745376, 120.14569568380644), (30.771285053360195, 120.14571813619773),
    #                        (30.771294097395685, 120.14571811408138), (30.77128516245375, 120.14569563622808)]}}
