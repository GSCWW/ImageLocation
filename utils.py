# -*- coding: utf-8 -*-
"""
    @Created on 2021/10/22
    @function: utilization tools for defects location in the DOM.
    @versions: 2.1
    @author: GSCWW
"""

import logging
import os
import yaml
import json
from glob import glob


def logger(name):
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    log_name = os.path.dirname(os.path.abspath(__file__)) + f'/{name}.log'
    logfile = log_name
    fh = logging.FileHandler(logfile, mode='w')
    fh.setLevel(logging.INFO)
    formatter = logging.Formatter("%(asctime)s - %(filename)s[line:%(lineno)d] - %(levelname)s: %(message)s")
    fh.setFormatter(formatter)
    logger.addHandler(fh)
    return logger


def data_read(infile):
    input_path_extension = infile.split('.')[-1]
    if input_path_extension in ['jpg', 'jpeg', 'JPG', 'png']:
        data = [infile]
    elif input_path_extension == "txt":
        with open(infile, "r") as f:
            data = f.read().splitlines()
    elif input_path_extension == 'yaml':
        with open(infile, 'r', encoding='utf-8') as f:
            data = yaml.load(f, Loader=yaml.FullLoader)
    elif input_path_extension == 'json':
        with open(infile, 'r') as f:
            data = json.load(f)
    else:
        data = glob(
            os.path.join(infile, "*.jpg")) + \
               glob(os.path.join(infile, "*.png")) + \
               glob(os.path.join(infile, "*.jpeg")) + \
               glob(os.path.join(infile, "*Z.JPG"))
    if data == None:
        raise ValueError("file not found, plz make sure there is inputfile")
    else:
        return data


def data_write(file, data):
    input_path_extension = file.split('.')[-1]
    if input_path_extension == 'yaml':
        with open(file, 'w', encoding='utf-8') as f:
            yaml.dump(data, f)
    elif input_path_extension == 'json':
        with open(file, 'w') as f:
            json.dump(data, f, indent=4)
    else:
        raise TypeError("file is not supported, plz input data with yaml or json extension")


def config_path(conf_name):
    dir_path = os.path.dirname(os.path.abspath(__file__))
    conf_path = os.path.join(dir_path, 'cfg', conf_name)
    if not os.path.isfile(conf_path):
        raise FileNotFoundError(f'{conf_name} not found')
    return conf_path
