from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import glob
import re
import sys
import urllib
import tarfile
import zipfile
import os.path as osp
from scipy.io import loadmat
import numpy as np
import h5py
from scipy.misc import imsave
import math
import xml.etree.ElementTree as ET

from .base import BaseImageDataset


class Aicity(BaseImageDataset):
    """
    Market1501

    Reference:
    Zheng et al. Scalable Person Re-identification: A Benchmark. ICCV 2015.

    URL: http://www.liangzheng.org/Project/project_reid.html

    Dataset statistics:
    # identities: 1501 (+1 for background)
    # images: 12936 (train) + 3368 (query) + 15913 (gallery)
    """
    dataset_dir = 'AIC20_track2'

    def __init__(self, root='datasets', verbose=True, **kwargs):
        super().__init__(root)
        self.n_image = 4
        self.include_sim = kwargs['include_sim']
        self.dataset_path = osp.join(root, self.dataset_dir, 'AIC20_ReID', 'train_label.xml')
        # if self.include_sim:
        #     self.dataset_path += [osp.join(root, self.dataset_dir, 'AIC20_ReID_Simulation', 'train_label.xml')]

        self.train = self._load_label(self.dataset_path)

        if self.include_sim:
            self.load_sim(osp.join(root, self.dataset_dir, 'AIC20_ReID_Simulation', 'train_label.xml'))
        self.query = [[g, int(os.path.basename(g)[:-4]), '0'] for g in sorted(glob.glob(os.path.join(root, self.dataset_dir, 'AIC20_ReID', 'image_query/*')))]
        self.gallery = [[g, int(os.path.basename(g)[:-4]), '0'] for g in sorted(glob.glob(os.path.join(root, self.dataset_dir, 'AIC20_ReID', 'image_test/*')))]

        self.num_train_pids, self.num_train_imgs, self.num_train_cams = self.get_imagedata_info(self.train)
        self.num_query_pids, self.num_query_imgs, self.num_query_cams = self.get_imagedata_info(self.query)
        self.num_gallery_pids, self.num_gallery_imgs, self.num_gallery_cams = self.get_imagedata_info(self.gallery)
    
    def _load_label(self, path):
        # name_info, id_img = dict(), defaultdict(list)
        # encoding = 'utf-8' if 'Sim' in paths else 'gb2312'
        # xmlp = ET.XMLParser(encoding=encoding)
        # f = ET.parse('a.xml',parser=xmlp)
        ids = {}
        results = []
        root = ET.parse(path)
        print(path)
        # print('prefix: ', prefix)
        for item in root.find('Items').findall('Item'):
            attr = item.attrib
            if int(attr['vehicleID']) not in ids:
                ids[int(attr['vehicleID'])] = len(ids)
            results.append(
                [osp.join('/'.join(path.split('/')[:-1]), 'image_train', attr['imageName']),
                ids[int(attr['vehicleID'])],
                int(attr['cameraID'][1:])]
            )
            results[-1][-1] = [results[-1][-1], -1, -1, -1]
                # ids.append(prefix + int(attr['vehicleID']))
        
        # ids = sorted(ids)

                # print(prefix + int(attr['vehicleID']) - 1)
                # name_info[attr['imageName']] = attr
                # id_img[prefix + attr['vehicleID']].append(attr['imageName'])

        return results

    def degree_to_categories(self, degree):
        if degree <= 60:
            return 0
        elif degree <= 120:
            return 1
        elif degree <= 180:
            return 2
        elif degree <= 240:
            return 3
        elif degree <= 300:
            return 4
        else:
            return 5
