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


class Sim(BaseImageDataset):
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
        self.include_sim = kwargs['include_sim']
        self.train = []
        self.train = self.load_sim(osp.join(root, self.dataset_dir, 'AIC20_ReID_Simulation', 'train_label.xml'))
        self.query = [[g, int(os.path.basename(g)[:-4]), '0'] for g in sorted(glob.glob(os.path.join(root, self.dataset_dir, 'AIC20_ReID', 'image_query/*')))]
        self.gallery = [[g, int(os.path.basename(g)[:-4]), '0'] for g in sorted(glob.glob(os.path.join(root, self.dataset_dir, 'AIC20_ReID', 'image_test/*')))]

        self.num_train_pids, self.num_train_imgs, self.num_train_cams = self.get_imagedata_info(self.train)
        # self.num_train_pids = 333
        self.num_query_pids, self.num_query_imgs, self.num_query_cams = self.get_imagedata_info(self.query)
        self.num_gallery_pids, self.num_gallery_imgs, self.num_gallery_cams = self.get_imagedata_info(self.gallery)

