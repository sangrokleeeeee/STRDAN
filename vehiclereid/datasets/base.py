from __future__ import absolute_import
from __future__ import print_function

import os.path as osp
import xml.etree.ElementTree as ET


def degree_to_categories(degree):
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


def load_sim(path, prefix):
    # name_info, id_img = dict(), defaultdict(list)
    encoding = 'utf-8'# if 'Sim' in paths else 'gb2312'
    # xmlp = ET.XMLParser(encoding=encoding)
    # f = ET.parse('a.xml',parser=xmlp)
    ids = {}
    results = []
    root = ET.parse(path)
    print(path)
    print('prefix: ', prefix)
    for item in root.find('Items').findall('Item'):
        attr = item.attrib
        if prefix + int(attr['vehicleID']) not in ids:
            ids[prefix + int(attr['vehicleID'])] = len(ids)
        results.append(
            [osp.join('/'.join(path.split('/')[:-1]), 'image_train', attr['imageName']),
            ids[prefix + int(attr['vehicleID'])] + prefix,
            int(attr['cameraID'][1:])]
        )
            # cam id, color, type, orientation
        results[-1][-1] = [results[-1][-1], int(attr['colorID']), degree_to_categories(float(attr['orientation'])), int(attr['typeID'])]

    return results


class BaseDataset(object):
    """
    Base class of reid dataset
    """

    def __init__(self, root):
        self.root = osp.expanduser(root)

    def load_sim(self, path, prefix):
        # compute real data pids
        self.num_train_pids = len(set([f[1] for f in self.train]))
        
        self.train += load_sim(path, num_of_pids)
        for t in self.train:
            if not isinstance(t[-1], list):
                t[-1] = [t[-1], -1, -1, -1]

    def get_imagedata_info(self, data):
        pids, cams = [], []
        if isinstance(data[-1][-1], list):
            is_train = True
            for _, pid, (camid, _, _, _) in data:
                pids += [pid]
                cams += [camid]
        else:
            is_train = False
            for _, pid, camid in data:
                pids += [pid]
                cams += [camid]
        pids = set(pids)
        cams = set(cams)
        num_pids = len(pids)
        num_cams = len(cams)
        num_imgs = len(data)
        if is_train and hasattr(self, num_train_pids):
            num_pids = self.num_train_pids
        return num_pids, num_imgs, num_cams

    def print_dataset_statistics(self):
        raise NotImplementedError


class BaseImageDataset(BaseDataset):
    """
    Base class of image reid dataset
    """

    def print_dataset_statistics(self, train, query, gallery):
        num_train_pids, num_train_imgs, num_train_cams = self.get_imagedata_info(train)
        num_query_pids, num_query_imgs, num_query_cams = self.get_imagedata_info(query)
        num_gallery_pids, num_gallery_imgs, num_gallery_cams = self.get_imagedata_info(gallery)

        print('Image Dataset statistics:')
        print('  ----------------------------------------')
        print('  subset   | # ids | # images | # cameras')
        print('  ----------------------------------------')
        print('  train    | {:5d} | {:8d} | {:9d}'.format(num_train_pids, num_train_imgs, num_train_cams))
        print('  query    | {:5d} | {:8d} | {:9d}'.format(num_query_pids, num_query_imgs, num_query_cams))
        print('  gallery  | {:5d} | {:8d} | {:9d}'.format(num_gallery_pids, num_gallery_imgs, num_gallery_cams))
        print('  ----------------------------------------')
