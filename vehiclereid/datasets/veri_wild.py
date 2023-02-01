from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import random
import os.path as osp

from .base import BaseImageDataset
from collections import defaultdict


class VeriWild(BaseImageDataset):
    """
    VehicleID

    Reference:
    @inproceedings{liu2016deep,
    title={Deep Relative Distance Learning: Tell the Difference Between Similar Vehicles},
    author={Liu, Hongye and Tian, Yonghong and Wang, Yaowei and Pang, Lu and Huang, Tiejun},
    booktitle={Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition},
    pages={2167--2175},
    year={2016}}

    Dataset statistics:
    # train_list: 13164 vehicles for model training
    # test_list_800: 800 vehicles for model testing(small test set in paper
    # test_list_1600: 1600 vehicles for model testing(medium test set in paper
    # test_list_2400: 2400 vehicles for model testing(large test set in paper
    # test_list_3200: 3200 vehicles for model testing
    # test_list_6000: 6000 vehicles for model testing
    # test_list_13164: 13164 vehicles for model testing
    """
    dataset_dir = 'veri-wild'

    def __init__(self, root='datasets', verbose=True, test_size=3000, **kwargs):
        super(VeriWild, self).__init__(root)
        self.dataset_dir = osp.join(self.root, self.dataset_dir)
        self.img_dir = osp.join(self.dataset_dir, 'images')
        # self.split_dir = osp.join(self.dataset_dir, 'train_test_split')
        self.train_list = osp.join(self.dataset_dir, 'train_list.txt')
        self.test_size = test_size

        if self.test_size == 3000:
            self.test_query = osp.join(self.dataset_dir, 'test_3000_query.txt')
            self.test_list = osp.join(self.dataset_dir, 'test_3000.txt')
        elif self.test_size == 5000:
            self.test_query = osp.join(self.dataset_dir, 'test_5000_query.txt')
            self.test_list = osp.join(self.dataset_dir, 'test_5000.txt')
        elif self.test_size == 10000:
            self.test_query = osp.join(self.dataset_dir, 'test_10000_query.txt')
            self.test_list = osp.join(self.dataset_dir, 'test_10000.txt')

        # print(self.test_list)

        train, query, gallery = self.process_split(relabel=True)
        self.train = train
        self.query = query
        self.gallery = gallery

        self.include_sim = kwargs['include_sim']
        if self.include_sim:
            self.load_sim(osp.join(root, 'AIC20_track2', 'AIC20_ReID_Simulation', 'train_label.xml'))

        if verbose:
            print('=> Veri-wild loaded')
            self.print_dataset_statistics(train, query, gallery)
        # print(self.query)
        self.num_train_pids, self.num_train_imgs, self.num_train_cams = self.get_imagedata_info(self.train)
        self.num_query_pids, self.num_query_imgs, self.num_query_cams = self.get_imagedata_info(self.query)
        self.num_gallery_pids, self.num_gallery_imgs, self.num_gallery_cams = self.get_imagedata_info(self.gallery)

    def check_before_run(self):
        """Check if all files are available before going deeper"""
        if not osp.exists(self.dataset_dir):
            raise RuntimeError('"{}" is not available'.format(self.dataset_dir))
        if not osp.exists(self.split_dir):
            raise RuntimeError('"{}" is not available'.format(self.split_dir))
        if not osp.exists(self.train_list):
            raise RuntimeError('"{}" is not available'.format(self.train_list))
        # if self.test_size not in [800, 1600, 2400]:
        #     raise RuntimeError('"{}" is not available'.format(self.test_size))
        if not osp.exists(self.test_list):
            raise RuntimeError('"{}" is not available'.format(self.test_list))

    def get_pid2label(self, pids):
        pid_container = set(pids)
        pid2label = {pid: label for label, pid in enumerate(pid_container)}
        return pid2label

    def parse_img_pids(self, nl_pairs, pid2label=None):
        # il_pair is the pairs of img name and label
        output = []
        for info in nl_pairs:
            name = info[0]
            pid = info[1]
            if pid2label is not None:
                ori_pid = pid
                pid = pid2label[pid]
            else:
                ori_pid = pid
                
            camid = 1  # don't have camid information use 1 for all
            img_path = osp.join(self.img_dir, str(ori_pid).zfill(5), name+'.jpg')
            output.append([img_path, pid, camid])
        return output

    def process_split(self, relabel=False):
        # read train paths
        train_pid_dict = defaultdict(list)

        # 'train_list.txt' format:
        # the first number is the number of image
        # the second number is the id of vehicle
        with open(self.train_list) as f_train:
            train_data = f_train.readlines()
            for data in train_data:
                pid, name = data.strip().split('/')
                pid = int(pid)
                train_pid_dict[pid].append([name, pid])
        train_pids = list(train_pid_dict.keys())
        num_train_pids = len(train_pids)

        print('num of train ids: {}'.format(num_train_pids))
        test_pid_dict = defaultdict(list)
        with open(self.test_list) as f_test:
            test_data = f_test.readlines()
            for data in test_data:
                pid, name = data.strip().split('/')
                test_pid_dict[pid].append([name, pid])
        test_pids = list(test_pid_dict.keys())
        num_test_pids = len(test_pids)

        query_pid_dict = defaultdict(list)
        with open(self.test_query) as f_test:
            test_data = f_test.readlines()
            for data in test_data:
                pid, name = data.strip().split('/')
                query_pid_dict[pid].append([name, pid])
        query_pids = list(query_pid_dict.keys())
        num_query_pids = len(query_pids)

        assert num_test_pids == num_query_pids, 'The number of IDs in Query and Test are different'

        train_data = []
        query_data = []
        gallery_data = []

        # for train ids, all images are used in the train set.
        for pid in train_pids:
            imginfo = train_pid_dict[pid]  # imginfo include image name and id
            train_data.extend(imginfo)

        for pid in test_pids:
            imginfo = test_pid_dict[pid]  # imginfo include image name and id
            gallery_data.extend(imginfo)

        for pid in query_pids:
            imginfo = query_pid_dict[pid]  # imginfo include image name and id
            query_data.extend(imginfo)

        # for each test id, random choose one image for gallery
        # and the other ones for query.

        if relabel:
            train_pid2label = self.get_pid2label(train_pids)
        else:
            train_pid2label = None
        # for key, value in train_pid2label.items():
        #     print('{key}:{value}'.format(key=key, value=value))

        train = self.parse_img_pids(train_data, train_pid2label)
        query = self.parse_img_pids(query_data)
        gallery = self.parse_img_pids(gallery_data)
        return train, query, gallery

