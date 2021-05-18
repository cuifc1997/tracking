from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import pycocotools.coco as coco
from pycocotools.cocoeval import COCOeval
import numpy as np
import json
import os
from collections import defaultdict
from ..generic_dataset import GenericDataset

import glob
import os
import motmetrics as mm
from collections import OrderedDict
from pathlib import Path

def compare_dataframes(gts, ts):
    accs = []
    names = []
    for k, tsacc in ts.items():
        if k in gts:
            accs.append(mm.utils.compare_to_groundtruth(gts[k], tsacc, 'iou', distth=0.5))
            names.append(k)
        else:
            print('No ground truth for {}, skipping.'.format(k))

    return accs, names

def find_mota(groundtruths, tests, gt_type):
    # groundtruths = '../data/mot17/train/'
    # tests = '../exp/tracking/mot17_06/results_mot17halfval/'
    # gt_type = '_val_half'
    eval_official = True
    fmt = 'mot15-2D'

    gtfiles = glob.glob(os.path.join(groundtruths, '*/gt/gt{}.txt'.format(gt_type)))
    tsfiles = [f for f in glob.glob(os.path.join(tests, '*.txt')) if not os.path.basename(f).startswith('eval')]

    # print(gtfiles)
    # print(tsfiles)

    gt = OrderedDict([(Path(f).parts[-3], mm.io.loadtxt(f, fmt=fmt, min_confidence=1)) for f in gtfiles])
    ts = OrderedDict([(os.path.splitext(Path(f).parts[-1])[0], mm.io.loadtxt(f, fmt=fmt)) for f in tsfiles])

    mh = mm.metrics.create()
    accs, names = compare_dataframes(gt, ts)

    # metrics = ['recall', 'precision', 'num_unique_objects', 'mostly_tracked', \
    #            'partially_tracked', 'mostly_lost', 'num_false_positives', 'num_misses', \
    #            'num_switches', 'num_fragmentations', 'mota', 'motp', 'num_objects']

    metrics = mm.metrics.motchallenge_metrics + ['num_objects']
    summary = mh.compute_many(
        accs, names=names,
        metrics=metrics, generate_overall=True)
    # summary = summary.round(4)
    mota = summary.loc['OVERALL', 'mota']
    motp = summary.loc['OVERALL', 'motp']
    return mota, motp

class MOT(GenericDataset):
    num_categories = 1
    default_resolution = [544, 960]
    class_name = ['']
    max_objs = 256
    cat_ids = {1: 1, -1: -1}
    def __init__(self, opt, split):
        self.dataset_version = opt.dataset_version
        self.year = int(self.dataset_version[:2])
        print('Using MOT {} {}'.format(self.year, self.dataset_version))
        data_dir = os.path.join(opt.data_dir, 'mot{}'.format(self.year))

        if opt.dataset_version in ['17trainval', '17test']:
            ann_file = '{}.json'.format('train' if split == 'train' else \
                                            'test')
        elif opt.dataset_version == '17halftrain':
            ann_file = '{}.json'.format('train_half')
        elif opt.dataset_version == '17halfval':
            ann_file = '{}.json'.format('val_half')
        img_dir = os.path.join(data_dir, '{}'.format(
            'test' if 'test' in self.dataset_version else 'train'))

        print('ann_file', ann_file)
        ann_path = os.path.join(data_dir, 'annotations', ann_file)

        self.images = None
        # load image list and coco
        super(MOT, self).__init__(opt, split, ann_path, img_dir)

        self.num_samples = len(self.images)
        print('Loaded MOT {} {} {} samples'.format(
            self.dataset_version, split, self.num_samples))

    def _to_float(self, x):
        return float("{:.2f}".format(x))

    def __len__(self):
        return self.num_samples

    def save_results(self, results, save_dir):
        results_dir = os.path.join(save_dir, 'results_mot{}'.format(self.dataset_version))
        if not os.path.exists(results_dir):
            os.mkdir(results_dir)
        for video in self.coco.dataset['videos']:
            video_id = video['id']
            file_name = video['file_name']
            out_path = os.path.join(results_dir, '{}.txt'.format(file_name))
            f = open(out_path, 'w')
            images = self.video_to_images[video_id]
            tracks = defaultdict(list)
            for image_info in images:
                if not (image_info['id'] in results):
                    continue
                result = results[image_info['id']]
                frame_id = image_info['frame_id']
                for item in result:
                    if not ('tracking_id' in item):
                        item['tracking_id'] = np.random.randint(100000)
                    if item['active'] == 0:
                        continue
                    tracking_id = item['tracking_id']
                    bbox = item['bbox']
                    bbox = [bbox[0], bbox[1], bbox[2], bbox[3]]
                    tracks[tracking_id].append([frame_id] + bbox)
            rename_track_id = 0
            for track_id in sorted(tracks):
                rename_track_id += 1
                for t in tracks[track_id]:
                    f.write('{},{},{:.2f},{:.2f},{:.2f},{:.2f},-1,-1,-1,-1\n'.format(
                        t[0], rename_track_id, t[1], t[2], t[3]-t[1], t[4]-t[2]))
            f.close()
        # print('save finish')

    def run_eval(self, results, data_dir, save_dir, only_mota=None):
        self.save_results(results, save_dir)
        gt_type_str = '{}'.format(
            '_train_half' if '17halftrain' in self.opt.dataset_version \
                else '_val_half' if '17halfval' in self.opt.dataset_version \
                else '')
        gt_type_str = '_val_half' if self.year in [16, 19] else gt_type_str
        gt_type_str = '--gt_type {}'.format(gt_type_str) if gt_type_str != '' else \
            ''
        if only_mota is None:
            os.system('python tools/eval_motchallenge.py ' + \
                      '../data/mot{}/{}/ '.format(self.year, 'train') + \
                      '{}/results_mot{}/ '.format(save_dir, self.dataset_version) + \
                      gt_type_str + ' --eval_official')
            print(gt_type_str)
            print('../data/mot{}/{}/ '.format(self.year, 'train'))
            print('{}/results_mot{}/ '.format(save_dir, self.dataset_version))
        else:
            groundtruths = '{}/mot{}/{}/'.format(data_dir, self.year, 'train')
            tests = '{}/results_mot17halfval/'.format(save_dir)
            gt_type = '_val_half'
            print(groundtruths)
            print(tests)
            print(gt_type)
            cur_mota, cur_motp = find_mota(groundtruths, tests, gt_type)

            return cur_mota, cur_motp



