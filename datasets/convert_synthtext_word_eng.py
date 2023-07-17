# *************************************************************************
# Copyright (2022) Bytedance Inc.
#
# Copyright (2022) oCLIP Authors 
#
# Licensed under the Apache License, Version 2.0 (the "License"); 
# you may not use this file except in compliance with the License. 
# You may obtain a copy of the License at 
#
#     http://www.apache.org/licenses/LICENSE-2.0 
#
# Unless required by applicable law or agreed to in writing, software 
# distributed under the License is distributed on an "AS IS" BASIS, 
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. 
# See the License for the specific language governing permissions and 
# limitations under the License. 
# *************************************************************************

import os
import scipy.io, scipy.ndimage
import torchvision
import random
import pickle
from tqdm import tqdm
import argparse
import sys

def synthtext(path):
    pass

def synthtext_curve1(path):
    pass

def synthtext_curve2(path):
    pass

def recog_indices_to_str(recog_indices, chars):
    recog_str = []
    for idx in recog_indices:
        if idx < len(chars):
            recog_str.append(chars[idx])
        else:
            break 
    return ''.join(recog_str)

class SynthText_curve(torchvision.datasets.CocoDetection):

    def __init__(self, root, annFile, chars):
        super().__init__(root, annFile)
        self.root = root
        self.chars = chars

    def __getitem__(self, index):
        image, anno = super().__getitem__(index)
        anno = [ele for ele in anno if 'iscrowd' not in anno or ele['iscrowd'] == 0]

        recog = [ele['rec'] for ele in anno]
        random.shuffle(recog)

        image_file_name = self.coco.loadImgs(self.ids[index])[0]['file_name']

        recog_strs = []
        for recog_idx in recog:

            recog_strs.append(recog_indices_to_str(recog_idx, self.chars))

        return image_file_name, recog_strs
        

if __name__ == "__main__":

    parser = argparse.ArgumentParser("Prepare SynthText Annotation for oCLIP Pre-training")
    # parser.add_argument("--data_dir", type=str, default='/home/wll/w4/datasets/SynthText_curve/emcs_imgs')
    # parser.add_argument("--anno_dir", type=str, default='/home/wll/w4/datasets/SynthText_curve/annotations/ecms_v1_maxlen25.json')
    parser.add_argument("--data_dir", type=str, default='/home/wll/w4/datasets/synthtext_curve/syntext_word_eng')
    parser.add_argument("--anno_dir", type=str, default='/home/wll/w4/datasets/synthtext_curve/syntext_word_eng.json')
    # parser.add_argument("--save_dir", type=str, default='./data/SynthText_cureve_1')
    parser.add_argument("--save_dir", type=str, default='./data/SynthText_cureve_2')
    parser.add_argument('--chars', type=str, default=' !"#$%&\'()*+,-./0123456789:;<=>?@ABCDEFGHIJKLMNOPQRSTUVWXYZ[\\]^_`abcdefghijklmnopqrstuvwxyz{|}~')
    args = parser.parse_args()

    synthtext_dataset = SynthText_curve(args.data_dir, args.anno_dir, args.chars)

    if not os.path.isdir(os.path.join(args.save_dir)):
        os.makedirs(os.path.join(args.save_dir))

    i_range = range(len(synthtext_dataset))

    print('num_images: %d'%len(i_range))

    data_out = ['filepath\ttitle\n']

    for i,im_idx in tqdm(enumerate(i_range)):
        i_name = os.path.join(args.data_dir, str(synthtext_dataset[im_idx][0]))
        i_txt = synthtext_dataset[im_idx][1]
        
        word_list = '\n'.join(i_txt)
        word_list = word_list.split()
        random.shuffle(word_list)

        data_out += ['{}\t{}\n'.format(i_name, ' '.join(word_list))]


    with open(os.path.join(args.save_dir, 'train_char.csv'), 'w') as f:
        f.writelines(data_out)

    char_list = list(range(33, 127))
    with open(os.path.join(args.save_dir, 'char_dict'), 'wb') as f:
        pickle.dump(char_list, f)
