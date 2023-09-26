#!/usr/bin/python
# -*- encoding: utf-8 -*-
########################################################################
#                                                                       
# Author: chenwuchen@baidu.com                                          
# Date: 2023/08/20 18:19:16 
#                                                                       
########################################################################

import sys
import os
from torch.utils.data import DataLoader
from src.dataset.dataset  import PunctStreamDataset
import multiprocessing


def test_dataloader():
    dataset = PunctStreamDataset('/mnt/AM4_disk9/chenwuchen/code/github/punc/data/train_data/lst/all_data_230822_train.json')
    dataloader = DataLoader(dataset, batch_size=24, num_workers=3)
    for sample in dataloader:
        print(sample)
        pass


def main(args):
    argv1 = sys.argv[1]
    argv2 = sys.argv[2]


if __name__ == '__main__':
    # main(sys.argv[1:])
    test_dataloader()
