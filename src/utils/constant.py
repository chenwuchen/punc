
"""Unility functions for Transformer."""

import math
from typing import Tuple, List

USE_ZH_PUNCTUATION = [
    '，',
    '、',
    '。',
    '？',
    '！',
    '；',
]

TOKENIER_CONF = '/mnt/AM4_disk9/chenwuchen/code/github/punc/bertpunc/mdl/bertpunc_conf'

def id_label_convert():
    label2id = {}
    id2label = {}

    for x_id, x in enumerate(['O'] + USE_ZH_PUNCTUATION):
        label = f"S-{x}"
        if x_id == 0:
            label = "O"
        label2id[label] = x_id
        id2label[x_id] = label
    return label2id, id2label

LABEL2ID, ID2LABEL = id_label_convert()
