"""Unility functions for Transformer."""

import math
from typing import Tuple, List


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
