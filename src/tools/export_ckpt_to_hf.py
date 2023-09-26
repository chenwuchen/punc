import argparse
import os
import torch
from src.model.core import BertPunc


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--ckpt','-i',help='ckpt_path')
    parser.add_argument('--out','-o',help='path to save')
    args = parser.parse_args()

    ModelClass = BertPunc
    
    # pun_bert = ZhprBert(args)
    assert os.path.isfile(args.ckpt)
    # 加载整个模型
    lighting_model = ModelClass.load_from_checkpoint(args.ckpt)
    lighting_model.model.save_pretrained(args.out)
    lighting_model.tokenizer.save_pretrained(args.out)
    print("Done")
