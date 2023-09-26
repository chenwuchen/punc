import argparse
import os
import sys

import torch
from torch.utils.data import DataLoader
from src.model.core import ZhprBert
from src.dataset.dataset import DocumentDataset, merge_stride, decode_pred
from transformers import AutoModelForTokenClassification, AutoTokenizer
import onnx
import onnxruntime

def prepare_data(text):
    window_size = 256
    step = 200
    dataset = DocumentDataset(text, window_size=window_size, step=step)
    dataloader = DataLoader(dataset=dataset, shuffle=False, batch_size=5)
    for batch in dataloader:
        inputs = {
            'input_ids': batch[0],
            'attention_mask':batch[1],
        }
        return inputs

def test_onnx(model_path, inputs):
    providers=['CPUExecutionProvider']
    # providers=['CUDAExecutionProvider']
    # onnx_model = onnx.load("model.onnx")
    
    ort_session = onnxruntime.InferenceSession(model_path + "/model.onnx", providers=providers)
    ort_inputs = {ort_session.get_inputs()[0].name: inputs['input_ids'].cpu().numpy(), 
                ort_session.get_inputs()[1].name: inputs['attention_mask'].cpu().numpy(),}

    ort_outs = ort_session.run(None, ort_inputs)
    print(ort_outs)


def mdl2onnx(model_path, inputs):
    model = AutoModelForTokenClassification.from_pretrained(model_name)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    
    torch.onnx.export(model, inputs, 
                  model_name + "/model.onnx", 
                  opset_version=12,
                  input_names=['input_ids', 'attention_mask'], 
                  output_names=['logits'])

def main(args):
    # argv1 = sys.argv[1]
    # argv2 = sys.argv[2]
    model_path = 'mdl/zhpr_test_mdl_230906'
    batch_data = prepare_data('今天天气怎么样')
    # mdl2onnx(model_path, batch_data)
    # print("model exported")
    test_onnx(model_path, batch_data)


    


if __name__ == '__main__':
    main(sys.argv[1:])
