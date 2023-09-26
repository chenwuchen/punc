import sys
import time
import os

from src.dataset.dataset import DocumentDataset, merge_stride, decode_pred
from src.model.common import id_label_convert

from transformers import AutoModelForTokenClassification, AutoTokenizer
from torch.utils.data import DataLoader
import torch
import onnx
import onnxruntime


def load_mdl(mdl_type, model_path):
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    if mdl_type == 'pt':
        model = AutoModelForTokenClassification.from_pretrained(model_path)
    else:
        providers=['CPUExecutionProvider'] # ['CUDAExecutionProvider', 'CPUExecutionProvider'] 
        model = onnxruntime.InferenceSession(model_path + "/model.onnx", providers=providers)
    return model, tokenizer

def infer(model, batch):
    if isinstance(model, torch.nn.Module):
        encodings = {'input_ids': batch[0], 'attention_mask':batch[1],}
        mdl_out = model(**encodings)
        output = mdl_out['logits']
    elif isinstance(model, onnxruntime.InferenceSession):
        ort_inputs = {model.get_inputs()[0].name:  batch[0].cpu().numpy(), 
                    model.get_inputs()[1].name: batch[1].cpu().numpy(),}
        mdl_out = model.run(None, ort_inputs)
        output = mdl_out[0]
    else:
        raise ValueError("Unknown model type")
    output = output.argmax(-1)
    return output
    

def predict_step(batch, model, tokenizer):
        batch_out = []
        t1 = time.time()
        output = infer(model, batch)
        print("cost time: ", (time.time() - t1))
        label2id, id2label = id_label_convert()

        # predicted_token_class_id_batch = output['logits'].argmax(-1)
        for predicted_token_class_ids, input_ids in zip(output, batch[0]):
            out=[]
            tokens = tokenizer.convert_ids_to_tokens(input_ids)
            
            # compute the pad start in input_ids and also truncate the predict
            input_ids = input_ids.tolist()
            try:
                input_id_pad_start = input_ids.index(tokenizer.pad_token_id)
            except:
                input_id_pad_start = len(input_ids)
            input_ids = input_ids[:input_id_pad_start]
            tokens = tokens[:input_id_pad_start]
    
            # predicted_token_class_ids
            predicted_tokens_classes = [id2label[t.item()] for t in predicted_token_class_ids]
            predicted_tokens_classes = predicted_tokens_classes[:input_id_pad_start]

            for token, ner in zip(tokens, predicted_tokens_classes):
                out.append((token, ner))
            batch_out.append(out)
        return batch_out

# def down_mdl():
#     cache_dir= './mdl/cache/'
#     model_name = 'p208p2002/zh-wiki-punctuation-restore'
#     model = AutoModelForTokenClassification.from_pretrained(model_name, cache_dir=cache_dir)
#     model.save_pretrained('./mdl/zh-wiki-punctuation-restore')
#     tokenizer = AutoTokenizer.from_pretrained(model_name, cache_dir=cache_dir)
#     tokenizer.save_pretrained('./mdl/zh-wiki-punctuation-restore')

def add_punc(text):
    window_size = 256
    step = 200
    dataset = DocumentDataset(text, window_size=window_size, step=step)
    dataloader = DataLoader(dataset=dataset, shuffle=False, batch_size=5)

    model_path = 'mdl/zhpr_test_mdl_230906'
    model, tokenizer = load_mdl(mdl_type='onnx', model_path=model_path)

    model_pred_out = []
    for batch in dataloader:
        batch_out = predict_step(batch, model, tokenizer)
        for out in batch_out:
            model_pred_out.append(out)
    
    merge_pred_result = merge_stride(model_pred_out, step)
    merge_pred_result_deocde = decode_pred(merge_pred_result)
    punc_txt = ''.join(merge_pred_result_deocde)
    return punc_txt


if __name__ == "__main__":
    # text = "意大利语Italiano，中文也简称为意语，隶属于印欧语系的罗曼语族。现有7千万人日常用意大利语，大多是意大利居民。另有28个国家使用意大利语，其中4个立它为官方语言。标准意大利语源自于托斯卡纳语中的佛罗伦斯方言，发音处于意大利中北部方言之间。标准音近来稍微加进了经济较为发达的米兰地区口音。在作曲领域中，亦使用为数不少的意大利文字词。意大利语和拉丁语一样，有长辅音。其他的罗曼语族语言如西班牙语、法语已无长辅音。"
    text = "意大利语Italiano中文也简称为意语隶属于印欧语系的罗曼语族现有7千万人日常用意大利语大多是意大利居民另有28个国家使用意大利语其中4个立它为官方语言标准意大利语源自于托斯卡纳语中的佛罗伦斯方言发音处于意大利中北部方言之间标准音近来稍微加进了经济较为发达的米兰地区口音在作曲领域中亦使用为数不少的意大利文字词意大利语和拉丁语一样有长辅音其他的罗曼语族语言如西班牙语法语已无长辅音"
    txt = add_punc(text)
    txt = txt.replace('[UNK]',' ')
    print(txt)
    # down_mdl()