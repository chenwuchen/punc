#!/usr/bin/python
# -*- encoding: utf-8 -*-
########################################################################
#                                                                       
# Author: chenwuchen@baidu.com                                          
# Date: 2023/08/19 15:16:16 
#                                                                       
########################################################################

import sys
import os
import json
import unicodedata
from typing import List
from zhconv import convert 


USE_ZH_PUNCTUATION = [
    '，',
    '、',
    '。',
    '？',
    '！',
    '；',
]

def isFull(c:str)->bool:
    # 全形字元的 unicode 編碼是從 65281 到 65374
    return ord(c) in range(65281,65374+1)

def isHalf(c:str)->bool:
    # 半形字元的 unicode 編碼是從 33 到 126
    return ord(c) in range(33,126+1)

def full2half(c: str) -> str:
    if isFull(c):
        return chr(ord(c)-65248)
    return c

def half2full(c: str) -> str:
    if isHalf(c):
        return chr(ord(c)+65248)
    return c

def ch_text_norm(text):
    # 將所有符號轉換至全形
    # 英文或是數字轉換成半形
    # 符號空白等移除
    # https://en.wikipedia.org/wiki/Template:General_Category_(Unicode)
    text = list(text)
    for t_id,c in enumerate(text):
        if unicodedata.category(c)[0] == 'P':
            new_char = half2full(c)
            text[t_id] = new_char
        if unicodedata.category(c)[0] == 'N' or unicodedata.category(c) in ['Lu','Ll']:
            new_char = full2half(c)
            text[t_id] = new_char
        if unicodedata.category(c)[0] in ['Z','O']:
            text[t_id] = "" # soft del
        if unicodedata.category(c) in ['So']:
            text[t_id] = "" # soft del
    
    return ''.join(text)

def tran2simple(sentence):
    """繁体字转简体字"""
    out = convert(sentence, 'zh-cn')
    return out 

def clean_punct(text, keep_use_punct=False):
    text = list(text)
    for t_id, c in enumerate(text):
        if keep_use_punct == True:
            if unicodedata.category(c)[0] == 'P' and c not in USE_ZH_PUNCTUATION:
                text[t_id] = ""
        elif keep_use_punct == False:
            if unicodedata.category(c)[0] == 'P':
                text[t_id] = ""
    return ''.join(text)


def gen_input_and_bio(text: str) -> List[str]:
    tokens = list(text)
    bio = []
    tokens_wo_punct = []
    for t_id, token in enumerate(tokens):
        if token in USE_ZH_PUNCTUATION:
            if len(bio) > 0 and bio[-1] == 'O':
                bio.pop(-1)
            bio.append(f"S-{token}")
        else:
            bio.append('O')
            tokens_wo_punct.append(token)
    return tokens_wo_punct,bio

def txt_proc(txt):
    txt = tran2simple(txt)
    txt = ch_text_norm(txt)
    txt = clean_punct(txt, keep_use_punct=True)
    return txt

def get_data(line, data_type='raw'):
    if data_type == 'raw':
        data = line.strip()
    elif data_type == 'json':
        data_json = json.loads(line.strip())
        data = data_json['text']
    return data


def data_process(infile, outfile):
    with open(infile, "r") as fr, open(outfile, "w") as fw:
        for line in fr:
            # data = json.loads(line)
            # data = clean_punct(ch_text_norm(data['text']), keep_use_punct=True)
            data = get_data(line, data_type='json')       
            data = txt_proc(data)
            data_lst = data.split("\n")
            data_lst = list(filter(lambda sent:len(sent)>5 and sent[-1]=='。', data_lst)) # 移除掉段落标题，或者结尾没有标点符号的句子
            tokens, bios = gen_input_and_bio(" ".join(data_lst))
            if len(tokens) != len(bios) or len(tokens) == 0:
                # 转换至BIO表示法可能会应为连续的符号导致转换出來的长度不通
                continue
            out = f"{json.dumps({'tokens':tokens,'bios':bios}, ensure_ascii=False)}\n"
            fw.write(out)
    pass

def data_process01(infile, outfile):
    with open(infile, "r",  errors='igore') as fr, open(outfile, "w") as fw:
        lines = json.loads(fr.readline().strip('\n'))
        print(lines[:3])
            # data = json.loads(line)
            # data = clean_punct(ch_text_norm(data['text']), keep_use_punct=True)
        # data = get_data(line)       
        # data = clean_punct(ch_text_norm(data), keep_use_punct=True)
        # data_lst = data.split("\n")
        # data_lst = list(filter(lambda sent:len(sent)>5 and sent[-1]=='。', data_lst)) # 移除掉段落标题，或者结尾没有标点符号的句子
        # tokens, bios = gen_input_and_bio(" ".join(data_lst))
        # if len(tokens) != len(bios) or len(tokens) == 0:
        #     # 转换至BIO表示法可能会应为连续的符号导致转换出來的长度不通
        #     continue
        # out = f"{json.dumps({'tokens':tokens,'bios':bios}, ensure_ascii=False)}\n"
        # fw.write(out)
    pass


def main(args):
    infile, outfile = sys.argv[1], sys.argv[2]
    data_process(infile, outfile)


if __name__ == '__main__':
    main(sys.argv[1:])
