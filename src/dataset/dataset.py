import json
import os
import threading
from multiprocessing import Queue
import torch
from torch.utils.data import Dataset, IterableDataset, DataLoader
import pytorch_lightning as pl

from src.model.core import get_tokenizer
from src.utils.constant import ID2LABEL, LABEL2ID


class DocumentDataset(Dataset):
    def __init__(self, document:str, window_size=384,step=307) -> None:
        super().__init__()
        self.tokenizer = get_tokenizer()
        self.label2id, self.id2label = LABEL2ID, ID2LABEL
        self.document = document
        self.window_size = window_size
        self.step = step
        self.data = list(self._stride(self.window_size))
        
    def _stride(self, window_size):
        
        tokens = list(self.document)
        for window_start in range(0, len(tokens), self.step):
            window_tokens = tokens[window_start:window_start+window_size]
            yield {
                'tokens': window_tokens,
            }

    def __getitem__(self, index):
        data = self.data[index]
        tokens = self.tokenizer.convert_tokens_to_ids(data['tokens'])
        masks = [1] * len(tokens)
        while len(tokens) < self.window_size:
            tokens.append(self.tokenizer.pad_token_id)
            masks.append(0)

        return torch.tensor(tokens), torch.tensor(masks)

    def __len__(self):
        return len(self.data)
    
def merge_stride(output:int,step:int):
    out = []
    for sent_idx, stride_sent in enumerate(output):
        token_idx = step * sent_idx
        for token_ner in stride_sent:
            if token_idx + 1 > len(out):
                out.append(token_ner)
            else:
                out[token_idx] = token_ner
            token_idx += 1
    return out
    
def decode_pred(token_ners):
    out = []
    for token_ner in token_ners:
        out.append(token_ner[0])
        if token_ner[-1] != 'O':
            out.append(token_ner[-1][-1])
    return out


class DataReader(threading.Thread):
    def __init__(self, file_path, queue):
        threading.Thread.__init__(self)
        self.f_handle = open(file_path, 'r', encoding='utf-8', errors='ignore')
        self.queue = queue
        self.daemon = True
        self.start()

    def run(self):
        for line in self.f_handle:
            line_json = json.loads(line.strip())
            if line_json == None:
                continue
            self.queue.put(line_json)
        self.queue.put(None)


class PunctStreamDataset(IterableDataset):
    def __init__(self, file_path, window_size=384, maxsize=10):
        """
        file_path: 文本文件
        """
        self.window_size = window_size
        self.tokenizer = get_tokenizer()
        self.label2id, self.id2label = LABEL2ID, ID2LABEL
        self.queue = Queue(maxsize=maxsize)
        self.data_reader = DataReader(file_path, self.queue)
    
    def process_line(self, line):
        data_lst = self._stride(line, self.window_size)
        for data in data_lst:
            bios = [self.label2id[x] for x in data['bios']]
            tokens = self.tokenizer.convert_tokens_to_ids(data['tokens'])
            masks = [1] * len(tokens)
            while len(tokens) < self.window_size:
                tokens.append(self.tokenizer.pad_token_id)
                masks.append(0)
                bios.append(-100)
            yield torch.tensor(tokens),torch.tensor(masks), torch.tensor(bios)

    def _stride(self, line, window_size):
        step = int(window_size * 0.8)
        tokens = line['tokens']
        bios = line['bios']
        data_lst = list()
        for window_start in range(0, len(tokens), step):
            window_tokens = tokens[window_start:window_start + window_size]
            window_bios = bios[window_start:window_start + window_size]
            out = {'tokens': window_tokens, 'bios': window_bios}
            data_lst.append(out)
        return data_lst

    def __iter__(self):
        while True:
            json = self.queue.get()
            if json is None:
                break
            for sample in self.process_line(json):
                yield sample

    # def __lenght__(self):
    #     return self.length

class PunctDataModule(pl.LightningDataModule):
    def __init__(self, batch_size=8, num_workers=3, prefetch_factor=100, pin_memory=True):
        super().__init__()
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.prefetch_factor = prefetch_factor
        self.pin_memory = pin_memory
        self.train = PunctStreamDataset(
            '/mnt/AM4_disk9/chenwuchen/code/github/punc/data/train_data/lst/all_data_230822_train.json')
        self.valid = PunctStreamDataset(
            '/mnt/AM4_disk9/chenwuchen/code/github/punc/data/train_data/lst/all_data_230822_dev.json')
        self.test = PunctStreamDataset(
            '/mnt/AM4_disk9/chenwuchen/code/github/punc/data/train_data/lst/all_data_230822_test.json')

    def train_dataloader(self):
        return DataLoader(self.train, batch_size=self.batch_size, num_workers=self.num_workers, prefetch_factor=self.prefetch_factor)

    def val_dataloader(self):
        return DataLoader(self.valid, batch_size=self.batch_size, num_workers=self.num_workers, prefetch_factor=self.prefetch_factor)

    def test_dataloader(self):
        return DataLoader(self.test, batch_size=self.batch_size)
