import json
import os
import threading
from multiprocessing import Queue
import time
import torch
from torch import optim
import pytorch_lightning as pl
from pytorch_lightning.callbacks import Callback

from src.model.models import PuncMobileBert, PuncBert

class ClassificationMetric(object): # 记录结果并计算指标
    def __init__(self, accuracy=True):
        self.accuracy = accuracy
        self.preds = []
        self.target = []

    def reset(self): # 重置结果
        self.preds.clear()
        self.target.clear()
        gc.collect()

    def update(self, preds, target): # 更新结果
        preds = list(preds.detach().argmax(1))
        target = list(target.detach().argmax(1))
        self.preds += preds
        self.target += target

    def compute(self): # 计算结果
        metrics = []
        if self.accuracy:
            metrics.append(accuracy_score(self.target, self.preds))
        self.reset()
        return metrics

class BertPunc(pl.LightningModule):
    def __init__(self, params):
        super().__init__()
        self.params = params
        self.save_hyperparameters(self.params)

        if params.model == 'bert':
            self.punc = PuncBert(params.base_mdl, params)
        elif params.model == 'mobilebert':
            self.punc = PuncMobileBert(params.base_mdl, params)
        else:
            raise NotImplementedError

        self.model = self.punc.model
        self.tokenizer = self.punc.tokenizer


    def training_step(self, batch, batch_idx):
        labels = batch[-1]
        mask = batch[1]
        encodings = {
            'input_ids': batch[0],
            'attention_mask':batch[1],
            'labels': labels,
        }
        output = self.model(**encodings)
        loss = output['loss']
        logits = output['logits']
        acc, precision = self._compute_acc(logits, labels, mask)
        # self.log("train_loss", loss, sync_dist=True)

        lr = self.trainer.optimizers[0].param_groups[0]['lr']
        values = {"epoch": self.current_epoch, "step": self.global_step, "lr":lr, "train_loss": round(loss.item(), 3), "acc": acc, "precision": precision}  # add more items if needed
        self.log_dict(values, prog_bar=True, sync_dist=True)
        return loss

    def validation_step(self, batch, batch_idx):
        encodings = {
            'input_ids': batch[0],
            'attention_mask':batch[1],
            'labels': batch[-1]
        }
        output = self.model(**encodings)
        loss = output['loss']

        self.log("val_loss", loss, prog_bar=True, sync_dist=True)
        return loss

    def test_step(self, batch, batch_idx):
        encodings = {
            'input_ids': batch[0],
            'attention_mask':batch[1],
            'labels': batch[-1]
        }
        output = self.model(**encodings)
        loss = output['loss']

        tf = open(os.path.join(self.trainer.log_dir, '_test.label'), 'a')
        with open(os.path.join(self.trainer.log_dir, 'pred.log'), 'a') as f:
            predicted_token_class_id_batch = output['logits'].argmax(-1)
            for predicted_token_class_ids, labels in zip(predicted_token_class_id_batch, labels := batch[-1]):

                # compute the pad start in lable
                # and also truncate the predict
                labels = labels.tolist()
                try:
                    labels_pad_start = labels.index(-100)
                except:
                    labels_pad_start = len(labels)
                labels = labels[:labels_pad_start]
                
                # predicted_token_class_ids
                predicted_tokens_classes = [self.model.config.id2label[t.item()] for t in predicted_token_class_ids]
                predicted_tokens_classes = predicted_tokens_classes[:labels_pad_start]
                predicted_tokens_classe_pred = ' '.join(
                    predicted_tokens_classes)
                f.write(f"{predicted_tokens_classe_pred}\n")

                # labels
                labels_tokens_classes = [self.model.config.id2label[t] for t in labels]
                labels_tokens_classes = ' '.join(labels_tokens_classes)
                tf.write(f"{labels_tokens_classes}\n")

        self.log("test_loss", loss, prog_bar=True, sync_dist=True)
        return loss


    def configure_optimizers(self):
        # ReduceLROnPlateau
        optimizer = optim.Adam(self.parameters(), lr=self.params.learning_rate)
        if self.params.scheduler != None:
            if self.params.scheduler == 'OneCycleLR':
                scheduler = optim.lr_scheduler.OneCycleLR(optimizer, max_lr=self.params.learning_rate, steps_per_epoch=self.params.steps_per_epoch, epochs=self.params.max_epochs)
            else:
                scheduler = {
                        "scheduler": ReduceLROnPlateau(optimizer, mode="min", patience=self.optim_param.patience, min_lr=self.optim_param.min_lr,),
                        "monitor": "val/loss",
                    }
            return [[optimizer], [scheduler]]
        return optimizer

    def _compute_acc(self, logits, labels, mask):
        preds = torch.argmax(logits, dim=2)
        total = mask.sum().item()
        correct = ((preds == labels) * mask).sum().item()
        acc = correct/total
        punc_mask = (labels > 0)
        correct_abs = ((preds == labels) * punc_mask).sum().item()
        total_abs = punc_mask.sum().item()
        precision = correct_abs/total_abs
        # print(f'mask: {mask[0]}\n label: {labels[0]}\n preds: {preds[0]}\tpunc_mask: {punc_mask[0]} ')
        # print(acc, acc_abs)
        # print(correct, total,  b , correct/total)
        return round(acc, 3), round(precision, 3)


class LogLearningRateCallback(Callback):
    def on_batch_start(self, trainer, pl_module):
        optimizer = pl_module.optimizers()  # 获取模型的优化器
        lr = optimizer.param_groups[0]['lr']  # 获取学习率
        trainer.logger.log_metrics({"lr": lr}, prog_bar=True, sync_dist=True)
