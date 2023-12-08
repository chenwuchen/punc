import torch.nn as nn
from transformers import BertTokenizerFast, BertForTokenClassification, BertConfig
from transformers import MobileBertTokenizerFast, MobileBertForTokenClassification, MobileBertConfig

from src.utils.constant import USE_ZH_PUNCTUATION, TOKENIER_CONF, ID2LABEL, LABEL2ID

class BaseModel(nn.Module):
    """
    BaseFeaturesExtractor class that will extract features according to the type of model
    https://huggingface.co/blog/fine-tune-wav2vec2-english
    """

    def __init__(self, params):
        super().__init__()
        self.params = params

    def forward(self, x):
        outputs = self.model(x)
        return outputs



class PuncMobileBert():
    """
    https://huggingface.co/docs/transformers/v4.16.2/en/model_doc/
    """
    def __init__(self, mdl_path, params):
        super().__init__()
        # if mdl_conf is None : mdl_conf = TOKENIER_CONF
        self.tokenizer = MobileBertTokenizerFast.from_pretrained(mdl_path)
        conf = MobileBertConfig.from_pretrained(mdl_path, label2id=LABEL2ID, id2label=ID2LABEL, num_labels=len(USE_ZH_PUNCTUATION) + 1)
        self.model = MobileBertForTokenClassification.from_pretrained(mdl_path, config=conf, ignore_mismatched_sizes=True)


class PuncBert():
    """
    https://huggingface.co/docs/transformers/v4.16.2/en/model_doc/
    """
    def __init__(self, mdl_path, params):
        super().__init__()
        # if mdl_conf is None : mdl_conf = TOKENIER_CONF
        self.tokenizer = tokenizer = BertTokenizerFast.from_pretrained(mdl_path)
        conf = BertConfig.from_pretrained(mdl_path, label2id=LABEL2ID, id2label=ID2LABEL, num_labels=len(USE_ZH_PUNCTUATION) + 1)
        self.model = BertForTokenClassification.from_pretrained(mdl_path, config=conf, ignore_mismatched_sizes=True)
