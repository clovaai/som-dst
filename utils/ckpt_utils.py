import wget
import os
import torch

# from pytorch_transformers import BertForPreTraining, BertConfig
from transformers import BertForPreTraining, BertConfig


BERT_PRETRAINED_MODEL_ARCHIVE_MAP = {
    "bert-base-uncased": "https://s3.amazonaws.com/models.huggingface.co/bert/bert-base-uncased-pytorch_model.bin",
    "bert-large-uncased": "https://s3.amazonaws.com/models.huggingface.co/bert/bert-large-uncased-pytorch_model.bin",
}


def download_ckpt(ckpt_path, config_path, target_path="assets"):
    key = None
    if "base" in ckpt_path.lower():
        key = "bert-base-uncased"
    if "large" in ckpt_path.lower():
        key = "bert-large-uncased"
    assert key in BERT_PRETRAINED_MODEL_ARCHIVE_MAP

    url_path = BERT_PRETRAINED_MODEL_ARCHIVE_MAP[key]
    print("start download %s from huggingface" % key)
    wget.download(url_path, out=target_path)
    ckpt_path = os.path.join(target_path, key + "-pytorch_model.bin")
    ckpt = convert_ckpt_compatible(ckpt_path, config_path)
    torch.save(ckpt, ckpt_path)

    return ckpt_path


def convert_ckpt_compatible(ckpt_path, config_path):
    ckpt = torch.load(ckpt_path, map_location="cpu")
    keys = list(ckpt.keys())
    for key in keys:
        if "LayerNorm" in key:
            if "gamma" in key:
                ckpt[key.replace("gamma", "weight")] = ckpt.pop(key)
            else:
                ckpt[key.replace("beta", "bias")] = ckpt.pop(key)

    model_config = BertConfig.from_json_file(config_path)
    model = BertForPreTraining(model_config)
    model.load_state_dict(ckpt)
    new_ckpt = model.bert.state_dict()

    return new_ckpt
