import random
import warnings
import time
import logging
import yaml
import os
import importlib
import torch
import numpy as np
from typing import Callable, Any
from easydict import EasyDict
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, average_precision_score, matthews_corrcoef
from Bio.PDB import PDBParser, PDBIO

from .util_classes import MyPrint

p = PDBParser(QUIET=True)


def set_seed(seed=42):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    np.random.seed(seed)
    random.seed(seed)


def prepare(config):
    # set TRAIN_PATH
    TRAIN_PATH = f'./runs/{int(time.time())}'
    if not os.path.exists(TRAIN_PATH):
        os.makedirs(TRAIN_PATH)
    warnings.filterwarnings("ignore")
    # set logger
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    file_handler = logging.FileHandler(f'{TRAIN_PATH}/run.log')
    file_handler.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    # set device
    device = torch.device(f'cuda:{config.train.gpu_id}' if torch.cuda.is_available() else 'cpu')
    # set print
    my_print = MyPrint(logger).pprint
    my_print('TRAIN_PATH:', TRAIN_PATH)
    my_print('Now the time is:', time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()))
    my_print(config)
    config.model.esm_path = os.path.join(config.model.model_dir, config.model.esm_version)
    config.train.save_path = TRAIN_PATH
    return logger, device, my_print


def read_config(config_path):
    with open(config_path, 'r', encoding='utf-8') as r:
        config = EasyDict(yaml.safe_load(r))
    set_seed(config.settings.seed)
    return config


def get_model(pprint, config):
    module = importlib.import_module('model.model')
    model_class = getattr(module, config.model_name)
    model = model_class(config=config)
    pprint('total params:', sum(p.numel() for p in model.parameters()))
    pprint('trainable params', sum(p.numel() for p in model.parameters() if p.requires_grad))
    return model


def cal_auroc(y_true, y_proba, average='macro', multi_class='ovo'):
    unique_classes = np.unique(y_true)
    y_proba_trimmed = y_proba[:, unique_classes]
    y_proba_normalized = y_proba_trimmed / y_proba_trimmed.sum(axis=1, keepdims=True)
    if len(unique_classes) == 2:
        auroc = roc_auc_score(y_true, y_proba_normalized[:, 1], average=average)
    else:
        auroc = roc_auc_score(y_true, y_proba_normalized, multi_class=multi_class, average=average)
    return auroc

def cal_auprc(y_true, y_proba, average='macro'):
    unique_classes = np.unique(y_true)
    y_proba_trimmed = y_proba[:, unique_classes]
    y_true_onehot = np.zeros_like(y_proba_trimmed)
    for i, cls in enumerate(unique_classes):
        y_true_onehot[:, i] = (y_true == cls).astype(int)
    auprc = average_precision_score(y_true_onehot, y_proba_trimmed, average=average)
    return auprc


def measure_graph(labels, preds, ptr, batch, verbose=False):
    
    accuracy, precision, recall, f1, auroc, auprc, mcc = 0, 0, 0, 0, 0, 0, 0
    for graph_idx in range(len(ptr) - 1):
        mask = (batch == graph_idx)
        label = labels[mask].detach().cpu().numpy()
        pred = preds[mask].argmax(dim=-1).detach().cpu().numpy()
        prob = preds[mask].detach().cpu().numpy()
        accuracy += accuracy_score(label, pred)
        precision += precision_score(label, pred, average='macro')
        recall += recall_score(label, pred, average='macro')
        f1 += f1_score(label, pred, average='macro')
        auroc += cal_auroc(label, prob, average='macro', multi_class='ovo')
        auprc += cal_auprc(label, prob, average='macro')
        mcc += matthews_corrcoef(label, pred)

    return {
        'accuracy': accuracy / (len(ptr) - 1),
        'precision': precision / (len(ptr) - 1),
        'recall': recall / (len(ptr) - 1),
        'f1': f1 / (len(ptr) - 1),
        'auroc': auroc / (len(ptr) - 1),
        'auprc': auprc / (len(ptr) - 1),
        'mcc': mcc / (len(ptr) - 1),
    }


def log_metrics(pprint: Callable[..., Any], thistype: str, epoch: int = None, metric_dict: dict = None, annot: str = None) -> None:
    if thistype == 'train':
        train_or_val = 'Train'
    elif thistype == 'val':
        train_or_val = 'Val'
    else:
        train_or_val = 'Test'
    info_str = f'{train_or_val} Epoch: {epoch} - {annot} : ' if epoch is not None else f'{train_or_val}: '
    for metric in metric_dict:
        info_str += f'{metric}: {metric_dict[metric]:.8f}, '
    pprint(info_str)


def save_pdb(pdb_dir, entity, score, flag):
    structure = p.get_structure('protein', f'{pdb_dir}/{entity}.pdb')
    for i, r in enumerate(structure[0]['A']):
        for a in r:
            a.set_bfactor(score[i])
    io = PDBIO()
    io.set_structure(structure)
    io.save(f'{pdb_dir}/{entity}-{flag}.pdb')
    print(f"Saved confidence score of {entity} to {pdb_dir}/{entity}-{flag}.pdb")
