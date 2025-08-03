import torch
import os
from torch import optim
from tqdm import tqdm
import torch.nn as nn
import numpy as np
import argparse
from sklearn.metrics import confusion_matrix
import sys
proj_path = os.path.abspath('.')
sys.path.append(proj_path)

from utils.util_functions import read_config, log_metrics, prepare, get_model, measure_graph, save_pdb
from utils.util_classes import WarmupCosineAnnealingLR
from torch_geometric.data import DataLoader
from datasets.dataset import ProteinDataset

device, pprint = None, None


def fuse(model, train_loader, val_loader):
    model.to(device)
    pprint('trainable params', sum(p.numel() for p in model.parameters() if p.requires_grad))

    epochs = config.train.epochs

    class_weights = torch.tensor([0.01, 1.0, 0.7, 1.0, 0.5, 2.5, 2]).to(device)
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=config.train.lr)
    scheduler = WarmupCosineAnnealingLR(optimizer, total_steps=epochs*len(train_loader))
    
    best_f_score_token, best_metric_token, info_str_token = 0.0, {}, 'Best val scores (Token): '

    for epoch in range(epochs):
        model.train()
        tr_metric_dict_token = {}
        for metric in config.train.metrics:
            tr_metric_dict_token[metric] = 0.0

        pp, tt = np.array([], dtype=int), np.array([], dtype=int)
        for batch in tqdm(train_loader, desc=f"Training: Epoch {epoch+1}/{epochs}"):

            optimizer.zero_grad()
            output = model(batch.to(device))
            cs_loss = criterion(output['token_logits'], batch.y)
            center_loss = output['center_loss']
            inter_loss = output['inter_loss']
            token_loss = cs_loss + center_loss * config.train.lambda1 + inter_loss * config.train.lambda2
            token_loss.backward()
            optimizer.step()
            scheduler.step()
            tr_metric_dict_token['loss'] += token_loss.item()
            token_measurements = measure_graph(batch.y, output['token_logits'], batch.ptr, batch.batch)
            for k, v in token_measurements.items():
                tr_metric_dict_token[k] += v

            p = output['token_logits'].argmax(dim=-1).cpu().numpy()
            t = batch.y.cpu().numpy()
            pp = np.concatenate((pp, p))
            tt = np.concatenate((tt, t))

        for k in tr_metric_dict_token.keys():
            tr_metric_dict_token[k] /= len(train_loader)

        log_metrics(pprint, 'train', epoch+1, tr_metric_dict_token, 'Token')
        val_metric_dict_token = evaluate(epochs, epoch, model, val_loader, criterion, align=False)

        if epoch == 0 or val_metric_dict_token['f1'] > best_f_score_token:
            best_f_score_token = val_metric_dict_token['f1']
            best_metric_token = val_metric_dict_token

            exist_files = [f for f in os.listdir(config.train.save_path) if f'best_model_' in f]
            for f in exist_files:
                os.remove(os.path.join(config.train.save_path, f))
            checkpoint_path_total = f"{config.train.save_path}/best_model_{best_f_score_token}.pth"
            pprint(f"Saving the best model to {checkpoint_path_total}")
            torch.save(model, checkpoint_path_total)

    for k, v in best_metric_token.items():
        info_str_token += f"{k}: {v:.4f}, "
    pprint(info_str_token)
    
    return checkpoint_path_total


def evaluate(epochs, epoch, model, val_loader, criterion, align=False):
    model.eval()
    val_metric_dict_token = {}
    for metric in config.train.metrics:
        val_metric_dict_token[metric] = 0.0
    acc_dataloader_size_token, acc_dataloader_size_cl = len(val_loader), len(val_loader)
    
    pp, tt = np.array([], dtype=int), np.array([], dtype=int)
    with torch.no_grad():
        for batch in tqdm(val_loader, desc=f"Validation: Epoch {epoch+1}/{epochs}"):

            output = model(batch.to(device))
            cs_loss = criterion(output['token_logits'], batch.y)
            center_loss = output['center_loss']
            inter_loss = output['inter_loss']
            token_loss = cs_loss + center_loss * config.train.lambda1 + inter_loss * config.train.lambda2
            val_metric_dict_token['loss'] += token_loss.item()
            token_measurements = measure_graph(batch.y, output['token_logits'], batch.ptr, batch.batch)
            for k, v in token_measurements.items():
                val_metric_dict_token[k] += v

            p = output['token_logits'].argmax(dim=-1).cpu().numpy()
            t = batch.y.cpu().numpy()
            pp = np.concatenate((pp, p))
            tt = np.concatenate((tt, t))

    for k in val_metric_dict_token.keys():
        val_metric_dict_token[k] /= acc_dataloader_size_token

    log_metrics(pprint, 'val', epoch+1, val_metric_dict_token, 'Token')

    return val_metric_dict_token


def test(checkpoint_path, test_loader, config):
    model = torch.load(checkpoint_path, map_location=device)
    model.eval()
    test_metric_dict_token = {}
    for metric in config.train.metrics:
        test_metric_dict_token[metric] = 0.0
    acc_dataloader_size_token = len(test_loader)
    
    pp, tt = np.array([], dtype=int), np.array([], dtype=int)
    with torch.no_grad():
        for batch in tqdm(test_loader, desc="Testing: "):

            output = model(batch.to(device))
            p = output['token_logits'].argmax(dim=-1).cpu().numpy()
            t = batch.y.cpu().numpy()
            pp = np.concatenate((pp, p))
            tt = np.concatenate((tt, t))

            if config.train.save_conf:
                for graph_idx in range(len(batch.ptr) - 1):
                    mask = (batch.batch == graph_idx)
                    stru_seq_seq = output['attn_weights_1'][graph_idx][:sum(mask)]
                    seq_stru_stru = output['attn_weights_2'][graph_idx][:sum(mask)]
                    stru_seq_seq = (stru_seq_seq - np.min(stru_seq_seq)) / (np.max(stru_seq_seq) - np.min(stru_seq_seq))
                    seq_stru_stru = (seq_stru_stru - np.min(seq_stru_stru)) / (np.max(seq_stru_stru) - np.min(seq_stru_stru))

                    save_pdb(config.dataset.pdb_dir, batch.entity[graph_idx], stru_seq_seq, "o_struct")
                    save_pdb(config.dataset.pdb_dir, batch.entity[graph_idx], seq_stru_stru, "o_seq")

        print('Confusion Matrix:')
        cm = confusion_matrix(tt, pp)
        ll = np.unique(np.concatenate([tt, pp]))
        print("    ", end="")
        for ilabel in ll:
            print(f" {ilabel:2d}", end="")
        print()

        for i, ilabel in enumerate(ll):
            print(f"{ilabel:2d} ", end="")
            for j in range(len(ll)):
                print(f" {cm[i, j]:2d}", end="")
            print()

    for k in test_metric_dict_token.keys():
        test_metric_dict_token[k] /= acc_dataloader_size_token
        
    log_metrics(pprint, 'test', metric_dict=test_metric_dict_token, annot='Token')


def prepare_dataloaders(config):
    train_dataset = ProteinDataset(config, 'train', process=config.dataset.process)
    valid_dataset = ProteinDataset(config, 'valid', process=config.dataset.process)
    test_dataset = ProteinDataset(config, 'test', process=config.dataset.process)

    train_dataloader = DataLoader(train_dataset, batch_size=config.dataset.batch_size, drop_last=True, shuffle=True)
    valid_dataloader = DataLoader(valid_dataset, batch_size=config.dataset.batch_size, drop_last=True, shuffle=False)
    test_dataloader = DataLoader(test_dataset, batch_size=config.dataset.batch_size, drop_last=True, shuffle=False)
    print('# of train, valid, test samples:', len(train_dataset), len(valid_dataset), len(test_dataset))
    return train_dataloader, valid_dataloader, test_dataloader


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train the model')
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', help="running configurations", type=str, required=False, default=f'{proj_path}/configs/config.yaml')
    args = parser.parse_args()
    config = read_config(args.config)
    _, device, pprint = prepare(config)

    train_loader, val_loader, test_loader = prepare_dataloaders(config)
    model = get_model(pprint, config)

    checkpoint_path_total = fuse(model, train_loader, val_loader)
    pprint("Testing ...")
    test(checkpoint_path_total, test_loader, config)
