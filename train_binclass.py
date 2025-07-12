import argparse
import os
import shutil
import warnings

import albumentations as A
import numpy as np
import pandas as pd
import torch
import torch.multiprocessing
from torchvision.transforms import ToTensor

from isplutils import utils, split

torch.multiprocessing.set_sharing_strategy('file_system')
import torch.nn as nn
from albumentations.pytorch import ToTensorV2
from sklearn.metrics import roc_auc_score
from tensorboardX import SummaryWriter
from torch import optim
from torch.utils.data import DataLoader
from tqdm import tqdm

from architectures import fornet
from isplutils.data import FrameFaceIterableDataset


def main():
    # Args
    parser = argparse.ArgumentParser()
    parser.add_argument('--net', type=str, help='Net model class', required=True)
    parser.add_argument('--traindb', type=str, nargs='+', default=['ff-c23-720-140-140'], required=True)
    parser.add_argument('--valdb', type=str, nargs='+', default=['ff-c23-720-140-140'], required=True)
    parser.add_argument('--ffpp_faces_df_path', type=str, required=True)
    parser.add_argument('--ffpp_faces_dir', type=str, required=True)
    parser.add_argument('--size', type=int, required=True)

    parser.add_argument('--batch', type=int, default=32)
    parser.add_argument('--lr', type=float, default=1e-5)
    parser.add_argument('--valint', type=int, default=500)
    parser.add_argument('--patience', type=int, default=10)
    parser.add_argument('--maxiter', type=int, default=20000)
    parser.add_argument('--init', type=str)
    parser.add_argument('--scratch', action='store_true')

    parser.add_argument('--trainsamples', type=int, default=-1)
    parser.add_argument('--valsamples', type=int, default=-1)

    parser.add_argument('--logint', type=int, default=100)
    parser.add_argument('--workers', type=int, default=6)
    parser.add_argument('--device', type=int, default=0)
    parser.add_argument('--seed', type=int, default=0)

    parser.add_argument('--log_dir', type=str, default='runs/binclass/')
    parser.add_argument('--models_dir', type=str, default='weights/binclass/')

    args = parser.parse_args()

    # Parse args
    net_class = getattr(fornet, args.net)
    device = torch.device(f'cuda:{args.device}' if torch.cuda.is_available() else 'cpu')
    seed = args.seed
    torch.manual_seed(seed)
    np.random.seed(seed)

    face_size = args.size
    train_datasets = args.traindb
    val_datasets = args.valdb
    ffpp_df_path = args.ffpp_faces_df_path
    ffpp_faces_dir = args.ffpp_faces_dir

    log_dir = args.log_dir
    weights_dir = args.models_dir

    net: nn.Module = net_class().to(device)

    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(net.get_trainable_parameters(), lr=args.lr)
    lr_scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1,
                                                        patience=args.patience, cooldown=2 * args.patience,
                                                        min_lr=args.lr * 1e-5)

    tag = utils.make_train_tag(net_class=net_class, traindb=train_datasets,
                               patch_size=face_size, seed=seed)

    weights_path = os.path.join(weights_dir, tag)
    os.makedirs(weights_path, exist_ok=True)

    bestval_path = os.path.join(weights_path, 'bestval.pth')
    last_path = os.path.join(weights_path, 'last.pth')
    periodic_path = os.path.join(weights_path, 'it{:06d}.pth')

    logdir = os.path.join(log_dir, tag)
    shutil.rmtree(logdir, ignore_errors=True)
    tb = SummaryWriter(logdir=logdir)

    dummy = torch.randn((1, 3, face_size, face_size), device=device)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        tb.add_graph(net, [dummy], verbose=False)

    # Load weights
    val_loss = min_val_loss = 10
    iteration = epoch = 0
    net_state = opt_state = None

    if args.init:
        state = torch.load(args.init, map_location='cpu')
        net_state = state['net']
    elif not args.scratch and os.path.exists(last_path):
        state = torch.load(last_path, map_location='cpu')
        net_state = state['net']
        opt_state = state['opt']
        iteration = state['iteration'] + 1
        epoch = state['epoch']
    if not args.scratch and os.path.exists(bestval_path):
        state = torch.load(bestval_path, map_location='cpu')
        min_val_loss = state['val_loss']

    if net_state is not None:
        net.load_state_dict(net_state, strict=False)
    if opt_state is not None:
        for param_group in opt_state['param_groups']:
            param_group['lr'] = args.lr
        optimizer.load_state_dict(opt_state)

    # Dataset
    transformer = utils.get_transformer(patch_size=face_size, net_normalizer=net.get_normalizer(), train=True)
    splits = split.make_splits(ffpp_df=ffpp_df_path, ffpp_dir=ffpp_faces_dir,
                               dbs={'train': train_datasets, 'val': val_datasets})

    train_dfs = [splits['train'][db][0] for db in splits['train']]
    train_roots = [splits['train'][db][1] for db in splits['train']]
    val_dfs = [splits['val'][db][0] for db in splits['val']]
    val_roots = [splits['val'][db][1] for db in splits['val']]

    train_dataset = FrameFaceIterableDataset(roots=train_roots, dfs=train_dfs,
                                             num_samples=args.trainsamples,
                                             transformer=transformer, size=face_size)
    val_dataset = FrameFaceIterableDataset(roots=val_roots, dfs=val_dfs,
                                           num_samples=args.valsamples,
                                           transformer=transformer, size=face_size)

    train_loader = DataLoader(train_dataset, num_workers=args.workers, batch_size=args.batch)
    val_loader = DataLoader(val_dataset, num_workers=args.workers, batch_size=args.batch)

    if len(train_dataset) == 0 or len(val_dataset) == 0:
        print("Empty train or val dataset.")
        return

    stop = False
    while not stop:
        train_loss = train_num = 0
        train_labels_list = []
        train_pred_list = []

        for train_batch in tqdm(train_loader, desc=f"Epoch {epoch}", leave=False):
            net.train()
            batch_data, batch_labels = train_batch
            batch_data = batch_data.to(device)
            batch_labels = batch_labels.to(device)

            optimizer.zero_grad()
            out = net(batch_data)
            loss = criterion(out, batch_labels)

            if torch.isnan(loss):
                raise ValueError("NaN loss")

            loss.backward()
            optimizer.step()

            train_batch_num = len(batch_labels)
            train_num += train_batch_num
            train_loss += loss.item() * train_batch_num
            pred = torch.sigmoid(out).detach().cpu().numpy()
            train_pred_list.append(pred.flatten())
            train_labels_list.append(batch_labels.cpu().numpy().flatten())

            if iteration > 0 and (iteration % args.logint == 0):
                train_loss /= train_num
                tb.add_scalar('train/loss', train_loss, iteration)
                tb.add_scalar('lr', optimizer.param_groups[0]['lr'], iteration)
                tb.add_scalar('epoch', epoch, iteration)
                save_model(net, optimizer, train_loss, val_loss, iteration, args.batch, epoch, last_path)
                train_loss = train_num = 0

            if iteration > 0 and (iteration % args.valint == 0):
                save_model(net, optimizer, train_loss, val_loss, iteration, args.batch, epoch,
                           periodic_path.format(iteration))
                train_labels = np.concatenate(train_labels_list)
                train_pred = np.concatenate(train_pred_list)
                train_roc_auc = roc_auc_score(train_labels, train_pred)
                tb.add_scalar('train/roc_auc', train_roc_auc, iteration)
                tb.add_pr_curve('train/pr', train_labels, train_pred, iteration)
                train_labels_list = []
                train_pred_list = []

                val_loss = validation_routine(net, device, val_loader, criterion, tb, iteration, 'val')
                tb.flush()
                lr_scheduler.step(val_loss)
                if val_loss < min_val_loss:
                    min_val_loss = val_loss
                    save_model(net, optimizer, train_loss, val_loss, iteration, args.batch, epoch, bestval_path)
                if optimizer.param_groups[0]['lr'] <= args.lr * 1e-5:
                    stop = True
                    break

            iteration += 1
            if iteration > args.maxiter:
                stop = True
                break

        epoch += 1

    tb.close()
    print("Training complete")


def validation_routine(net, device, val_loader, criterion, tb, iteration, tag):
    net.eval()
    val_loss = 0
    val_num = 0
    pred_list = []
    label_list = []

    for val_batch in tqdm(val_loader, desc="Validation", leave=False):
        batch_data, batch_labels = val_batch
        batch_data = batch_data.to(device)
        batch_labels = batch_labels.to(device)

        with torch.no_grad():
            out = net(batch_data)
            loss = criterion(out, batch_labels)

        pred = torch.sigmoid(out).cpu().numpy().flatten()
        label = batch_labels.cpu().numpy().flatten()

        val_loss += loss.item() * len(label)
        val_num += len(label)
        pred_list.append(pred)
        label_list.append(label)

    val_loss /= val_num
    tb.add_scalar(f"{tag}/loss", val_loss, iteration)

    val_labels = np.concatenate(label_list)
    val_pred = np.concatenate(pred_list)
    val_auc = roc_auc_score(val_labels, val_pred)
    tb.add_scalar(f"{tag}/roc_auc", val_auc, iteration)
    tb.add_pr_curve(f"{tag}/pr", val_labels, val_pred, iteration)

    return val_loss


def save_model(net, optimizer, train_loss, val_loss, iteration, batch_size, epoch, path):
    torch.save({
        'net': net.state_dict(),
        'opt': optimizer.state_dict(),
        'train_loss': train_loss,
        'val_loss': val_loss,
        'iteration': iteration,
        'batch_size': batch_size,
        'epoch': epoch,
    }, path)


if __name__ == "__main__":
    main()
