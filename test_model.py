import argparse
import gc
from collections import OrderedDict
from pathlib import Path

import albumentations as A
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from architectures import fornet
from architectures.fornet import FeatureExtractor
from isplutils import utils, split
from isplutils.data import FrameFaceDatasetTest


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument('--testsets', type=str, help='Testing dataset', nargs='+', choices=['ff-c23-720-140-140'],
                        required=True)
    parser.add_argument('--testsplits', type=str, help='Test split', nargs='+',
                        default=['val', 'test'], choices=['train', 'val', 'test'])
    parser.add_argument('--ffpp_faces_df_path', type=str, action='store',
                        help='Path to the Pandas Dataframe obtained from extract_faces.py on the FF++ dataset. '
                             'Required for training/validating on the FF++ dataset.')
    parser.add_argument('--ffpp_faces_dir', type=str, action='store',
                        help='Path to the directory containing the faces extracted from the FF++ dataset. '
                             'Required for training/validating on the FF++ dataset.')


    parser.add_argument('--model_path', type=Path, required=True,
                        help='Full path to the trained model')
    parser.add_argument('--batch', type=int, default=128,
                        help='Batch size for inference')
    parser.add_argument('--workers', type=int, default=6,
                        help='Number of DataLoader workers')
    parser.add_argument('--device', type=int, default=0,
                        help='GPU id')
    parser.add_argument('--debug', action='store_true')
    parser.add_argument('--num_video', type=int,
                        help='Limit number of real/fake videos')
    parser.add_argument('--results_dir', type=Path, default='results/',
                        help='Directory to store results')
    parser.add_argument('--override', action='store_true',
                        help='Override existing results')

    args = parser.parse_args()

    device = torch.device(f'cuda:{args.device}' if torch.cuda.is_available() else 'cpu')
  
    test_sets = args.testsets
    test_splits = args.testsplits
    batch_size = args.batch
    num_workers = args.workers
    max_num_videos_per_label = args.num_video
    model_path: Path = args.model_path
    results_dir: Path = args.results_dir
    override: bool = args.override
    debug: bool = args.debug
    ffpp_df_path = args.ffpp_faces_df_path
    ffpp_faces_dir = args.ffpp_faces_dir

    face_policy = str(model_path).split('face-')[1].split('_')[0]
    patch_size = int(str(model_path).split('size-')[1].split('_')[0])
    net_name = str(model_path).split('net-')[1].split('_')[0]
    model_name = '_'.join(model_path.with_suffix('').parts[-2:])

    # Load model
    print('Loading model...')
    net_class = getattr(fornet, net_name)
    state_tmp = torch.load(model_path, map_location='cpu')
    state = {'net': OrderedDict()}
    if 'net' in state_tmp:
        state = state_tmp
    else:
        for k, v in state_tmp.items():
            state['net'][f'model.{k}'] = v
    net: FeatureExtractor = net_class().eval().to(device)
    incomp_keys = net.load_state_dict(state['net'], strict=True)
    print(incomp_keys)
    print('Model loaded.')

    # Transformer
    test_transformer = utils.get_transformer(face_policy, patch_size, net.get_normalizer(), train=False)

    print('Loading data...')
    if ffpp_df_path is None or ffpp_faces_dir is None:
        raise RuntimeError('Specify DataFrame and directory for FF++ faces!')
    
    splits = split.make_splits(ffpp_df=ffpp_df_path, ffpp_dir=ffpp_faces_dir, dbs={'train': test_sets, 'val': test_sets, 'test': test_sets})

    # Extract DataFrames + roots
    train_dfs = [splits['train'][db][0] for db in splits['train']]
    train_roots = [splits['train'][db][1] for db in splits['train']]
    val_dfs = [splits['val'][db][0] for db in splits['val']]
    val_roots = [splits['val'][db][1] for db in splits['val']]
    test_dfs = [splits['test'][db][0] for db in splits['test']]
    test_roots = [splits['test'][db][1] for db in splits['test']]

    # Output dir
    out_folder = results_dir.joinpath(model_name)
    out_folder.mkdir(mode=0o775, parents=True, exist_ok=True)

    # Optional: Limit videos
    def select_dfs(dfs):
        return [select_videos(df, max_num_videos_per_label) for df in dfs] if max_num_videos_per_label else dfs

    dfs_out_train = select_dfs(train_dfs)
    dfs_out_val = select_dfs(val_dfs)
    dfs_out_test = select_dfs(test_dfs)

    # Build list of datasets to run
    extr_list = []
    if 'train' in test_splits:
        for idx, dataset in enumerate(test_sets):
            extr_list.append((dfs_out_train[idx], out_folder / f'{dataset}_train.pkl', train_roots[idx], f'{dataset} TRAIN'))
    if 'val' in test_splits:
        for idx, dataset in enumerate(test_sets):
            extr_list.append((dfs_out_val[idx], out_folder / f'{dataset}_val.pkl', val_roots[idx], f'{dataset} VAL'))
    if 'test' in test_splits:
        for idx, dataset in enumerate(test_sets):
            extr_list.append((dfs_out_test[idx], out_folder / f'{dataset}_test.pkl', test_roots[idx], f'{dataset} TEST'))

    # Run prediction
    for df, df_path, df_root, tag in extr_list:
        if override or not df_path.exists():
            print(f'\n##### PREDICT VIDEOS FROM {tag} #####')
            print(f'Real frames: {sum(df["label"] == False)}')
            print(f'Fake frames: {sum(df["label"] == True)}')
            print(f'Real videos: {df[df["label"] == False]["video"].nunique()}')
            print(f'Fake videos: {df[df["label"] == True]["video"].nunique()}')

            result = process_dataset(df=df, root=df_root, net=net, criterion=nn.BCEWithLogitsLoss(reduction='none'),
                                     patch_size=patch_size, face_policy=face_policy, transformer=test_transformer,
                                     batch_size=batch_size, num_workers=num_workers, device=device)

            df['score'] = result['score'].astype(np.float32)
            df['loss'] = result['loss'].astype(np.float32)
            df.to_pickle(str(df_path))
            print(f'Saved to: {df_path}')

            if debug:
                plt.figure()
                plt.title(tag)
                plt.hist(df[df.label == True].score, bins=100, alpha=0.6, label='FAKE frames')
                plt.hist(df[df.label == False].score, bins=100, alpha=0.6, label='REAL frames')
                plt.legend()

            del result, df
            gc.collect()

    if debug:
        plt.show()

    print('Completed!')


def process_dataset(df: pd.DataFrame, root: str, net: FeatureExtractor, criterion,
                    patch_size: int, face_policy: str, transformer: A.BasicTransform,
                    batch_size: int, num_workers: int, device: torch.device) -> dict:
    dataset = FrameFaceDatasetTest(root=root, df=df, size=patch_size, scale=face_policy, transformer=transformer)

    score = np.zeros(len(df))
    loss = np.zeros(len(df))

    loader = DataLoader(dataset, batch_size=batch_size, num_workers=num_workers, shuffle=False, drop_last=False)
    with torch.no_grad():
        idx0 = 0
        for batch_data in tqdm(loader):
            images = batch_data[0].to(device)
            labels = batch_data[1].to(device)
            out = net(images)
            batch_loss = criterion(out, labels)
            batch_size_actual = len(images)
            score[idx0:idx0 + batch_size_actual] = out.cpu().numpy()[:, 0]
            loss[idx0:idx0 + batch_size_actual] = batch_loss.cpu().numpy()[:, 0]
            idx0 += batch_size_actual

    return {'score': score, 'loss': loss}


def select_videos(df: pd.DataFrame, max_videos_per_label: int) -> pd.DataFrame:
    st0 = np.random.get_state()
    np.random.seed(42)

    def select(df_label, n):
        videos = df_label['video'].unique()
        selected = np.random.choice(videos, min(n, len(videos)), replace=False)
        return df_label[df_label['video'].isin(selected)]

    df_fake = select(df[df.label == True], max_videos_per_label)
    df_real = select(df[df.label == False], max_videos_per_label)

    np.random.set_state(st0)
    return pd.concat([df_fake, df_real], axis=0, verify_integrity=True).copy()


if __name__ == '__main__':
    main()
