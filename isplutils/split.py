from typing import List, Dict, Tuple
import numpy as np
import pandas as pd


def load_df(ffpp_df_path: str, ffpp_faces_dir: str, dataset: str) -> (pd.DataFrame, str):
    df = pd.read_pickle(ffpp_df_path)
    root = ffpp_faces_dir
    return df, root


def get_split_df(df: pd.DataFrame, split: str) -> pd.DataFrame:
    st0 = np.random.get_state()
    np.random.seed(41)

    crf = 'c23'
    random_youtube_videos = np.random.permutation(
        df[(df['source'] == 'youtube') & (df['quality'] == crf)]['video'].unique()
    )
    total = len(random_youtube_videos)
    n_train = int(0.75 * total)
    n_val = int(0.1 * total)
    n_test = total - n_train - n_val  

    train_orig = random_youtube_videos[:n_train]
    val_orig = random_youtube_videos[n_train:n_train + n_val]
    test_orig = random_youtube_videos[n_train + n_val:]

    
    if split == 'train':
        split_df = pd.concat((df[df['original'].isin(train_orig)], df[df['video'].isin(train_orig)]), axis=0)
    elif split == 'val':
        split_df = pd.concat((df[df['original'].isin(val_orig)], df[df['video'].isin(val_orig)]), axis=0)
    elif split == 'test':
        split_df = pd.concat((df[df['original'].isin(test_orig)], df[df['video'].isin(test_orig)]), axis=0)
    else:
        raise NotImplementedError(f'Unknown split: {split}')

    
    np.random.set_state(st0)
    print(f"{split.upper()} set: {split_df['video'].nunique()} unique videos")
    all_videos = split_df['video'].unique()
    real_videos = split_df[split_df['label'] == 0]['video'].unique()
    fake_videos = split_df[split_df['label'] == 1]['video'].unique()

    print(f"{split.upper()} set: {len(all_videos)} unique videos")
    print(f"{split.upper()} set: {len(real_videos)} real videos, {len(fake_videos)} fake videos")
    return split_df


def make_splits(ffpp_df: str, ffpp_dir: str, dbs: Dict[str, List[str]]) -> Dict[str, Dict[str, Tuple[pd.DataFrame, str]]]:
    split_dict = {}
    full_df, root = load_df(ffpp_df, ffpp_dir, 'ff-c23-720-140-140')

    for split_name, split_dbs in dbs.items():
        split_dict[split_name] = {}
        for split_db in split_dbs:
            split_df = get_split_df(df=full_df, split=split_name)
            split_dict[split_name][split_db] = (split_df, root)

    return split_dict
