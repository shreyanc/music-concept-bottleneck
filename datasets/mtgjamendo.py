from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MultiLabelBinarizer
from torch.utils.data import Dataset

from datasets.dataset_base import DatasetBase
from datasets.dataset_utils import slice_func, normalize_spec, get_dataset_stats
from datasets.midlevel import MidlevelDataset
from helpers.audio import MadmomAudioProcessor
import pandas as pd
import numpy as np

from helpers.specaugment import SpecAugment
from utils import *
from paths import *
import logging

logger = logging.getLogger()


def get_subsets(df=None, subsets=None, op='and', invert_selection=False):
    """Returns a dataframe with tracks which have entries the subsets listed in variable subsets.
    For example, if subsets = ['GENRE', 'MOOD'], it will return tracks which have both GENRE and MOOD tags.
    If invert_selection is True, returns tracks which do not have entries for the subsets."""
    if df is None:
        df = pd.read_csv(path_mtgjamendo_raw_30s_labels_50artists_processed, sep='\t')

    if subsets is None:
        subsets = ['GENRE', 'MOOD']

    assert op in ['and', 'or']

    select = [op == 'and'] * len(df)
    for s in subsets:
        if op == 'and':
            select &= df[s].notna()
        else:
            select |= df[s].notna()

    if invert_selection:
        select = np.invert(select)

    df_r = df[select]
    return df_r, select


def filter_subset(df=None, filter=None, subset=None, exclusive=False):
    """Returns two dataframes. First one with tracks which contain the tag 'filter' under the subset 'subset',
    and the second one with tracks that don't"""
    if not exclusive:
        select = df.apply(
            lambda r: filter in r[subset].replace(' ', '').split(',') if type(r[subset]) == str else False, axis=1)
    else:
        select = df.apply(lambda r: filter in r[subset].split(',') and len(r[subset].split(',')) == 1 if type(
            r[subset]) == str else False, axis=1)
    df_r_in, df_r_out = df[select], df[~select]
    return df_r_in, df_r_out, select


def load_mtgjamendo_dfs(seed=None, mediaeval=False, subsets=None, trainset_fraction=None):
    if trainset_fraction is None:
        trainset_fraction = 1.0

    if subsets is None:
        subsets = ['MOOD']

    if mediaeval:
        tr_df = pd.read_csv(path_mtgjamendo_mood_labels_split['train'], sep='\t')
        va_df = pd.read_csv(path_mtgjamendo_mood_labels_split['validation'], sep='\t')
        te_df = pd.read_csv(path_mtgjamendo_mood_labels_split['test'], sep='\t')
        mtg_df = pd.concat([tr_df, va_df, te_df])
    else:
        mtg_df = get_subsets(subsets=subsets)[0]
        tr_df, te_df = train_test_split(mtg_df, test_size=8033, random_state=seed)
        te_df, va_df = train_test_split(te_df, test_size=3802, random_state=seed)

    subset_tags = {s: mtg_df[s].dropna().values for s in subsets}

    mlbs = {}
    for s, stags in subset_tags.items():
        labels = []
        mlb_subset = MultiLabelBinarizer()
        for tags in stags:
            labels.append(set([i.strip() for i in tags.split(",")]))
        mlb_subset = mlb_subset.fit(labels)
        mlbs[s] = mlb_subset

    tr_df, va_df = tr_df.sample(frac=trainset_fraction, axis=0), va_df.sample(frac=trainset_fraction, axis=0)

    return tr_df, va_df, te_df, mlbs


class MTGJamendoDataset(DatasetBase):
    def __init__(self, **kwargs):
        self.binarizer = kwargs.get('binarizer')

        assert (subset := kwargs.get('subset')) in ['GENRE', 'INSTRUMENT', 'MOOD'], print("subset must be in ['GENRE', 'INSTRUMENT', 'MOOD'], is", subset)
        self.subset = subset

        if (ann_df := kwargs.pop('annotations')) is None:
            ann_df = pd.read_csv(path_mtgjamendo_raw_30s_labels_50artists_processed, sep='\t')

        super().__init__(name='mtgjamendo', annotations=ann_df, scale_labels=False, **kwargs)

        self.verbose = kwargs.get('verbose', False)

    def _load_annotations(self, annotations):
        assert isinstance(annotations, pd.DataFrame)
        lbls = []
        for tg in annotations[self.subset].dropna().values:
            lbls.append(set([i.strip() for i in tg.split(",")]))
        bins = self.binarizer.transform(lbls)
        annotations[f'{self.subset}_binary'] = bins.tolist()
        return annotations

    def _get_label_names_from_annotations_df(self):
        return self.subset + '_binary'

    def _get_id_col(self):
        return 'PATH'

    def _get_song_id_from_idx(self, idx):
        return self.song_id_list[idx]

    def _get_audio_path_from_idx(self, idx):
        return self.song_id_list[idx]

    def _get_spectrogram(self, audio_path, dataset_cache_dir):
        mtg_folder, mtg_aud_name = os.path.split(audio_path)
        full_audio_path = os.path.join(path_mtgjamendo_audio, audio_path)
        specpath = os.path.join(dataset_cache_dir, self.processor.get_params.get("name"), mtg_folder.split('/')[-1], mtg_aud_name.split('.')[0] + '.npy')
        specdir = os.path.split(specpath)[0]
        if not os.path.exists(specdir):
            os.makedirs(specdir)
        try:
            return np.load(specpath)
        except Exception as e:
            # if self.verbose:
            print(f"Could not load {specpath} -- {e}")
            print(f"Calculating spectrogram for {audio_path} and saving to {specpath}")
            spec_obj = self.processor(full_audio_path)
            np.save(specpath, spec_obj.spec)
            return spec_obj.spec

    def _get_labels(self, song_id):
        return torch.tensor(self.annotations.loc[self.annotations[self.id_col] == song_id][self.label_names].values[0])


class MTGGenericDataset(Dataset):
    def __init__(self, cache_dir=path_cache_fs,
                 name="mtgjamendo",
                 df=None,
                 duration=15,
                 return_labels=True,
                 binarizer=None,
                 audio_processor=None,
                 slice_mode='start',
                 **kwargs):
        self.dset_name = name
        self.binarizer = binarizer
        if audio_processor is None:
            self.processor = MadmomAudioProcessor(fps=31.3)
        else:
            self.processor = audio_processor
        self.cache_dir = cache_dir
        self.duration = duration
        self.dataset_cache_dir = os.path.join(cache_dir, name)
        if return_labels:
            self.audio_paths, self.labels = df['PATH'].values, df[df.columns[1:]].astype(np.double).values
        else:
            try:
                self.audio_paths = df['PATH'].values
            except:
                self.audio_paths = df.values

        self.return_labels = return_labels
        self.slice_mode = slice_mode

        if isinstance(kwargs.get('augment'), SpecAugment):
            self.augment = kwargs['augment']
        elif kwargs.get('augment') == 'none' or kwargs.get('augment') is None:
            self.augment = lambda x: x
        else:
            logger.info(f"WARNING: No spec augment function assigned -- got {kwargs.get('augment')}; should be SpecAugment instance or None or 'none'!")
            self.augment = lambda x: x

    def __getitem__(self, ind):
        song_path = self.audio_paths[ind]
        spec = self._get_spectrogram(song_path, self.dataset_cache_dir).spec
        slice_length = self.processor.times_to_frames(self.duration)
        spec_sliced, start_time, end_time = slice_func(spec, slice_length, self.processor, mode=self.slice_mode)
        spec_aug = self.augment(spec_sliced).astype(np.float32)
        if self.return_labels:
            labels = self.labels[ind]
            return song_path, torch.from_numpy(spec_aug), torch.from_numpy(labels)
        else:
            return song_path, torch.from_numpy(spec_aug)

    def _get_spectrogram(self, audio_path, dataset_cache_dir):
        mtg_folder, mtg_aud_name = os.path.split(audio_path)
        full_audio_path = os.path.join(path_mtgjamendo_audio, audio_path)
        specpath = os.path.join(dataset_cache_dir, self.processor.get_params.get("name"), mtg_folder.split('/')[-1], mtg_aud_name + '.specobj')
        specdir = os.path.split(specpath)[0]
        if not os.path.exists(specdir):
            os.makedirs(specdir)
        try:
            return pickleload(specpath)
        except Exception as e:
            print(f"Could not load {specpath} -- {e}")
            print(f"Calculating spectrogram for {audio_path} and saving to {specpath}")
            spec_obj = self.processor(full_audio_path)
            pickledump(spec_obj, specpath)
            return spec_obj

    def __len__(self):
        return len(self.audio_paths)


if __name__ == '__main__':
    tr, va, te, mlb = load_mtgjamendo_dfs(subsets=['GENRE', 'INSTRUMENT'])
    print(tr)
    mtg_dataset = MTGJamendoDataset(annotations=tr, binarizer=mlb['GENRE'], subset='GENRE', duration=30)
    # midlevel_dataset = MidlevelDataset(select_song_ids=[1, 2, 3, 4, 5], duration=30)
    print(mtg_dataset[2])
    # print(midlevel_dataset[2])
    print(tr.head())
