import os

HOME_ROOT = ''
DATASETS_ROOT = ''
MAIN_RUN_DIR = ''

path_cache_fs = os.path.join(HOME_ROOT, 'data_caches')

path_data_out = os.path.join(HOME_ROOT, 'data_output')

path_midlevel_root = os.path.join(DATASETS_ROOT, 'MidlevelFeatures')
path_midlevel_annotations_dir = os.path.join(path_midlevel_root, 'metadata_annotations')
path_midlevel_annotations = os.path.join(path_midlevel_annotations_dir, 'annotations.csv')
path_midlevel_metadata = os.path.join(path_midlevel_annotations_dir, 'metadata.csv')
path_midlevel_metadata_piano = os.path.join(path_midlevel_annotations_dir, 'metadata_piano.csv')
path_midlevel_metadata_instruments = os.path.join(path_midlevel_annotations_dir, 'metadata_domains.csv')
path_midlevel_audio_dir = os.path.join(path_midlevel_root, 'audio')
path_midlevel_compressed_audio_dir = os.path.join(path_midlevel_root, 'audio_compressed/audio')


path_mtgjamendo_audio = os.path.join(DATASETS_ROOT, 'MTG-Jamendo/audio')
path_mtgjamendo_raw_30s_labels = os.path.join(DATASETS_ROOT, 'MTG-Jamendo/MTG-Jamendo_annotations/raw_30s.tsv')
path_mtgjamendo_raw_30s_labels_processed = os.path.join(DATASETS_ROOT, 'MTG-Jamendo/MTG-Jamendo_annotations/raw_30s_processed.tsv')
path_mtgjamendo_raw_30s_labels_50artists_processed = os.path.join(DATASETS_ROOT, 'MTG-Jamendo/annotations/raw_30s_cleantags_50artists_processed.tsv')
path_mtgjamendo_raw_30s_labels_50artists = os.path.join(DATASETS_ROOT, 'MTG-Jamendo/annotations/raw_30s_cleantags_50artists.tsv')
path_mtgjamendo_mood_labels_split ={
    "train": os.path.join(DATASETS_ROOT, 'MTG-Jamendo/annotations/train_processed.tsv'),
    "validation": os.path.join(DATASETS_ROOT, 'MTG-Jamendo/annotations/validation_processed.tsv'),
    "test": os.path.join(DATASETS_ROOT, 'MTG-Jamendo/annotations/test_processed.tsv')
}

