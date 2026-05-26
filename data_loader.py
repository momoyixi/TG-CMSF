# data_loader.py
import logging
import pickle
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

__all__ = ['MMDataLoader']
logger = logging.getLogger('MMSA')

# ===== Missing Modality Experiment Defaults (edit here if you want) =====
# 'none' / 'text' / 'audio' / 'vision' / ['audio','vision'] 
DEFAULT_MISSING_MODALITY = 'text'        
DEFAULT_MISSING_RATE = 0               
DEFAULT_MISSING_SEED = 42              
DEFAULT_MISSING_APPLY = ['test'] 
# ======================================================================


class MMDataset(Dataset):
    def __init__(self, args, mode='train'):
        self.mode = mode
        self.args = args

        DATASET_MAP = {
            'mosi': self.__init_mosi,
            'mosei': self.__init_mosei,
            # 'sims': self.__init_sims,
        }
        if args.get('dataset_name') not in DATASET_MAP:
            raise ValueError(f"Unsupported dataset_name: {args.get('dataset_name')}")

        DATASET_MAP[args['dataset_name']]()

    def __init_mosi(self):
        with open(self.args['featurePath'], 'rb') as f:
            data = pickle.load(f)

        # ----- load base features -----
        if self.args.get('use_bert', False):
            self.text = data[self.mode]['text_bert'].astype(np.float32)
        else:
            self.text = data[self.mode]['text'].astype(np.float32)

        self.vision = data[self.mode]['vision'].astype(np.float32)
        self.audio = data[self.mode]['audio'].astype(np.float32)

        self.raw_text = data[self.mode]['raw_text']
        self.ids = data[self.mode]['id']

        # ----- optional replacement feature files -----
        if self.args.get('feature_T', "") != "":
            with open(self.args['feature_T'], 'rb') as f:
                data_T = pickle.load(f)
            if self.args.get('use_bert', False):
                self.text = data_T[self.mode]['text_bert'].astype(np.float32)
                self.args['feature_dims'][0] = 768
            else:
                self.text = data_T[self.mode]['text'].astype(np.float32)
                self.args['feature_dims'][0] = self.text.shape[2]

        if self.args.get('feature_A', "") != "":
            with open(self.args['feature_A'], 'rb') as f:
                data_A = pickle.load(f)
            self.audio = data_A[self.mode]['audio'].astype(np.float32)
            self.args['feature_dims'][1] = self.audio.shape[2]

        if self.args.get('feature_V', "") != "":
            with open(self.args['feature_V'], 'rb') as f:
                data_V = pickle.load(f)
            self.vision = data_V[self.mode]['vision'].astype(np.float32)
            self.args['feature_dims'][2] = self.vision.shape[2]

        # ----- labels -----
        self.labels = {
            'M': np.array(data[self.mode]['regression_labels']).astype(np.float32)
        }
        logger.info(f"{self.mode} samples: {self.labels['M'].shape}")

        # ----- lengths (when not aligned) -----
        if not self.args.get('need_data_aligned', True):
            if self.args.get('feature_A', "") != "":
                self.audio_lengths = list(data_A[self.mode]['audio_lengths'])
            else:
                self.audio_lengths = data[self.mode]['audio_lengths']

            if self.args.get('feature_V', "") != "":
                self.vision_lengths = list(data_V[self.mode]['vision_lengths'])
            else:
                self.vision_lengths = data[self.mode]['vision_lengths']

        # sanitize
        self.audio[self.audio == -np.inf] = 0

        # normalize (optional)
        if self.args.get('need_normalized', False):
            self.__normalize()

        # build missing-modality flags (important!)
        self._build_missing_flags()

    def __init_mosei(self):
        return self.__init_mosi()

    def __init_sims(self):
        return self.__init_mosi()

    def __truncate(self):
        def do_truncate(modal_features, length):
            if length == modal_features.shape[1]:
                return modal_features
            truncated_feature = []
            padding = np.array([0 for _ in range(modal_features.shape[2])])
            for instance in modal_features:
                for index in range(modal_features.shape[1]):
                    if (instance[index] == padding).all():
                        if index + length >= modal_features.shape[1]:
                            truncated_feature.append(instance[index:index + 20])
                            break
                    else:
                        truncated_feature.append(instance[index:index + 20])
                        break
            truncated_feature = np.array(truncated_feature)
            return truncated_feature

        text_length, audio_length, video_length = self.args['seq_lens']
        self.vision = do_truncate(self.vision, video_length)
        self.text = do_truncate(self.text, text_length)
        self.audio = do_truncate(self.audio, audio_length)

    def __normalize(self):
        self.vision = np.mean(self.vision, axis=1, keepdims=True)
        self.audio = np.mean(self.audio, axis=1, keepdims=True)
        self.vision[self.vision != self.vision] = 0
        self.audio[self.audio != self.audio] = 0

    def _build_missing_flags(self):
        """
        missing_modality supports:
          - 'none'
          - 'text' / 'audio' / 'vision'
          - list/tuple: ['audio','vision'] for multi-missing
        missing_rate:
          - float in [0,1], applied to EACH modality in missing_modality
            e.g. modalities=['audio','vision'], rate=0.5 means:
                 50% samples missing audio, and independently 50% missing vision
            if you want "same samples missing both", see note below.
        """
        n = len(self.labels['M'])
        self.missing_flags = {
            'text': np.zeros(n, dtype=np.bool_),
            'audio': np.zeros(n, dtype=np.bool_),
            'vision': np.zeros(n, dtype=np.bool_),
        }

        modalities = self.args.get('missing_modality', 'none')
        rate = float(self.args.get('missing_rate', 0.0))
        apply_to = set(self.args.get('missing_apply_to', ['train']))

        if modalities in (None, 'none') or rate <= 0.0 or (self.mode not in apply_to):
            return

        # normalize to list
        if isinstance(modalities, str):
            modalities = [modalities]
        elif isinstance(modalities, (list, tuple)):
            modalities = list(modalities)
        else:
            raise ValueError(f"missing_modality must be str/list/tuple, got: {type(modalities)}")

        # clamp
        rate = max(0.0, min(1.0, rate))
        seed = int(self.args.get('missing_seed', 42))

        # mode-specific offset so train/valid/test don't share identical pattern
        mode_offset = {'train': 0, 'valid': 100, 'test': 200}.get(self.mode, 0)
        rng = np.random.RandomState(seed + mode_offset)

        # Option: whether to force the SAME sample subset missing for all modalities in the list
        # Default False: each modality missing independently.
        same_subset = bool(self.args.get('missing_same_subset', False))

        if same_subset and len(modalities) > 1:
            common_mask = (rng.rand(n) < rate)
            for m in modalities:
                if m not in ('text', 'audio', 'vision'):
                    raise ValueError(f"Wrong missing_modality: {m}")
                self.missing_flags[m] = common_mask
        else:
            for m in modalities:
                if m not in ('text', 'audio', 'vision'):
                    raise ValueError(f"Wrong missing_modality: {m}")
                self.missing_flags[m] = (rng.rand(n) < rate)

        # logging
        msg_parts = []
        for m in modalities:
            msg_parts.append(f"{m}={self.missing_flags[m].mean():.4f}")
        logger.info(
            f"[MissingModality] mode={self.mode} modalities={modalities} "
            f"rate={rate} same_subset={same_subset} seed={seed} "
            f"actual({', '.join(msg_parts)})"
        )

    def __len__(self):
        return len(self.labels['M'])

    def get_seq_len(self):
        if self.args.get('use_bert', False):
            # Keep original behavior for compatibility
            return (self.text.shape[2], self.audio.shape[1], self.vision.shape[1])
        else:
            return (self.text.shape[1], self.audio.shape[1], self.vision.shape[1])

    def get_feature_dim(self):
        return self.text.shape[2], self.audio.shape[2], self.vision.shape[2]

    def __getitem__(self, index):
        text = torch.tensor(self.text[index], dtype=torch.float32)
        audio = torch.tensor(self.audio[index], dtype=torch.float32)
        vision = torch.tensor(self.vision[index], dtype=torch.float32)

        # 1=present, 0=missing  (order: [T, A, V])
        m_text, m_audio, m_vision = 1.0, 1.0, 1.0

        if hasattr(self, 'missing_flags'):
            if self.missing_flags['text'][index]:
                text = torch.zeros_like(text)
                m_text = 0.0
            if self.missing_flags['audio'][index]:
                audio = torch.zeros_like(audio)
                m_audio = 0.0
            if self.missing_flags['vision'][index]:
                vision = torch.zeros_like(vision)
                m_vision = 0.0

        sample = {
            'raw_text': self.raw_text[index],
            'text': text,
            'audio': audio,
            'vision': vision,
            'missing_mask': torch.tensor([m_text, m_audio, m_vision], dtype=torch.float32),
            'index': index,
            'id': self.ids[index],
            'labels': {
                k: torch.tensor(v[index].reshape(-1), dtype=torch.float32)
                for k, v in self.labels.items()
            }
        }

        if not self.args.get('need_data_aligned', True):
            sample['audio_lengths'] = self.audio_lengths[index]
            sample['vision_lengths'] = self.vision_lengths[index]

        return sample


def MMDataLoader(args, num_workers):
    """
    args should contain:
      - dataset_name: 'mosi'/'mosei'
      - featurePath: path to pickle
      - batch_size
      - (optional) use_bert, need_data_aligned, need_normalized, feature_T/A/V ...
    missing-modality controls (optional):
      - missing_modality: 'text'/'audio'/'vision'/'none' or ['audio','vision']
      - missing_rate: float in [0,1]
      - missing_seed: int
      - missing_apply_to: list of splits, e.g. ['test']
      - missing_same_subset: bool, if True and modalities is list -> same samples missing for all
    """

    # ---- set defaults if not provided from outside ----
    args['missing_modality'] = args.get('missing_modality', DEFAULT_MISSING_MODALITY)
    args['missing_rate'] = args.get('missing_rate', DEFAULT_MISSING_RATE)
    args['missing_seed'] = args.get('missing_seed', DEFAULT_MISSING_SEED)
    args['missing_apply_to'] = args.get('missing_apply_to', DEFAULT_MISSING_APPLY)
    # extra: for multi-missing
    args['missing_same_subset'] = args.get('missing_same_subset', True)
    # --------------------------------------------------

    datasets = {
        'train': MMDataset(args, mode='train'),
        'valid': MMDataset(args, mode='valid'),
        'test': MMDataset(args, mode='test')
    }

    if 'seq_lens' in args:
        args['seq_lens'] = datasets['train'].get_seq_len()

    dataLoader = {
        ds: DataLoader(
            datasets[ds],
            batch_size=args['batch_size'],
            num_workers=num_workers,
            shuffle=True
        )
        for ds in datasets.keys()
    }

    return dataLoader
