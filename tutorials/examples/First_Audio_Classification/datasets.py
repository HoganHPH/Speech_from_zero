import os
from torch.utils.data import Dataset, DataLoader
from audio_utils import AudioUtil
from prepare_data import load_metadata
from torch.utils.data import random_split

# ----------------------------
# Sound Dataset
# ----------------------------
class SoundDataset(Dataset):
    def __init__(self, df, data_path='./DATA'):
        self.df = df
        self.data_path = str(data_path)
        self.duration = 4000
        self.sr = 44100
        self.channel = 2
        self.shift_pct = 0.4
        
    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, idx):
        # Absolute file path of the audio file - concatenate the audio directory with
        # the relative path
        
        audio_file = os.path.join(self.data_path, self.df.loc[idx, 'relative_path'])
        class_id = self.df.loc[idx, 'classID']
        
        aud = AudioUtil.open(audio_file=audio_file)
        # Some sounds have a higher sample rate, or fewer channels compared to the
        # majority. So make all sounds have the same number of channels and same 
        # sample rate. Unless the sample rate is the same, the pad_trunc will still
        # result in arrays of different lengths, even though the sound duration is
        # the same.
        reaud = AudioUtil.resample(aud, self.sr)
        rechan = AudioUtil.rechannel(reaud, self.channel)
        
        dur_aud = AudioUtil.pad_trunc(rechan, self.duration)
        shift_aurd = AudioUtil.time_shift(dur_aud, self.shift_pct)
        sgram = AudioUtil.spectro_gram(shift_aurd, n_mels=64, n_fft=1024, hop_len=None)
        aug_sgram = AudioUtil.spectro_augment(sgram, max_mask_pct=0.1, n_freq_masks=2, n_time_masks=2)
        
        return aug_sgram, class_id
    

# ----------------------------
# Sound Data Loader
# ----------------------------
class SoundDataLoader:
    def __init__(self, data_path='./DATA', 
                 test_size=0.2,
                 batch_size=64,
                 phase='train'):
        self.data_path = data_path
        self.test_size = test_size
        self.batch_size = batch_size
        self.phase = phase
        self.df = load_metadata(root_data=data_path)
        
    
    def load_data(self):
        soundDataset = SoundDataset(df=self.df, data_path=self.data_path)
        
        # Random split training and validation
        num_items = len(soundDataset)
        num_train = round(num_items * (1 - self.test_size))
        num_val = num_items - num_train
        train_ds, val_ds = random_split(soundDataset, [num_train, num_val])
        
        # Create training and validation data loaders
        if self.phase == 'train':
            train_dl = DataLoader(dataset=train_ds, batch_size=self.batch_size, shuffle=True)
            return train_dl
        elif self.phase == 'test':
            val_dl = DataLoader(dataset=val_ds, batch_size=self.batch_size, shuffle=False)
            return val_dl
           