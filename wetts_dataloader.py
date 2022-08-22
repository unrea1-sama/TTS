import torch
import json
import torchaudio

from mel_processing import spectrogram_torch


def read_jsonl(path):
    with open(path) as f:
        return [json.loads(line.strip()) for line in f]


def read_id(path):
    """
    id file: token id
    """
    id_dict = {}
    with open(path) as f:
        for line in f:
            token, id = line.strip().split()
            id_dict[token] = int(id)
    return id_dict


class WeTTSDataset(torch.utils.data.Dataset):

    def __init__(self,
                 datalist,
                 phn2id_path,
                 spk2id_path,
                 sr=22050,
                 n_fft=1024,
                 hop_size=256,
                 win_size=1024) -> None:
        super().__init__()
        self.data = None
        self.datalist = datalist
        self.sr = sr
        self.phn2id_path = phn2id_path
        self.spk2id_path = spk2id_path
        self.n_fft = n_fft
        self.hop_size = hop_size
        self.win_size = win_size

    def load_dataset_meta_info(self):
        if self.data is None:
            self.data = read_jsonl(self.datalist)
            self.phn2id = read_id(self.phn2id_path)
            self.spk2id = read_id(self.spk2id_path)

    def __len__(self):
        self.load_dataset_meta_info()
        return len(self.data)

    def __getitem__(self, index):
        self.load_dataset_meta_info()
        wav = self.get_wav(self.data[index]['wav_path'])
        spec = self.get_spec(wav)
        text = self.get_text(self.data[index]['text'])
        speaker = self.get_speaker(self.data[index]['speaker'])
        key = self.data[index]['key']
        return {
            'key': key,
            'wav': wav.T,
            'spec': spec.squeeze(0).T,  # (t,d)
            'text': text,
            'speaker': speaker
        }

    def get_speaker(self, speaker):
        return torch.LongTensor([self.spk2id[speaker]])

    def get_wav(self, path):
        wav, original_sr = torchaudio.load(path)
        if original_sr != self.sr:
            wav = torchaudio.functional.resample(wav, original_sr,
                                                 self.sr).clamp(min=-1, max=1)
        # truncate to make sure wav has same length as spectrogram
        wav = wav[:, :wav.size(1) // 256 * 256]
        return wav

    def get_spec(self, wav):
        # sample rate is not used here, set to 0
        return spectrogram_torch(wav, self.n_fft, 0, self.hop_size,
                                 self.win_size)

    def get_text(self, text):
        """Inserting blank tokens in between and convert all phonemes to
        corresponding id
        """
        new_text = ['<unk>'] * (len(text) * 2 - 1)
        new_text[::2] = text
        return torch.LongTensor([self.phn2id[token] for token in new_text])

    @property
    def lengths(self):
        self.load_dataset_meta_info()
        return [len(x['text']) for x in self.data]


def collate(batch):
    sorted_text_length, sorted_idx = torch.sort(torch.LongTensor(
        [len(sample['text']) for sample in batch]),
                                                descending=True)
    sorted_spec_length = torch.LongTensor(
        [batch[i]['spec'].size(0) for i in sorted_idx])
    sorted_wav_length = torch.LongTensor(
        [batch[i]['wav'].size(0) for i in sorted_idx])

    sorted_spec = torch.nn.utils.rnn.pad_sequence(
        [batch[i]['spec'] for i in sorted_idx], batch_first=True)
    sorted_wav = torch.nn.utils.rnn.pad_sequence(
        [batch[i]['wav'] for i in sorted_idx], batch_first=True)
    sorted_text = torch.nn.utils.rnn.pad_sequence(
        [batch[i]['text'] for i in sorted_idx], batch_first=True)
    sorted_speaker = torch.cat([batch[i]['speaker'] for i in sorted_idx])
    return (sorted_text, sorted_text_length,
            sorted_spec.permute(0, 2, 1), sorted_spec_length,
            sorted_wav.permute(0, 2, 1), sorted_wav_length, sorted_speaker)


if __name__ == '__main__':
    dataset = WeTTSDataset('test_datalist.jsonl', 'phn2id', 'spk2id')
    (sorted_text, sorted_text_length, sorted_spec, sorted_spec_length,
     sorted_wav, sorted_wav_length,
     sorted_speaker) = collate([dataset[i] for i in range(3)])
    print(sorted_text.shape, sorted_text_length.shape, sorted_spec.shape,
          sorted_spec_length.shape, sorted_wav.shape, sorted_wav_length.shape,
          sorted_speaker.shape)
    print(sorted_wav_length)
