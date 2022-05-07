import json
import math
import os

import numpy as np
from torch.utils.data import Dataset

# from text import text_to_sequence
from utils.tools import pad_1D, pad_2D

import json
import re


def text_to_sequence(text, symbols_dict):
    return np.array(
        [symbols_dict[token] + 1 for token in text.split(" ")]
    )


class Dataset(Dataset):
    def __init__(
        self, filename, preprocess_config, train_config, sort=False, drop_last=False
    ):
        self.dataset_name = preprocess_config["dataset"]
        self.preprocessed_path = preprocess_config["path"]["preprocessed_path"]
        self.cleaners = preprocess_config["preprocessing"]["text"]["text_cleaners"]
        self.batch_size = train_config["optimizer"]["batch_size"]

        (
            self.basename,
            self.speaker,
            self.phone,
            self.phone_full_label,
        ) = self.process_meta(filename)
        # with open(os.path.join(self.preprocessed_path, "speakers.json")) as f:
        #    self.speaker_map = json.load(f)
        self.sort = sort
        self.drop_last = drop_last

        with open(preprocess_config["path"]["dict"]) as f:
            self.phone_dict = json.load(f)

    def __len__(self):
        return len(self.phone)

    """
    def text_to_sequence(self, text):
        phone_seq = []
        pingyin_state_seq = []
        prosodic_structure_seq = []
        for token in text.strip("{").strip("}").split(" "):
            if token == "-":
                pingyin_state_seq[-1] = 1
            elif token == "|":
                pingyin_state_seq[-1] = 2
            elif token == "^":
                prosodic_structure_seq[-1] = 2
            elif token == "*":
                prosodic_structure_seq[-1] = 3
            elif token == "&":
                prosodic_structure_seq[-1] = 4
            elif token == "$":
                prosodic_structure_seq[-1] = 5
            else:
                phone_seq.append(self.phone_dict[token] + 1)
                pingyin_state_seq.append(1)
                prosodic_structure_seq.append(1)
        return (
            np.array(phone_seq),
            np.array(pingyin_state_seq),
            np.array(prosodic_structure_seq),
        )
    """

    #def text_to_sequence(self, text):
    #    return np.array(
    #        [
    #            self.phone_dict[token] + 1
    #            for token in text.strip("{").strip("}").split(" ")
    #        ]
    #    )

    def __getitem__(self, idx):
        basename = self.basename[idx]
        speaker = self.speaker[idx]
        phone_full_label = self.phone_full_label[idx]
        #phone = text_to_sequence(phone_full_label,self.phone_dict)
        phone = np.array([self.phone_dict[token] + 1 for token in phone_full_label])
        mel_path = os.path.join(
            self.preprocessed_path,
            "mel",
            "{}-mel-{}.npy".format(speaker, basename),
        )
        mel = np.load(mel_path)
        pitch_path = os.path.join(
            self.preprocessed_path,
            "pitch",
            "{}-pitch-{}.npy".format(speaker, basename),
        )
        pitch = np.load(pitch_path)
        energy_path = os.path.join(
            self.preprocessed_path,
            "energy",
            "{}-energy-{}.npy".format(speaker, basename),
        )
        energy = np.load(energy_path)
        duration_path = os.path.join(
            self.preprocessed_path,
            "duration",
            "{}-duration-{}.npy".format(speaker, basename),
        )
        duration = np.load(duration_path)

        sample = {
            "id": basename,
            "speaker": speaker,
            "phone_full_label": phone_full_label,
            "phone": phone,
            "mel": mel,
            "pitch": pitch,
            "energy": energy,
            "duration": duration,
        }

        return sample

    def process_meta(self, filename):
        with open(
            os.path.join(self.preprocessed_path, filename), "r", encoding="utf-8"
        ) as f:
            meta_json = json.load(f)
        name = []
        speaker = []
        phone = []
        phone_full_label = []
        for n,s,t,_ in meta_json:
            name.append(n)
            speaker.append(s)
            phone.append(t)
            phone_full_label.append(t)
        return name, speaker, phone, phone_full_label

    def reprocess(self, data, idxs):
        ids = [data[idx]["id"] for idx in idxs]
        speakers = [data[idx]["speaker"] for idx in idxs]
        phones = [data[idx]["phone"] for idx in idxs]
        phone_full_label = [data[idx]["phone_full_label"] for idx in idxs]
        # pingyin_states = [data[idx]["pingyin_state"] for idx in idxs]
        # prosodic_structures = [data[idx]["prosodic_structure"] for idx in idxs]
        # raw_texts = [data[idx]["raw_text"] for idx in idxs]
        mels = [data[idx]["mel"] for idx in idxs]
        pitches = [data[idx]["pitch"] for idx in idxs]
        energies = [data[idx]["energy"] for idx in idxs]
        durations = [data[idx]["duration"] for idx in idxs]

        text_lens = np.array([phone.shape[0] for phone in phones])
        mel_lens = np.array([mel.shape[0] for mel in mels])

        speakers = np.array(speakers)
        phones = pad_1D(phones)
        mels = pad_2D(mels)
        pitches = pad_1D(pitches)
        energies = pad_1D(energies)
        durations = pad_1D(durations)

        pitches = np.nan_to_num(pitches)
        energies = np.nan_to_num(energies)
        return (
            ids,
            phone_full_label,
            speakers,
            phones,
            text_lens,
            max(text_lens),
            mels,
            mel_lens,
            max(mel_lens),
            pitches,
            energies,
            durations,
        )

    def collate_fn(self, data):
        data_size = len(data)

        if self.sort:
            len_arr = np.array([d["phone"].shape[0] for d in data])
            idx_arr = np.argsort(-len_arr)
        else:
            idx_arr = np.arange(data_size)

        tail = idx_arr[len(idx_arr) - (len(idx_arr) % self.batch_size) :]
        idx_arr = idx_arr[: len(idx_arr) - (len(idx_arr) % self.batch_size)]
        idx_arr = idx_arr.reshape((-1, self.batch_size)).tolist()
        if not self.drop_last and len(tail) > 0:
            idx_arr += [tail.tolist()]

        output = list()
        for idx in idx_arr:
            output.append(self.reprocess(data, idx))

        return output


class TextDataset(Dataset):
    def __init__(self, filepath, preprocess_config):
        self.dataset_name = preprocess_config["dataset"]
        self.preprocessed_path = preprocess_config["path"]["preprocessed_path"]
        # self.cleaners = preprocess_config["preprocessing"]["text"]["text_cleaners"]

        (
            self.basename,
            self.speaker,
            self.phone,
            self.phone_full_label,
        ) = self.process_meta(filepath)
        # with open(
        #    os.path.join(
        #        preprocess_config["path"]["preprocessed_path"], "speakers.json"
        #    )
        # ) as f:
        #    self.speaker_map = json.load(f)

        with open(preprocess_config["path"]["dict"]) as f:
            self.phone_dict = json.load(f)

    def __len__(self):
        return len(self.phone)

    def text_to_sequence(self, text):
        phone_seq = []
        pingyin_state_seq = []
        prosodic_structure_seq = []
        for token in text.strip("{").strip("}").split(" "):
            if token == "-":
                pingyin_state_seq[-1] = 1
            elif token == "|":
                pingyin_state_seq[-1] = 2
            elif token == "^":
                prosodic_structure_seq[-1] = 2
            elif token == "*":
                prosodic_structure_seq[-1] = 3
            elif token == "&":
                prosodic_structure_seq[-1] = 4
            elif token == "$":
                prosodic_structure_seq[-1] = 5
            else:
                phone_seq.append(self.phone_dict[token] + 1)
                pingyin_state_seq.append(1)
                prosodic_structure_seq.append(1)
        return (
            np.array(phone_seq),
            np.array(pingyin_state_seq),
            np.array(prosodic_structure_seq),
        )

    def __getitem__(self, idx):
        basename = self.basename[idx]
        speaker = self.speaker[idx]
        # speaker_id = self.speaker_map[speaker]
        phone_full_label = self.phone_full_label[idx]
        #phone, pingyin_state, prosodic_structure = self.text_to_sequence(
        #    phone_full_label
        #)
        phone = np.array([self.phone_dict[token] + 1 for token in phone_full_label])
        # phone = np.array(text_to_sequence(self.text[idx], self.cleaners))

        return (
            basename,
            speaker,
            phone_full_label,
            phone,
        )

    def process_meta(self, filename):
        with open(
            os.path.join(self.preprocessed_path, filename), "r", encoding="utf-8"
        ) as f:
            meta_json = json.load(f)
        name = []
        speaker = []
        phone = []
        phone_full_label = []
        for n,s,t,_ in meta_json:
            name.append(n)
            speaker.append(s)
            phone.append(t)
            phone_full_label.append(t)
        return name, speaker, phone, phone_full_label

    def collate_fn(self, data):
        ids = [d[0] for d in data]
        speakers = np.array([d[1] for d in data])
        phone_full_labels = [d[2] for d in data]
        phone = [d[3] for d in data]
        text_lens = np.array([p.shape[0] for p in phone])

        phone = pad_1D(phone)

        return (
            ids,
            phone_full_labels,
            speakers,
            phone,
            text_lens,
            max(text_lens),
        )


if __name__ == "__main__":
    # Test
    import torch
    import yaml
    from torch.utils.data import DataLoader
    from utils.tools import to_device

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    preprocess_config = yaml.load(
        open("./config/LJSpeech/preprocess.yaml", "r"), Loader=yaml.FullLoader
    )
    train_config = yaml.load(
        open("./config/LJSpeech/train.yaml", "r"), Loader=yaml.FullLoader
    )

    train_dataset = Dataset(
        "train.txt", preprocess_config, train_config, sort=True, drop_last=True
    )
    val_dataset = Dataset(
        "val.txt", preprocess_config, train_config, sort=False, drop_last=False
    )
    train_loader = DataLoader(
        train_dataset,
        batch_size=train_config["optimizer"]["batch_size"] * 4,
        shuffle=True,
        collate_fn=train_dataset.collate_fn,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=train_config["optimizer"]["batch_size"],
        shuffle=False,
        collate_fn=val_dataset.collate_fn,
    )

    n_batch = 0
    for batchs in train_loader:
        for batch in batchs:
            to_device(batch, device)
            n_batch += 1
    print(
        "Training set  with size {} is composed of {} batches.".format(
            len(train_dataset), n_batch
        )
    )

    n_batch = 0
    for batchs in val_loader:
        for batch in batchs:
            to_device(batch, device)
            n_batch += 1
    print(
        "Validation set  with size {} is composed of {} batches.".format(
            len(val_dataset), n_batch
        )
    )
