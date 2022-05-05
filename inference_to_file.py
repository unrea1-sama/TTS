import os
import json
import math
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader
from argparse import ArgumentParser
from tqdm import tqdm
import commons
import utils
from data_utils import (
    TextAudioLoader,
    TextAudioCollate,
    TextAudioSpeakerLoader,
    TextAudioSpeakerCollate,
)
from models import SynthesizerTrn

from scipy.io.wavfile import write

count = 1


def write_wave_file(
    audios, lengths, hop_length, output_dir, max_wav_value, sampling_rate
):
    """
    audios: b*1*max_t
    lengths: b*1
    """
    global count
    for audio, length in zip(audios, lengths):
        write(
            os.path.join(output_dir, "{}.wav".format(count)),
            sampling_rate,
            (audio[: int(length) * hop_length] * max_wav_value)
            .cpu()
            .numpy()
            .astype("int16"),
        )
        count += 1


def main(args):
    os.makedirs(args.output_dir, exist_ok=True)
    hps = utils.get_hparams_from_file(args.hparams)
    net_g = SynthesizerTrn(
        hps.data.symbols,
        hps.data.filter_length // 2 + 1,
        hps.train.segment_size // hps.data.hop_length,
        **hps.model,
        use_sdp=args.use_sdp
    ).cuda()
    _ = net_g.eval()
    _ = utils.load_checkpoint(args.checkpoint, net_g, None)

    inference_dataset = TextAudioLoader(hps.data.test_files, hps.data)
    inference_dataloader = DataLoader(
        inference_dataset, 16, False, collate_fn=TextAudioCollate()
    )
    for text_padded, text_lengths, _, _, _, _ in tqdm(inference_dataloader):
        with torch.no_grad():
            audios, _, y_mask, _ = net_g.infer(
                text_padded.cuda(),
                text_lengths.cuda(),
                noise_scale=0.667,
                noise_scale_w=0.8,
                length_scale=1,
            )
            # audio: b*1*max_t

            lengths = y_mask.sum(dim=-1).squeeze()
            audios = audios.clamp(-1, 1).squeeze()
            write_wave_file(
                audios,
                lengths,
                hps.data.hop_length,
                args.output_dir,
                hps.data.max_wav_value,
                hps.data.sampling_rate,
            )


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--hparams", type=str, default="configs/myself.json")
    parser.add_argument("--checkpoint", type=str, default="logs/myself/G_401000.pth")
    parser.add_argument("--output_dir", type=str, default="output")
    parser.add_argument("--use_sdp", action="store_true")
    args = parser.parse_args()
    main(args)
