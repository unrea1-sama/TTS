from argparse import ArgumentParser
import json
import numpy as np
import torch
import models
from utils import get_hparams_from_file, load_checkpoint
from data_utils import TextLoader, TextCollate
from torch.utils.data import DataLoader

import hifi_gan_models
import os
from scipy.io.wavfile import write
from tqdm import tqdm

count = 1


def parse_args():
    parser = ArgumentParser()
    parser.add_argument("--glow_hparams", type=str)
    parser.add_argument("--glow_ckpt", type=str)
    parser.add_argument("--input", type=str)
    parser.add_argument("--noise_scale", type=float, default=0.3)
    parser.add_argument("--length_scale", type=float, default=1.0)
    parser.add_argument("--vocoder_ckpt", type=str)
    parser.add_argument("--vocoder_config", type=str)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--output_dir", type=str, default="output")
    return parser.parse_args()


def write_wave_file(
    audios, lengths, hop_length, output_dir, max_wav_value, sampling_rate
):
    """
    audios: b*1*max_t
    lengths: b*1
    """
    global count
    for audio, length in zip(audios, lengths):
        length = length[0]
        audio = audio[0]
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
    glow_hparams = get_hparams_from_file(args.glow_hparams)
    hifi_gan_hparams = get_hparams_from_file(args.vocoder_config)

    model = models.FlowGenerator(
        glow_hparams.data.symbols,
        out_channels=glow_hparams.data.n_mel_channels,
        **glow_hparams.model
    ).to("cuda")
    load_checkpoint(args.glow_ckpt, model)
    model.decoder.store_inverse()
    model.eval()
    dataset = TextLoader(args.input, glow_hparams.data)
    collate_fn = TextCollate()
    loader = DataLoader(dataset, batch_size=args.batch_size, collate_fn=collate_fn)

    hifi_gan_generator = hifi_gan_models.Generator(hifi_gan_hparams).cuda()
    hifi_gan_state_dict = torch.load(args.vocoder_ckpt, map_location="cpu")
    hifi_gan_generator.load_state_dict(hifi_gan_state_dict["generator"])

    with torch.no_grad():
        for texts, lengths in tqdm(loader):
            texts = texts.cuda()
            lengths = lengths.cuda()
            (y, _, _, _, y_mask), (_, _, _), (attn_gen, _, _) = model(
                texts,
                lengths,
                gen=True,
                noise_scale=args.noise_scale,
                length_scale=args.length_scale,
            )
            # y: b*d*l
            # y_mask: b*1*l
            y_lengths = y_mask.sum(dim=-1)
            audios = hifi_gan_generator(y)
            # audios: b,1,l
            # y_lengths: b,1
            write_wave_file(
                audios,
                y_lengths,
                glow_hparams.data.hop_length,
                args.output_dir,
                glow_hparams.data.max_wav_value,
                hifi_gan_hparams.sampling_rate,
            )


if __name__ == "__main__":
    main(parse_args())
