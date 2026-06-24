#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Copyright 2021 Tencent Inc. (Author: Yougen Yuan).
# Apach 2.0

import argparse
import sys
import os

import pandas as pd
import os
import time
import torch
from AR.models.t2s_lightning_module import Text2SemanticLightningModule
from AR.utils.io import load_yaml_config
from text import cleaned_text_to_sequence
from text.cleaner import text_to_sequence, clean_text


import soundfile
import utils
from module.models import SynthesizerTrn
from module.mel_processing import spectrogram_torch
from feature_extractor import cnhubert as content_module

import pdb


vits_model_cache = None
def _load_model(config_file, device="cuda"):
    global vits_model_cache
    if vits_model_cache is not None:
        return vits_model_cache
    hps = utils.get_hparams_from_file(config_file)
    model_dir = hps.s2_ckpt_dir
    net_g = SynthesizerTrn(
        hps.data.filter_length // 2 + 1,
        hps.train.segment_size // hps.data.hop_length,
        n_speakers=hps.data.n_speakers,
        **hps.model).to(device)

    utils.load_checkpoint(utils.latest_checkpoint_path(model_dir, "backup/G_*.pth"), net_g,
                          None, True)
    net_g.eval()
    vits_model_cache = (hps, net_g)
    return hps, net_g


def get_spepc(hps, filename):
    audio, sampling_rate = utils.load_wav_to_torch(filename)
    if sampling_rate != hps.data.sampling_rate:
        raise ValueError("{} SR doesn't match target {} SR".format(
            sampling_rate, hps.data.sampling_rate))
    audio_norm = audio
    audio_norm = audio_norm.unsqueeze(0)
    spec = spectrogram_torch(audio_norm, hps.data.filter_length,
                             hps.data.sampling_rate, hps.data.hop_length, hps.data.win_length,
                             center=False)
    return spec

def text2phoneid(text, lang='zh'):
    phones = clean_text(text, lang)
    print(text, lang, phones)
    return cleaned_text_to_sequence(phones)


#-----------------------------------------------------------------------------#
#                              UTILITY FUNCTIONS                              #
#-----------------------------------------------------------------------------#


def check_argv():
    """Check the command line arguments."""
    parser = argparse.ArgumentParser(
        description="", add_help=False)
    parser.add_argument("name2text", type=str, help="the archive filename")
    parser.add_argument("config_file", type=str, help="the archive filename") 
    parser.add_argument("output_list", type=str, help="the archive filename") 
    parser.add_argument("--ckpt_path", type=str, default='logs/aishell3/s1/ckpt/epoch=50-step=765.ckpt', help="the archive filename")
    if len(sys.argv) < 4:
        parser.print_help()
        sys.exit(1)
    return parser.parse_args()


#-----------------------------------------------------------------------------#
#                                MAIN FUNCTION                                #
#-----------------------------------------------------------------------------#


def main():
    args = check_argv()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    hps, net_g = _load_model(args.config_file, device=device)
    content_model = content_module.get_model().to(device)

    prompt_text = "广州女大学生登山失联四天警方找到疑似女尸。"
    prompt_phones = text2phoneid(prompt_text, lang='zh')
    prompt_wav_path = "SSB00050001.wav"
    semantic_data = pd.read_csv('dump_aishell3/6-name2semantic.tsv', delimiter='\t')
    prompt_semantic = semantic_data[semantic_data['item_name'] == prompt_wav_path]['semantic_audio'].values[0]
    prompt_semantic = torch.LongTensor([int(idx) for idx in prompt_semantic.split(' ')])
    prompt = prompt_semantic.unsqueeze(0).to(device)
    refer_path = "/apdcephfs_qy3/share_301069248/users/yougenyuan/backup/models/GPT-SoVITS/output/slicer_opt/AISHELL-3/train/wav/SSB0005/" + prompt_wav_path
    refer = get_spepc(hps, refer_path).to(device)

    config = load_yaml_config("configs/s1.yaml")
    ckpt_path = args.ckpt_path #'logs/aishell3/s1/ckpt/epoch=99-step=1500.ckpt'

    hz = 50
    max_sec = config['data']['max_sec']

    # get models
    t2s_model = Text2SemanticLightningModule.load_from_checkpoint(checkpoint_path=ckpt_path, config=config, map_location=device)
    t2s_model.to(device)
    t2s_model.eval()

    total = sum([param.nelement() for param in t2s_model.parameters()])
    print("Number of parameter: %.2fM" % (total / 1e6))

    st = time.time()
    with open(args.output_list, 'w') as fout:
        for line in open(args.name2text, 'r', encoding="utf-8"):
            path, phoneme_data, bert, text = line.split("\n")[0].split("\t")
            try: 
                phones = text2phoneid(text)
                all_phoneme_ids = torch.LongTensor(prompt_phones+phones).to(device).unsqueeze(0)
                #print(all_phoneme_ids.shape)
                all_phoneme_len = torch.tensor([all_phoneme_ids.shape[-1]]).to(device)
                with torch.no_grad():
                    pred_semantic = t2s_model.model.infer(
                        all_phoneme_ids,
                        all_phoneme_len,
                        prompt,
                        top_k=config['inference']['top_k'],
                        early_stop_num=hz * max_sec)
                print("pred_semantic: ", pred_semantic.shape)
                phonemes = torch.LongTensor(prompt_phones+phones).unsqueeze(0).to(device)
                #phonemes = torch.LongTensor(phones).unsqueeze(0).to(device) 
                print("phonemes: ", phonemes)
                #pdb.set_trace()
                codes = pred_semantic.unsqueeze(0).transpose(0, 1) 
                audio = net_g.decode(codes, phonemes, refer).detach().cpu().numpy()[0, 0]
                print("audio: ", audio.shape) 
                new_dir = os.path.join(hps.data.data_dir, "test_s1_infer")
                os.makedirs(os.path.dirname(new_dir), exist_ok=True)
                new_path = os.path.join(new_dir, path)
                soundfile.write(new_path, audio, hps.data.sampling_rate)
                fout.write("\t".join([new_path, refer_path, path, phoneme_data, bert, text])+"\n")
            except Exception as e:
                print(f"tts failed! {line}")
                continue

if __name__ == "__main__":
    main()
