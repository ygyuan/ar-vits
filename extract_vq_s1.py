import math
import multiprocessing
import os
from random import shuffle
import torch.multiprocessing as mp

import torch
from glob import glob
from tqdm import tqdm

import utils
import logging

from data_conf import data_root
from module.models import SynthesizerTrn

logging.getLogger("numba").setLevel(logging.WARNING)
import librosa

import numpy as np
import sys
import argparse
import pdb

def process_one(f, file_path, phoneme_dict, vq_model, device):
    try:
        ssl_path = phoneme_dict[file_path][0].replace("5-wav32k", "4-cnhubert")+".pt"
        #print(ssl_path, os.path.exists(ssl_path))
        if os.path.exists(ssl_path):
            ssl_content = torch.load(ssl_path).float().to(device)
            #print("ssl_content: ", ssl_content)
            codes = vq_model.extract_latent(ssl_content)
            #print("codes: ", codes.size())
            semantic = " ".join([str(i) for i in codes[0, 0, :].tolist()])
            #print("semantic: ", semantic)
            f.write(f"{file_path}\t{semantic}\n")
            f.flush()
        else:
            print("check: ", ssl_path)
    
    except:
        print("skip", file_path)

def process_batch(filenames, phoneme_dict, configs):
    print("Loading models ...")
    process_idx = mp.current_process()._identity
    rank = process_idx[0] if len(process_idx) > 0 else 0
    gpu_id = rank % torch.cuda.device_count()
    device = torch.device(f"cuda:{gpu_id}")
    print(device)
    hps = utils.get_hparams_from_file(configs)
    vq_model = SynthesizerTrn(
        hps.data.filter_length // 2 + 1,
        hps.train.segment_size // hps.data.hop_length,
        n_speakers=hps.data.n_speakers,
        **hps.model).to(device)
    vq_model.eval()
    utils.load_checkpoint(utils.latest_checkpoint_path(hps.s2_ckpt_dir, "backup/G_*.pth"), vq_model,
                                                       None, True)

    print("Loaded .")
    with torch.no_grad():
        with open(f"TEMP/semantic_{process_idx[0]}.tsv", "w") as f:
            for filename in tqdm(filenames):
                process_one(f, filename, phoneme_dict, vq_model, device)

#-----------------------------------------------------------------------------#
#                              UTILITY FUNCTIONS                              #
#-----------------------------------------------------------------------------#


def check_argv():
    """Check the command line arguments."""
    parser = argparse.ArgumentParser(
        description="", add_help=False)
    parser.add_argument("phoneme_path", type=str, help="the archive filename")
    parser.add_argument("configs", type=str, help="the archive filename") 
    parser.add_argument("semantic_tsv", type=str, help="the archive filename")
    if len(sys.argv) != 4:
        parser.print_help()
        sys.exit(1)
    return parser.parse_args()

#-----------------------------------------------------------------------------#
#                                MAIN FUNCTION                                #
#-----------------------------------------------------------------------------#


def main():
    args = check_argv()
    print("Reading phoneme dict %s" % args.phoneme_path)
    phoneme_dict = np.load(args.phoneme_path, allow_pickle=True).item()
    filenames = []
    for key, value in phoneme_dict.items():
        filenames.append(key)
    print("filenames has the size of %d" % len(filenames))

    multiprocessing.set_start_method("spawn", force=True)

    num_processes = 48  #24 #8
    chunk_size = int(math.ceil(len(filenames) / num_processes))
    chunks = [
        filenames[i : i + chunk_size] for i in range(0, len(filenames), chunk_size)
    ]
    print([len(c) for c in chunks])
    processes = [
        multiprocessing.Process(target=process_batch, args=(chunk, phoneme_dict, args.configs)) for chunk in chunks
    ]
    for p in processes:
        p.start()

    for p in processes:
        p.join()
    with open(args.semantic_tsv, "w") as f:
        f.write("item_name\tsemantic_audio\n")
        for i in range(num_processes):
            with open(f"TEMP/semantic_{i+1}.tsv", "r") as f2:
                f.write(f2.read())
            os.remove(f"TEMP/semantic_{i+1}.tsv")

if __name__ == "__main__":
    main()
