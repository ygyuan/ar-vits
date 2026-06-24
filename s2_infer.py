import sys
import argparse
import librosa
import soundfile
import torch
import os
import utils
from module.models import SynthesizerTrn
from module.mel_processing import spectrogram_torch
from feature_extractor import cnhubert as content_module
from text import cleaned_text_to_sequence

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


def decode_to_file(codes, phonemes, save_path, refer_path, transform='valle'):
    device = codes.device
    hps, net_g = _load_model(device=device)
    if transform=='valle':
        codes = codes.transpose(0, 1).unsqueeze(1)
    else:
        codes = codes.transpose(0, 1)
    refer = get_spepc(hps, refer_path).to(device)
    audio = net_g.decode(codes,phonemes, refer).detach().cpu().numpy()[0, 0]
    soundfile.write(save_path, audio, hps.data.sampling_rate)


def encode_from_file(path, device='cpu'):
    hps, net_g = _load_model(device=device)
    content_model = content_module.get_model().to(device)
    wav16k, sr = librosa.load(path, sr=16000)
    with torch.no_grad():
        wav16k = torch.from_numpy(wav16k).to(device)
        ssl_content = content_module.get_content(content_model, wav_16k_tensor=wav16k)
        codes = net_g.extract_latent(ssl_content)
    return codes.cpu()

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
    if len(sys.argv) != 4:
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
    os.makedirs(os.path.join(hps.data.data_dir, "test_s2_infer"), exist_ok=True)

    with open(args.output_list, 'w') as fout:
        for line in open(args.name2text, 'r', encoding="utf-8"):
            path, phoneme_data, bert, text = line.split("\n")[0].split("\t")
            try:
                phoneme = phoneme_data.split(' ')
                phoneme_id = cleaned_text_to_sequence(phoneme)
                phonemes = torch.LongTensor(phoneme_id).unsqueeze(0).to(device)
            except Exception as e:
                print(f"read phoneme data failed! {path} {phoneme_data}")
                continue
            print(phonemes)

            ori_path = os.path.join(hps.data.data_dir, "5-wav32k", path)
            wav16k, sr = librosa.load(ori_path, sr=16000)
            with torch.no_grad():
                wav16k = torch.from_numpy(wav16k).to(device)
                ssl_content = content_module.get_content(content_model, wav_16k_tensor=wav16k)
                codes = net_g.extract_latent(ssl_content)
                print(codes.shape)
            
            new_path = os.path.join(hps.data.data_dir, "test_infer", path)
            refer_path = "/apdcephfs_qy3/share_301069248/users/yougenyuan/backup/models/GPT-SoVITS/output/slicer_opt/AISHELL-3/train/wav/SSB0005/SSB00050001.wav"

            codes = codes.transpose(0, 1)
            refer = get_spepc(hps, refer_path).to(device)
            audio = net_g.decode(codes, phonemes, refer).detach().cpu().numpy()[0, 0]
            soundfile.write(new_path, audio, hps.data.sampling_rate)

            #decode_to_file(codes, phonemes, new_path, refer_path, transform="raw")
            #str_phonemes = " ".join([str(i) for i in phonemes])
            #fout.write("\t".join([new_path, refer_path, str_phonemes])+"\n")
            fout.write("\t".join([new_path, refer_path, path, phoneme_data, bert, text])+"\n")


if __name__ == '__main__':
    main()  
