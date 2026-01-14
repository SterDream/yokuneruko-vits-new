from F0_MB_iSTFT_VITS.models import SynthesizerTrn
import F0_MB_iSTFT_VITS.commons as commons
import F0_MB_iSTFT_VITS.utils as utils

from F0_MB_iSTFT_VITS.text.phonemize.IPA_tokenizer import Tokenizer

import os
import torch
import soundfile as sf


class Get_text:
    def __init__(self):
        self.tok = Tokenizer()

    def get_text(self, text, hps):
        x_tst, x_tst_lengths, lang = self.tok.to_tensor(text)
        if hps.data.add_blank:
            x_tst = commons.intersperse(x_tst, 0)
        return x_tst, x_tst_lengths, lang


class TTS:
    def __init__(self, config, model_path):
        self.tok = Tokenizer()
        self.get = Get_text()

        # check device
        if  torch.cuda.is_available() is True:
            print("Enter the device number to use.")
            key = input("GPU:0, CPU:1 ===> ")
            if key == "0":
                device="cuda:0"
            elif key=="1":
                device="cpu"
            print(f"Device : {device}")
        else:
            print(f"CUDA is not available. Device : cpu")
            device = "cpu"

        self.hps = utils.get_hparams_from_file(config)

        # load checkpoint
        self.net_g = SynthesizerTrn(
            len(self.tok.vocab),
            self.hps.data.filter_length // 2 + 1,
            self.hps.train.segment_size // self.hps.data.hop_length,
            **self.hps.model
        ).cuda()
        
        _ = self.net_g.eval()
        _ = utils.load_checkpoint(model_path, self.net_g, None)


    def inference(self, text, save_file):
        # parameter settings
        noise_scale     = torch.tensor(0.66)    # adjust z_p noise
        noise_scale_w   = torch.tensor(0.8)    # adjust SDP noise
        length_scale    = torch.tensor(1.0)     # adjust sound length scale (talk speed)

        if save_file is True:
            n_save = 0
            save_dir = os.path.join("./infer_logs/")
            os.makedirs(save_dir, exist_ok=True)

        with torch.inference_mode():
            x_tst, x_tst_lengths, lang = self.get(text)

            audio = self.net_g.infer(
                x_tst, 
                x_tst_lengths, 
                lang_id=lang,
                noise_scale=noise_scale, 
                noise_scale_w=noise_scale_w, 
                length_scale=length_scale
            )[0][0,0].data.cpu().float().numpy()

            # save audio
            if save_file is True:
                n_save += 1
                data = audio

                try:
                    save_path = os.path.join(save_dir, str(n_save).zfill(3)+f"_{text}.wav")
                    sf.write(
                        file=save_path,
                        data=data,
                        samplerate=self.hps.data.sampling_rate,
                        format="WAV"
                    )
                except:
                    save_path = os.path.join(save_dir, str(n_save).zfill(3)+f"_{text[:10]}〜.wav")
                    sf.write(
                        file=save_path,
                        data=data,
                        samplerate=self.hps.data.sampling_rate,
                        format="WAV"
                    )


if __name__ == "__main__":
    model_path = ""
    config_path = ""
    text = "今日は天気が良いから散歩しよう。" # The weather's nice today, so let's go for a walk. 

    tts = TTS(config_path, model_path)
    tts.inference(text, save_file=True)
