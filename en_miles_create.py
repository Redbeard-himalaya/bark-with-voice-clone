from bark.generation import load_codec_model, generate_text_semantic
from encodec.utils import convert_audio

import torchaudio
import torch

model = load_codec_model(use_gpu=True)

# Load and pre-process the audio waveform
audio_filepath = '/data/6b1a6c70-e896-4e88-88e8-0e66cf991700/seg/0.wav' # the audio you want to clone (will get truncated so 5-10 seconds is probably fine, existing samples that I checked are around 7 seconds)
text = "昨天和美国的政府部门也有见面."
device = 'cuda' # or 'cpu'
wav, sr = torchaudio.load(audio_filepath)
wav = convert_audio(wav, sr, model.sample_rate, model.channels)
wav = wav.unsqueeze(0).to(device)

# Extract discrete codes from EnCodec
with torch.no_grad():
    encoded_frames = model.encode(wav)
codes = torch.cat([encoded[0] for encoded in encoded_frames], dim=-1).squeeze()  # [n_q, T]

# get seconds of audio
seconds = wav.shape[-1] / model.sample_rate
# generate semantic tokens
semantic_tokens = generate_text_semantic(text, max_gen_duration_s=seconds, top_k=50, top_p=.95, temp=0.7) # not 100% sure on this part

# move codes to cpu
codes = codes.cpu().numpy()

import numpy as np
voice_name = 'en_miles' # whatever you want the name of the voice to be
output_path = 'bark/assets/prompts/' + voice_name + '.npz'
np.savez(output_path, fine_prompt=codes, coarse_prompt=codes[:2, :], semantic_prompt=semantic_tokens)
