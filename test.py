from bark.api import generate_audio
from transformers import BertTokenizer
from bark.generation import SAMPLE_RATE, preload_models, codec_decode, generate_coarse, generate_fine, generate_text_semantic
from scipy.io.wavfile import write as write_wav

# Enter your prompt and speaker here
text_prompt = "I had a meeting with U S government yesterday as well."
voice_name = "en_miles" # use your custom voice name here if you have one

# load the tokenizer
tokenizer = BertTokenizer.from_pretrained("bert-base-multilingual-cased")

# download and load all models
preload_models()

# simple generation
audio_array = generate_audio(text_prompt, history_prompt=voice_name, text_temp=0.7, waveform_temp=0.7)

# save audio
write_wav("/data/bark_output.wav", SAMPLE_RATE, audio_array)
