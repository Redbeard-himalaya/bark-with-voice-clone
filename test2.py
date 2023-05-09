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

# generation with more control
x_semantic = generate_text_semantic(
    text_prompt,
    history_prompt=voice_name,
    temp=0.7,
    top_k=50,
    top_p=0.95,
)

x_coarse_gen = generate_coarse(
    x_semantic,
    history_prompt=voice_name,
    temp=0.7,
    top_k=50,
    top_p=0.95,
)
x_fine_gen = generate_fine(
    x_coarse_gen,
    history_prompt=voice_name,
    temp=0.5,
)
audio_array = codec_decode(x_fine_gen)

# save audio
write_wav("/data/bark_output.wav", SAMPLE_RATE, audio_array)
