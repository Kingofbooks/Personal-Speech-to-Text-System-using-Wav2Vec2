import sounddevice as sd
from scipy.io.wavfile import write
from scipy.io import wavfile
import torch
import numpy as np
from transformers import Wav2Vec2Processor, Wav2Vec2ForCTC
from pyctcdecode import build_ctcdecoder

MODEL_PATH = "./my_asr_model_aryan"

processor = Wav2Vec2Processor.from_pretrained(MODEL_PATH)
model = Wav2Vec2ForCTC.from_pretrained(MODEL_PATH)
model.eval()

vocab_dict = processor.tokenizer.get_vocab()
vocab = sorted(vocab_dict, key=vocab_dict.get)
vocab[vocab.index("|")] = " "

decoder = build_ctcdecoder(vocab)

SR = 16000
DURATION = 5

def record():
    print("\n🎙 Speak...")
    audio = sd.rec(int(DURATION * SR), samplerate=SR, channels=1)
    sd.wait()
    write("input.wav", SR, audio)

def transcribe():

    sr, audio = wavfile.read("input.wav")

    if audio.ndim > 1:
        audio = audio[:,0]

    audio = audio.astype(np.float32)

    inputs = processor(audio, sampling_rate=16000, return_tensors="pt").input_values

    with torch.no_grad():
        logits = model(inputs).logits

    text = decoder.decode(logits.numpy()[0], beam_width=30)

    return text

while True:
    record()
    text = transcribe()
    print("\n🧠 You said:", text)
