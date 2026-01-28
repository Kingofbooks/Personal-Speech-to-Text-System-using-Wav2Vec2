import torch
import sounddevice as sd
import numpy as np
from transformers import Wav2Vec2Processor, Wav2Vec2ForCTC
from pyctcdecode import build_ctcdecoder

MODEL_PATH = "./my_asr_model_aryan"

processor = Wav2Vec2Processor.from_pretrained(MODEL_PATH)
model = Wav2Vec2ForCTC.from_pretrained(MODEL_PATH)
model.eval()

# Build decoder
vocab_dict = processor.tokenizer.get_vocab()
vocab = sorted(vocab_dict, key=vocab_dict.get)
vocab[vocab.index("|")] = " "

# decoder = build_ctcdecoder(
#     vocab,
#     kenlm_model_path="models/4-gram.bin",
#     alpha=0.6,
#     beta=1.0
# )
decoder = build_ctcdecoder(vocab)


SAMPLE_RATE = 16000
CHUNK_SECONDS = 3

def callback(indata, frames, time, status):
    if status:
        print(status)

    audio = indata[:,0]

    inputs = processor(audio, sampling_rate=16000, return_tensors="pt").input_values

    with torch.no_grad():
        logits = model(inputs).logits

    text = decoder.decode(logits.numpy()[0], beam_width=40)

    if len(text.strip()) > 0:
        print("\r🧠", text, end="")

print("🎙 LIVE — speak (Ctrl+C to stop)")

with sd.InputStream(
    samplerate=SAMPLE_RATE,
    channels=1,
    blocksize=int(SAMPLE_RATE * CHUNK_SECONDS),
    callback=callback,
):
    while True:
        pass
