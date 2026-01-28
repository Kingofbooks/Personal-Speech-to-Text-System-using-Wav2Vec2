# 🎙️ Personal Speech-to-Text System  
### Wav2Vec2 Fine-Tuning + Live Inference

This project implements a complete end-to-end **Automatic Speech Recognition (ASR)** pipeline using **Wav2Vec2**, including:

- Dataset creation  
- Model fine-tuning  
- Language model decoding  
- Live microphone inference  

The goal: build a speech-to-text system that understands *my own voice*, starting from a pretrained model and progressively adapting it using personal recordings.

---

## 🚀 Project Overview

The system consists of three major stages:

---

## 1️⃣ Dataset Creation

Two datasets were used.

### Common Voice (Baseline Training)

Public dataset downloaded from Kaggle:

> *([link will be added here](https://www.kaggle.com/datasets/aryansharma790/aryan-voice-dataset))*

Used to understand the standard ASR training pipeline and experiment with preprocessing.

---

### Personal Voice Dataset (Custom)

I recorded **500+ audio samples** of my own voice and manually corrected all transcriptions.

Each sample contains:

- `.wav` audio  
- Ground truth sentence  

#### Structure
my_voice_dataset/
├── clips/
│ ├── 001.wav
│ ├── 002.wav
│ └── ...
└── metadata.tsv

Each row in `metadata.tsv`:

path sentence
001.wav hello my name is aryan

This dataset was created using:

collect_voice.py
---

## 2️⃣ Model Fine-Tuning

### Base Model
facebook/wav2vec2-base-960h

Fine-tuned using:

- HuggingFace Transformers  
- PyTorch  
- CTC Loss  
- Custom DataCollator  

Training script:

finetune_personal.py

### Key Steps

- Audio normalization  
- Tokenization  
- Train / test split  
- Feature encoder freezing  
- Gradient accumulation  
- Multi-epoch training  

After training, the personalized model is saved locally:

my_asr_model_aryan/

---

## 3️⃣ Live Speech Recognition

Implemented in:

speech_live.py

### Pipeline

Microphone → Audio Buffer → Wav2Vec2 → Logits → Beam Search → Text

### Components

- `sounddevice` for microphone input  
- Wav2Vec2 for acoustic modeling  
- `pyctcdecode` + KenLM 4-gram language model  
- Beam search decoding  

Beam search significantly improves output quality compared to greedy decoding.

---

## 🧠 Full Pipeline

Voice Input
↓
Audio Recording
↓
Preprocessing (16kHz + normalization)
↓
Wav2Vec2 Acoustic Model
↓
CTC Logits
↓
Beam Search + KenLM Language Model
↓
Final Transcription


---

## 📁 Repository Structure

collect_voice.py → Record and label personal dataset
finetune_personal.py → Fine-tune Wav2Vec2 on custom voice
record_live.py → Live microphone inference
requirements.txt → Dependencies
README.md


---

## 🎥 Demo

Live speech recognition demo:

👉 *[Video link here](https://drive.google.com/file/d/1Uw6L1-_R81MFq_Ajrl8cKuTwxccVcMu_/view?usp=sharing)*

---

## ⚠️ Challenges Faced

This project involved real-world ML engineering problems:

### Dataset Creation

Manual recording and labeling of 500+ samples.

---

### HuggingFace Trainer Errors

Version conflicts between:

- PyTorch  
- NumPy  
- Transformers  

Required rebuilding environments multiple times.

---

### KenLM on Windows

KenLM does not compile easily on Windows.

Solution:

- Used WSL to build  
- Converted `.arpa → .bin`  
- Returned to Windows for microphone support  

---

### WSL Microphone Limitations

Linux could not access Windows microphone reliably.

Split workflow:

- Linux → language model  
- Windows → inference  

---

### Model Accuracy

Even after training:

- Accent variation  
- Limited dataset size  
- Background noise  

still impact quality.

This shows why production ASR systems require **thousands of hours of data**.

---

## 📉 Limitations

- Only ~500 personal samples  
- CPU inference only  
- English only  
- No punctuation restoration  
- No speaker adaptation  
- Limited language model vocabulary  

Accuracy is close — but imperfect.

---

## 🔮 Future Improvements

- Larger personal dataset (2000+ samples)  
- Streaming inference  
- Noise augmentation  
- Speaker adaptation  
- Punctuation model  
- Integration with LLM agent  

---

## 🛠️ Tech Stack

- Python  
- PyTorch  
- HuggingFace Transformers  
- Wav2Vec2  
- KenLM  
- PyCTCDecode  
- SoundDevice  
- SciPy  
- Kaggle  

---

## 📌 Key Learning Outcomes

- End-to-end ASR pipeline design  
- Dataset engineering  
- CTC decoding  
- Language model integration  
- Debugging cross-platform ML systems  
- Real-time inference  

---

## 👤 Author

**Aryan Sharma**  
Machine Learning Enthusiast
