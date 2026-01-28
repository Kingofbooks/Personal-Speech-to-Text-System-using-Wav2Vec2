import os
import torch
import pandas as pd
import numpy as np
from scipy.io import wavfile
from datasets import Dataset
from transformers import (
    Wav2Vec2Processor,
    Wav2Vec2ForCTC,
    TrainingArguments,
    Trainer
)
from dataclasses import dataclass

BASE_MODEL = "facebook/wav2vec2-base-960h"

DATASET_PATH = r"E:\JAVERS\SOUND\SPEECH_TO_TEXT\my_voice_dataset\metadata.tsv"
AUDIO_DIR = r"E:\JAVERS\SOUND\SPEECH_TO_TEXT\my_voice_dataset\clips"

TARGET_SR = 16000

device = "cuda" if torch.cuda.is_available() else "cpu"

processor = Wav2Vec2Processor.from_pretrained(BASE_MODEL)
model = Wav2Vec2ForCTC.from_pretrained(BASE_MODEL).to(device)

df = pd.read_csv(DATASET_PATH, sep="\t")
dataset = Dataset.from_pandas(df)

# ======================
# PREPARE AUDIO
# ======================

def prepare(batch):
    sr, speech = wavfile.read(os.path.join(AUDIO_DIR, batch["path"]))

    if speech.ndim > 1:
        speech = speech[:, 0]

    speech = speech.astype(np.float32) / 32768.0

    batch["input_values"] = processor(
        speech,
        sampling_rate=TARGET_SR
    ).input_values[0]

    batch["labels"] = processor(text=batch["sentence"]).input_ids

    return batch

dataset = dataset.map(
    prepare,
    remove_columns=dataset.column_names
)

dataset = dataset.train_test_split(test_size=0.1)

# ======================
# COLLATOR
# ======================

@dataclass
class DataCollatorCTC:
    processor: Wav2Vec2Processor

    def __call__(self, features):

        input_features = [{"input_values": f["input_values"]} for f in features]
        label_features = [{"input_ids": f["labels"]} for f in features]

        batch = self.processor.feature_extractor.pad(
            input_features,
            return_tensors="pt"
        )

        labels_batch = self.processor.tokenizer.pad(
            label_features,
            return_tensors="pt"
        )

        labels = labels_batch["input_ids"].masked_fill(
            labels_batch.attention_mask.ne(1),
            -100
        )

        batch["labels"] = labels

        return batch

collator = DataCollatorCTC(processor)

# ======================
# TRAINING
# ======================

# training_args = TrainingArguments(
#     output_dir="./my_asr_model",
#     per_device_train_batch_size=2,
#     gradient_accumulation_steps=2,
#     num_train_epochs=15,
#     learning_rate=1e-4,
#     fp16=torch.cuda.is_available(),
#     logging_steps=10,
#     save_steps=200,
#     save_total_limit=2,
#     report_to="none"
# )

training_args = TrainingArguments(
    output_dir="./my_asr_model",
    per_device_train_batch_size=2,
    gradient_accumulation_steps=2,
    num_train_epochs=3,
    learning_rate=1e-4,
    logging_steps=10,
    save_steps=500,
    save_total_limit=2,
    report_to="none"
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=dataset["train"],
    eval_dataset=dataset["test"],
    data_collator=collator,
)

model.freeze_feature_encoder()

trainer.train()

model.save_pretrained("./my_asr_model")
processor.save_pretrained("./my_asr_model")

print("\n✅ PERSONAL MODEL SAVED TO my_asr_model/")
