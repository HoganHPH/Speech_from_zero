import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = '3'
import torch


### Load test data
from datasets import load_dataset, Audio

dataset = load_dataset("PolyAI/minds14", "en-US", split="train")
dataset = dataset.cast_column("audio", Audio(sampling_rate=16000))
sampling_rate = dataset.features["audio"].sampling_rate
audio_file = dataset[0]["audio"]["path"]


### Define processor and model

from transformers import AutoProcessor

checkpoint = "hoganpham/my_first_asr_model"
processor = AutoProcessor.from_pretrained(checkpoint)
inputs = processor(dataset[0]["audio"]["array"], sampling_rate=sampling_rate, return_tensors="pt")

from transformers import AutoModelForCTC

model = AutoModelForCTC.from_pretrained(checkpoint)
with torch.no_grad():
    logits = model(**inputs).logits


### Inference

predicted_ids = torch.argmax(logits, dim=-1)
transcription = processor.batch_decode(predicted_ids)
print(transcription)