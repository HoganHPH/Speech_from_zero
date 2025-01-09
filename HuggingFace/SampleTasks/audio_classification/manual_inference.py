import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = '3'


### Load test data
from datasets import load_dataset, Audio

dataset = load_dataset("PolyAI/minds14", name="en-US", split="train", trust_remote_code=True)
dataset = dataset.cast_column("audio", Audio(sampling_rate=16000))
sampling_rate = dataset.features["audio"].sampling_rate
audio_file = dataset[0]["audio"]["path"]

labels = dataset.features["intent_class"].names
label2id, id2label = dict(), dict()
for i, label in enumerate(labels):
    label2id[label] = str(i)
    id2label[str(i)] = label
audio_label = id2label[str(dataset[0]["intent_class"])]
print("True label : ", audio_label)


### Define FeatureExtractor
from transformers import AutoFeatureExtractor

checkpoint = "hoganpham/my_first_audio_cls"

feature_extractor = AutoFeatureExtractor.from_pretrained(checkpoint)
inputs = feature_extractor(dataset[0]["audio"]["array"], sampling_rate=sampling_rate, return_tensors="pt")


### Define model
from transformers import AutoModelForAudioClassification

model = AutoModelForAudioClassification.from_pretrained(checkpoint)


### Inference
import torch
with torch.no_grad():
    logits = model(**inputs).logits
    
predicted_class_ids = torch.argmax(logits).item()
predicted_label = model.config.id2label[predicted_class_ids]
print(predicted_label)