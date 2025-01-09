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


### Define pipeline
from transformers import pipeline
checkpoint = "hoganpham/my_first_audio_cls"
classifier = pipeline("audio-classification", model=checkpoint)
results = classifier(audio_file)
print(results)


### Inference
import numpy as np
pred_cls_idx = np.argmax([sample['score'] for sample in results])
print(pred_cls_idx)
pred_label = id2label[str(pred_cls_idx)]
print("Predicted label : ", pred_label)