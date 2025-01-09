import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = '3'


### Load test data
from datasets import load_dataset, Audio

dataset = load_dataset("PolyAI/minds14", "en-US", split="train")
dataset = dataset.cast_column("audio", Audio(sampling_rate=16000))
sampling_rate = dataset.features["audio"].sampling_rate
audio_file = dataset[0]["audio"]["path"]


### Define pipeline
from transformers import pipeline

checkpoint = "hoganpham/my_first_asr_model"
transcriber = pipeline("automatic-speech-recognition", model=checkpoint)

### Inference
result = transcriber(audio_file)
print(result)