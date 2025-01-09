import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import transformers
print(transformers.__version__)

### =================================================================================
### Audio Classification
### Model: Wav2Vec2
### Dataset: MInDS-14
### =================================================================================

###
### Load dataset
###
from datasets import load_dataset, Audio
minds = load_dataset("PolyAI/minds14", name="en-US", split="train", trust_remote_code=True)

###
### Pre-process dataset
###

### Train-Test split
minds = minds.train_test_split(test_size=0.2)
# print(minds)

### Keep only columns "audio" and "intent_class", remove others
minds = minds.remove_columns(["path", "transcription", "english_transcription", "lang_id"])
# print(minds["train"][0])

### Mapping label_name and label_id
labels = minds["train"].features["intent_class"].names
label2id, id2label = dict(), dict()
for i, label in enumerate(labels):
    label2id[label] = str(i)
    id2label[str(i)] = label
# print(id2label[str(2)])

### Load Wav2Vec2 feature extractor to process the audio signal
from transformers import AutoFeatureExtractor

checkpoint = "facebook/wav2vec2-base"
feature_extractor = AutoFeatureExtractor.from_pretrained(checkpoint)

### Resample sample rate from 8000Hz to 16000Hz
minds = minds.cast_column("audio", Audio(sampling_rate=16_000))
# print(minds["train"][0])

### Pre-processing function
def preprocess_function(examples):
    audio_arrays = [x["array"] for x in examples["audio"]]
    inputs = feature_extractor(
        audio_arrays, 
        sampling_rate=feature_extractor.sampling_rate, 
        max_length=16000, 
        truncation=True
    )
    return inputs
    
### Apply pre-processing function over the entire dataset
encoded_minds = minds.map(preprocess_function, remove_columns="audio", batched=True)
encoded_minds = encoded_minds.rename_column("intent_class", "label")


###
### Define metrics
###
import evaluate

accuracy = evaluate.load("accuracy")

### compute accuracy function
import numpy as np

def compute_metrics(eval_pred):
    predictions = np.argmax(eval_pred.predictions, axis=1)
    return accuracy.compute(predictions=predictions, references=eval_pred.label_ids)

###
### Training
###

### Define model
from transformers import AutoModelForAudioClassification, TrainingArguments, Trainer

num_labels = len(id2label)
model = AutoModelForAudioClassification.from_pretrained(
    checkpoint,
    num_labels=num_labels,
    label2id=label2id,
    id2label=id2label
)

### Define training arguments
training_args = TrainingArguments(
    output_dir="my_first_audio_cls",
    eval_strategy="epoch",
    save_strategy="epoch",
    learning_rate=3e-5,
    per_device_train_batch_size=32,
    gradient_accumulation_steps=4,
    per_device_eval_batch_size=32,
    num_train_epochs=10,
    warmup_ratio=0.1,
    logging_steps=10,
    load_best_model_at_end=True,
    metric_for_best_model="accuracy",
    push_to_hub=True,
    gradient_checkpointing=True
)

### Define trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=encoded_minds["train"],
    eval_dataset=encoded_minds["test"],
    processing_class=feature_extractor,
    compute_metrics=compute_metrics,
)

### Run training
trainer.train()
trainer.push_to_hub()
