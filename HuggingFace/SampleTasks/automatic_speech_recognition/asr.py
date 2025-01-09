import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = '3'

### ===============================================================================
### Automatic Speech Recognition
### Model: Wav2Vec2
### Dataset: MInDS-14
### ===============================================================================


###
### Load dataset
###

from datasets import load_dataset, Audio
minds = load_dataset("PolyAI/minds14", name="en-US", split="train[:100]", trust_remote_code=True)

### Train-Test split
minds = minds.train_test_split(test_size=0.2)
# print(minds)
# print(minds["train"][0])

### Focus on 2 fields: "audio" and "transcription"

###
### Pre-process data
###

from transformers import AutoProcessor

checkpoint = "facebook/wav2vec2-base"
processor = AutoProcessor.from_pretrained(checkpoint)

### Resample sample rate from 8000 to 16000
minds = minds.cast_column("audio", Audio(sampling_rate=16000))

### Uppercase transcript
def uppercase(example):
    return {"transcription": example["transcription"].upper()}
minds = minds.map(uppercase)

### Pre-processing function
def prepare_dataset(batch):
    audio = batch["audio"]
    batch = processor(audio["array"], sampling_rate=audio["sampling_rate"], text=batch["transcription"])
    batch["input_length"] = len(batch["input_values"][0])
    return batch

### Apply pre-processing function over entire dataset
encoded_minds = minds.map(prepare_dataset, remove_columns=minds.column_names["train"], num_proc=4)

### Dynamicaly pad text and labels to the length of the longest element in its batch (not entire dataset)
import torch
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Union

@dataclass
class DataCollatorCTCWithPadding:
    processor: AutoProcessor
    padding: Union[bool, str] = "longest"
    
    def __call__(self,
                 features: List[Dict[str, Union[List[int], torch.Tensor]]]
    ) -> Dict[str, torch.Tensor]:
        
        # split inputs and labels since they have to be of different lengths and need
        # different padding methods
        input_features = [{"input_values": feature["input_values"][0]} for feature in features]
        label_features = [{"input_ids": feature["labels"]} for feature in features]

        batch = self.processor.pad(input_features, padding=self.padding, return_tensors="pt")
        
        labels_batch = self.processor.pad(labels=label_features, padding=self.padding, return_tensors="pt")
        
        # replace padding with -100 to ignore loss correctly
        labels = labels_batch["input_ids"].masked_fill(labels_batch.attention_mask.ne(1), -100)
        
        batch["labels"] = labels
        
        return batch
    
### Define data collator
data_collator = DataCollatorCTCWithPadding(processor=processor, padding="longest")

###
### Define metrics
###
import evaluate
wer = evaluate.load("wer")

### Define compute metrics function
def compute_metrics(pred):
    pred_logits = pred.predictions
    pred_ids = np.argmax(pred_logits, axis=-1)
    
    pred.label_ids[pred.label_ids == -100] = processor.tokenizer.pad_token_id
    
    pred_str = processor.batch_decode(pred_ids)
    label_str = processor.batch_decode(pred.label_ids, group_tokens=False)
    
    wer = wer.compute(predictions=pred_str, referecens=label_str)

    return {"wer": wer}


###
### Training
###

### Define model
from transformers import AutoModelForCTC, TrainingArguments, Trainer

model = AutoModelForCTC.from_pretrained(
    checkpoint,
    ctc_loss_reduction="mean",
    pad_token_id=processor.tokenizer.pad_token_id
)


### Define training arguments
training_args = TrainingArguments(
    output_dir="my_first_asr_model",
    per_device_train_batch_size=8,
    gradient_accumulation_steps=2,
    learning_rate=1e-5,
    warmup_steps=500,
    max_steps=2000,
    gradient_checkpointing=True,
    fp16=True,
    group_by_length=True,
    eval_strategy="steps",
    per_device_eval_batch_size=8,
    save_steps=1000,
    eval_steps=1000,
    logging_steps=25,
    load_best_model_at_end=True,
    metric_for_best_model="wer",
    greater_is_better=False,
    push_to_hub=True,
)


### Define trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=encoded_minds["train"],
    eval_dataset=encoded_minds["test"],
    processing_class=processor,
    data_collator=data_collator,
    compute_metrics=compute_metrics,
)


### Run training
trainer.train()
trainer.push_to_hub()