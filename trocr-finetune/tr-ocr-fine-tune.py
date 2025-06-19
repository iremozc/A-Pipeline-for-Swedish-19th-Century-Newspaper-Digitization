# -*- coding: utf-8 -*-
"""Fine_tune_TrOCR_on_IAM_Handwriting_Database_using_Seq2SeqTrainer.ipynb



* TrOCR paper: https://arxiv.org/abs/2109.10282
* TrOCR documentation: https://huggingface.co/transformers/master/model_doc/trocr.html


Note that Patrick also wrote a very good [blog post](https://huggingface.co/blog/warm-starting-encoder-decoder) on warm-starting encoder-decoder models (which is what the TrOCR authors did). This blog post was very helpful for me to create this notebook.

We will fine-tune the model using the Seq2SeqTrainer, which is a subclass of the 🤗 Trainer that lets you compute generative metrics such as BLEU, ROUGE, etc by doing generation (i.e. calling the `generate` method) inside the evaluation loop.
"""

# Import required libraries
import torch
from torch.utils.data import Dataset
import pandas as pd
from PIL import Image
from sklearn.model_selection import train_test_split
from transformers import TrOCRProcessor, VisionEncoderDecoderModel
from transformers import Seq2SeqTrainer, Seq2SeqTrainingArguments
import evaluate
from datasets import load_dataset
from transformers import default_data_collator
import transformers
import torch
from torch.utils.data import Dataset
import pandas as pd
from PIL import Image
from sklearn.model_selection import train_test_split
from transformers import TrOCRProcessor, VisionEncoderDecoderModel
from transformers import Seq2SeqTrainer, Seq2SeqTrainingArguments
from transformers import default_data_collator
import transformers
from datasets import load_dataset
import numpy as np
from jiwer import wer, cer




ds = load_dataset("Riksarkivet/swedish_fraktur")

df = pd.DataFrame(ds['train'])
df.head()

df.rename(columns={0: "image", 1: "text"}, inplace=True)

df.head()

"""We split up the data into training + testing, using sklearn's `train_test_split` function."""

train_df, test_df = train_test_split(df, test_size=0.2)
# we reset the indices to start from zero
train_df.reset_index(drop=True, inplace=True)
test_df.reset_index(drop=True, inplace=True)
train_df.head()

# Print lengths before filtering empty strings
print("BEFORE FILTERING:")
print("Length of training dataset after split:", len(train_df))
print("Length of test dataset after split:", len(test_df))

# For train dataset
empty_train = train_df[train_df['text'].str.strip() == '']
print("Empty strings in train_df:", empty_train.shape[0])

# For test dataset
empty_test = test_df[test_df['text'].str.strip() == '']
print("Empty strings in test_df:", empty_test.shape[0])

# Filter out rows where the "text" is empty in the training set
train_df = train_df[train_df['text'].str.strip() != ''].reset_index(drop=True)

# Filter out rows where the "text" is empty in the test set
test_df = test_df[test_df['text'].str.strip() != ''].reset_index(drop=True)

# Print lengths after filtering empty strings
print("\nAFTER FILTERING:")
print("Length of training dataset after filtering:", len(train_df))
print("Length of test dataset after filtering:", len(test_df))

for k,v in train_df.items():
  print(k, v.shape)

"""Each element of the dataset should return 2 things:
* `pixel_values`, which serve as input to the model.
* `labels`, which are the `input_ids` of the corresponding text in the image.

We use `TrOCRProcessor` to prepare the data for the model. `TrOCRProcessor` is actually just a wrapper around a `ViTFeatureExtractor` (which can be used to resize + normalize images) and a `RobertaTokenizer` (which can be used to encode and decode text into/from `input_ids`).
"""

from torch.utils.data import Dataset
from PIL import Image

class Swedish_fraktur(Dataset):
    def __init__(self, df, processor, model, max_target_length=128):
        self.df = df
        self.processor = processor
        self.model = model
        self.max_target_length = max_target_length

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        text = self.df['text'].iloc[idx]
        image_data = self.df['image'].iloc[idx]

        # 1) load image
        if isinstance(image_data, str):
            image = Image.open(image_data).convert("RGB")
        elif isinstance(image_data, Image.Image):
            image = image_data.convert("RGB")
        else:
            raise TypeError("image must be file path or PIL Image")

        # 2) prepare pixel_values (all the same shape!)
        enc = self.processor(images=image, return_tensors="pt")
        pixel_values = enc.pixel_values.squeeze(0)

        # 3) tokenize & mask pads for labels
        lbl = self.processor.tokenizer(
            text,
            padding="max_length",
            max_length=self.max_target_length,
            return_tensors="pt"
        ).input_ids.squeeze(0)
        lbl[lbl == self.processor.tokenizer.pad_token_id] = -100

        # 4) SHIFT to get decoder_input_ids
        dec_in = self.model.prepare_decoder_input_ids_from_labels(
            lbl.unsqueeze(0)
        ).squeeze(0)

        return {
            "pixel_values": pixel_values,
            "labels": lbl,
            "decoder_input_ids": dec_in,
        }


from transformers import TrOCRProcessor, VisionEncoderDecoderModel

processor = TrOCRProcessor.from_pretrained("microsoft/trocr-base-printed")
model     = VisionEncoderDecoderModel.from_pretrained("microsoft/trocr-base-printed")



# labels


# set special tokens used for creating the decoder_input_ids from the labels
model.config.decoder_start_token_id = processor.tokenizer.cls_token_id
model.config.pad_token_id = processor.tokenizer.pad_token_id
# make sure vocab size is set correctly
model.config.vocab_size = model.config.decoder.vocab_size

# set beam search parameters
model.config.eos_token_id = processor.tokenizer.sep_token_id
model.config.max_length = 64
model.config.early_stopping = True
model.config.no_repeat_ngram_size = 3
model.config.length_penalty = 2.0
model.config.num_beams = 4

train_dataset = Swedish_fraktur(train_df, processor, model)
eval_dataset  = Swedish_fraktur(test_df,  processor, model)
print("Number of training examples:", len(train_dataset))
print("Number of validation examples:", len(eval_dataset))
"""Next, we can define some training hyperparameters by instantiating the `training_args`. Note that there are many more parameters, all of which can be found in the [documentation](https://huggingface.co/transformers/main_classes/trainer.html#seq2seqtrainingarguments). You can for example decide what the batch size is for training/evaluation, whether to use mixed precision training (lower memory), the frequency at which you want to save the model, etc."""
training_args = Seq2SeqTrainingArguments(
    output_dir="./trocr-output",
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    predict_with_generate=True,
    logging_steps=2,
    fp16=True,
    num_train_epochs=10,  
    do_eval=True,  
    learning_rate=5e-5,           # Learning rate
    optim="adamw_torch",          # Optimizer type
    weight_decay=0.01,            # Weight decay for regularization
    warmup_ratio=0.1,             # Portion of steps for warmup
    label_smoothing_factor=0.1,   # Label smoothing for better generalization
    max_grad_norm=1.0,      # Enable evaluation
)


class EvaluateEveryEpochCallback(transformers.TrainerCallback):
    def __init__(self, trainer, processor):
        self.trainer = trainer
        self.processor = processor

    def on_epoch_end(self, args, state, control, **kwargs):
        print(f"\n=== EPOCH {state.epoch} COMPLETED ===")
        # Run evaluation on the eval dataset
        output = self.trainer.predict(self.trainer.eval_dataset, metric_key_prefix="eval")
        preds = output.predictions
        labels = output.label_ids
        
        # Decode predictions and labels
        pred_str = self.processor.batch_decode(preds, skip_special_tokens=True)
        labels[labels == -100] = self.processor.tokenizer.pad_token_id
        label_str = self.processor.batch_decode(labels, skip_special_tokens=True)
        
        # Compute metrics using jiwer (average over examples)
        cer_list = [cer(ref, pred) for ref, pred in zip(label_str, pred_str)]
        wer_list = [wer(ref, pred) for ref, pred in zip(label_str, pred_str)]
        avg_cer = np.mean(cer_list)
        avg_wer = np.mean(wer_list)
        
        # Print evaluation examples and metrics
        print("Evaluation Examples:")
        for i in range(min(3, len(pred_str))):
            print(f"Reference: {label_str[i]}")
            print(f"Prediction: {pred_str[i]}")
            print("-" * 50)
        print(f"Average CER: {avg_cer:.4f}")
        print(f"Average WER: {avg_wer:.4f}\n")
        return control

from transformers import DataCollatorForSeq2Seq


trainer = Seq2SeqTrainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
    data_collator=default_data_collator,
    tokenizer=processor.tokenizer,
)

# Add our custom evaluation callback
trainer.add_callback(EvaluateEveryEpochCallback(trainer, processor))



print("\n=== STARTING TRAINING ===")
trainer.train()

print("\n=== FINAL EVALUATION ===")
final_metrics = trainer.evaluate()
print(f"Final Evaluation Metrics:\nCER: {final_metrics['eval_loss']:.4f} (loss reported, see callback output for jiwer-based metrics)")

# Save model after training
trainer.save_model("./trocr-finetuned-swedish-fraktur")

"""
## Inference

After training, you can load the model with:
    VisionEncoderDecoderModel.from_pretrained("./trocr-finetuned-swedish-fraktur")
and perform inference on new images.
"""

