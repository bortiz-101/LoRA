""" First we will import all necessary libraries
"""
from datasets import load_dataset
from typing import cast
import torch
from transformers import AutoProcessor, Gemma4ForCausalLM, AutoTokenizer
from peft import LoraConfig, get_peft_model, TaskType, PeftModel
from trl.trainer.sft_trainer import SFTTrainer
from trl.trainer.sft_config import SFTConfig
from sklearn.metrics import accuracy_score, f1_score, classification_report, confusion_matrix
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Check if a GPU is available and set the device accordingly
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Next we will load the pre-trained model and tokenizer
model = Gemma4ForCausalLM.from_pretrained("google/gemma-4-E2B")
model = model.to(device)
tokenizer = AutoTokenizer.from_pretrained("google/gemma-4-E2B")

"""
This step is optional but helpful for offline work. We will save the model and tokenizer locally so that we can load them without needing to download them again.
"""
# model.save_pretrained("offline/gemma4-emotion")
# tokenizer.save_pretrained("offline/gemma4-emotion")

# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# model = Gemma4ForCausalLM.from_pretrained("offline/gemma4-emotion")
# model = model.to(device)
# tokenizer = AutoTokenizer.from_pretrained("offline/gemma4-emotion")

"""
It is a good idea to check the module names because Gemma implementations can change depending on Transformers version or the Gemma version. Let's print the module names to make sure we are applying LoRA to the correct layers.
"""
for name, module in model.named_modules():
    print(name)


"""
Now we will set up the LoRA configuration. The parameters are as follows:
r: is the rank of the update matrices (smaller = fewer parameters to train)
lora_alpha: is the scaling factor for the update matrices (larger = more impact on the model)
lora_dropout: is the dropout rate for the LoRA layers (0.1 = 10% dropout)
target_modules: is a list of the model's modules to which the LoRA adapters will be applied
"""

lora_config = LoraConfig(
    task_type=TaskType.CAUSAL_LM,
    r=16,
    lora_alpha=32,
    lora_dropout=0.05,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
    "gate_proj", "up_proj", "down_proj"]
    )

"""
Now we will attach the Lora adapters to the pre-trained model using the get_peft_model function. This will create a new model that includes the LoRA layers.
"""

model = get_peft_model(model, lora_config)
model = cast(PeftModel, model)
""" 
Here we will print the number of trainable parameters in the model to verify that only a small portion of the model's parameters are being updated during training.
"""
model.print_trainable_parameters()

"""
load_dataset is a function from the Hugging Face Datasets library that allows us to easily load and preprocess datasets for training and evaluation. In this case, we are loading the "cardiffnlp/tweet_eval" dataset with the "emotion" configuration, which contains tweets labeled with different emotions.
"""
ds = load_dataset("cardiffnlp/tweet_eval", "emotion")

"""
Since the dataset is not in the format that is expected by SFTTrainer, we will need to do some small preprocessing. The SFTTrainer expects the input to be a dictionary with a "text" key that contains the input text and a "label" key that contains the label for the input. We will create a function that formats the examples in the dataset to match this format.
"""

id2label = {
    0: "anger",
    1: "joy",
    2: "optimism",
    3: "sadness",
}

def format_example(example):
    return {
        "text": (
            f"Classify the emotion of this tweet.\n"
            f"Tweet: {example['text']}\n"
            f"Emotion: {id2label[example['label']]}"
        )
    }

ds = ds.map(format_example)

# Configure the training arguments for the SFTTrainer. 
training_args = SFTConfig(
    output_dir="lora-gemma4-emotion",
    num_train_epochs=10,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
    learning_rate=2e-4,
    logging_steps=10,
    max_length=256,
    eval_strategy="epoch",
    save_strategy="epoch",
    load_best_model_at_end=True,
    bf16=False,
    fp16=True,
)
# Create the SFTTrainer instance, which will handle the training loop, evaluation, and saving of the model
trainer = SFTTrainer(
    model=model,
    train_dataset=ds["train"],
    eval_dataset=ds["validation"],
    args=training_args,
    processing_class=tokenizer,
)

"""
Now let's train the model using the trainer's train method. This will start the training loop and the model will be fine-tuned on the emotion classification task using the LoRA adapters.
"""
trainer.train()

# For reproducebility, we will want save the LoRA adapter weights after training. This can be done using the save_pretrained method of the model, which will save only the LoRA adapter weights and not the full model weights.

model.save_pretrained("lora-gemma4-emotion")

# Now we will reload the base model and merge the LoRA weights into the base model using the merge_and_unload function. This will create a new model that includes the fine-tuned weights from the LoRA training. This step is necessary to save the model in a format that can be easily loaded and used for inference without needing to set up the LoRA configuration again. We have effectively "baked in" our fine-tuned weights without needing to train the entire 5B parameter Gemma 4 model.

base_model = Gemma4ForCausalLM.from_pretrained("google/gemma-4-E2B")
lora_model = PeftModel.from_pretrained(base_model, "./lora-gemma4-emotion")

# If working offline
# base_model = AutoModelForCausalLM.from_pretrained("offline/gemma4-emotion")
# lora_model = PeftModel.from_pretrained(base_model, "lora-gemma4-emotion/adapter")


# Let's create use the Test dataset from TweetEval examples to infrence our model against. We will generate predictions for each example in the test set and compare them to the ground truth labels. We will also implement a more robust output parsing strategy to handle cases where the model's output may not exactly match the expected emotion labels, which is common in generative models. This will allow us to evaluate the model's performance more accurately, even when it produces slightly different outputs than the exact label names.

"""
LoRA-Adapted Model Evaluation
"""

test_ds = ds["test"]

# Helper function for improved output matching
def match_emotion_label(model_output, id2label):
    """
    Match model output to emotion label using multiple strategies.
    Returns (label_id, confidence_score) or (None, 0.0) if no confident match found.
    """
    output_lower = model_output.lower().strip()
    
    # Strategy 1: Exact word match for each label
    for label_id, label_text in id2label.items():
        if label_text.lower() == output_lower:
            return label_id, 1.0  # Perfect match
    
    # Strategy 2: Check if label appears as substring
    for label_id, label_text in id2label.items():
        if label_text.lower() in output_lower:
            return label_id, 0.9  # High confidence substring match
    
    # Strategy 3: Check if label appears as a word
    words = output_lower.split()
    for label_id, label_text in id2label.items():
        if label_text.lower() in words:
            return label_id, 0.8  # Good confidence word match
    
    return None, 0.0  # No confident match

# Store predictions and ground truth
lora_predictions = []
lora_ground_truth = []
lora_unmatched = []
lora_match_scores = []

print("LoRA-Adapted Model Results")
print("-" * 60)

for example in test_ds:
    prompt = f"Classify the emotion of this tweet.\nTweet: {example['text']}\nEmotion:"
    inputs = tokenizer(prompt, return_tensors="pt").to(device)

    outputs = model.generate(
        **inputs,
        max_new_tokens=3,
        do_sample=False,
        temperature=0.0
    )

    pred = tokenizer.decode(
        outputs[0][inputs["input_ids"].shape[-1]:],
        skip_special_tokens=True
    ).strip()

    # Match prediction to label using improved strategy
    pred_label, confidence = match_emotion_label(pred, id2label)
    
    lora_ground_truth.append(example["label"])
    
    if pred_label is not None:
        lora_predictions.append(pred_label)
        lora_match_scores.append(confidence)
    else:
        # Treat unmatched as incorrect: assign to a dummy class (-1)
        lora_predictions.append(-1)  # Invalid prediction
        lora_unmatched.append({
            'model_output': pred,
            'ground_truth': id2label[example['label']]
        })

# Calculate metrics
lora_accuracy = accuracy_score(lora_ground_truth, lora_predictions)
lora_f1 = f1_score(lora_ground_truth, lora_predictions, average='weighted', zero_division=0)

print(f"Accuracy (includes unmatched as incorrect): {lora_accuracy:.4f}")
print(f"F1 Score (Weighted, includes unmatched): {lora_f1:.4f}")
print(f"Successfully matched predictions: {len(lora_match_scores)}/{len(test_ds)}")
print(f"Unmatched predictions: {len(lora_unmatched)} ({len(lora_unmatched)/len(test_ds)*100:.1f}%)")
if lora_match_scores:
    print(f"Average match confidence (for matched only): {np.mean(lora_match_scores):.3f}")

print("\nClassification Report (including unmatched as incorrect):")
print(classification_report(lora_ground_truth, lora_predictions, 
                          labels=[0, 1, 2, 3, -1],
                          target_names=list(id2label.values()) + ['UNMATCHED'], zero_division=0))

print("Confusion Matrix:")
print(confusion_matrix(lora_ground_truth, lora_predictions, labels=[0, 1, 2, 3, -1]))

if lora_unmatched:
    print(f"\nSample Unmatched Predictions ({min(5, len(lora_unmatched))} of {len(lora_unmatched)}):")
    for i, item in enumerate(lora_unmatched[:5]):
        print(f"  {i+1}. Model output: '{item['model_output']}' | Ground truth: {item['ground_truth']}")

# %%
"""
Base Model Evaluation and Model Comparison
"""

# Load the base model (without LoRA)
base_model_eval = Gemma4ForCausalLM.from_pretrained("offline/gemma4-emotion").to(device)

base_predictions = []
base_ground_truth = []
base_unmatched = []
base_match_scores = []

print("\n" + "="*60)
print("Base Model Results (Zero-Shot)")
print("="*60)

for example in test_ds:
    prompt = f"Classify the emotion of this tweet.\nTweet: {example['text']}\nEmotion:"
    inputs = tokenizer(prompt, return_tensors="pt").to(device)

    with torch.no_grad():
        outputs = base_model_eval.generate(
            **inputs,
            max_new_tokens=3,
            do_sample=False,
            temperature=0.0
        )

    pred = tokenizer.decode(
        outputs[0][inputs["input_ids"].shape[-1]:],
        skip_special_tokens=True
    ).strip()

    # Match prediction to label using improved strategy
    pred_label, confidence = match_emotion_label(pred, id2label)
    
    base_ground_truth.append(example["label"])
    
    if pred_label is not None:
        base_predictions.append(pred_label)
        base_match_scores.append(confidence)
    else:
        # Treat unmatched as incorrect: assign to a dummy class (-1)
        base_predictions.append(-1)  # Invalid prediction
        base_unmatched.append({
            'model_output': pred,
            'ground_truth': id2label[example['label']]
        })

# Calculate metrics
base_accuracy = accuracy_score(base_ground_truth, base_predictions)
base_f1 = f1_score(base_ground_truth, base_predictions, average='weighted', zero_division=0)

print(f"Accuracy (includes unmatched as incorrect): {base_accuracy:.4f}")
print(f"F1 Score (Weighted, includes unmatched): {base_f1:.4f}")
print(f"Successfully matched predictions: {len(base_match_scores)}/{len(test_ds)}")
print(f"Unmatched predictions: {len(base_unmatched)} ({len(base_unmatched)/len(test_ds)*100:.1f}%)")
if base_match_scores:
    print(f"Average match confidence (for matched only): {np.mean(base_match_scores):.3f}")

print("\nClassification Report (including unmatched as incorrect):")
print(classification_report(base_ground_truth, base_predictions, 
                          labels=[0, 1, 2, 3, -1],
                          target_names=list(id2label.values()) + ['UNMATCHED'], zero_division=0))

print("Confusion Matrix:")
print(confusion_matrix(base_ground_truth, base_predictions, labels=[0, 1, 2, 3, -1]))

if base_unmatched:
    print(f"\nSample Unmatched Predictions ({min(5, len(base_unmatched))} of {len(base_unmatched)}):")
    for i, item in enumerate(base_unmatched[:5]):
        print(f"  {i+1}. Model output: '{item['model_output']}' | Ground truth: {item['ground_truth']}")

print("\n" + "="*60)
print("Model Comparison")
print("="*60)

# Create comparison dataframe
comparison_df = pd.DataFrame({
    'Model': ['Base Model (Zero-Shot)', 'LoRA-Adapted Model'],
    'Accuracy': [base_accuracy, lora_accuracy],
    'F1 Score': [base_f1, lora_f1],
    'Matched %': [f"{(len(base_match_scores)/len(test_ds)*100):.1f}%", 
                  f"{(len(lora_match_scores)/len(test_ds)*100):.1f}%"]
})

print("\n" + comparison_df.to_string(index=False))

# Calculate improvements
accuracy_improvement = ((lora_accuracy - base_accuracy) / base_accuracy * 100) if base_accuracy > 0 else 0
f1_improvement = ((lora_f1 - base_f1) / base_f1 * 100) if base_f1 > 0 else 0

print(f"\nAccuracy Improvement: {accuracy_improvement:+.2f}%")
print(f"F1 Score Improvement: {f1_improvement:+.2f}%")

# Visualization
fig, axes = plt.subplots(1, 3, figsize=(15, 5))

# Accuracy comparison
models = ['Base Model\n(Zero-Shot)', 'LoRA-Adapted']
accuracies = [base_accuracy, lora_accuracy]
axes[0].bar(models, accuracies, color=['#ff6b6b', '#51cf66'], edgecolor='black', linewidth=1.5)
axes[0].set_ylabel('Accuracy')
axes[0].set_title('Accuracy Comparison')
axes[0].set_ylim([0, 1])
for i, v in enumerate(accuracies):
    axes[0].text(i, v + 0.02, f'{v:.4f}', ha='center', fontweight='bold')
axes[0].grid(axis='y', alpha=0.3)

# F1 Score comparison
f1_scores = [base_f1, lora_f1]
axes[1].bar(models, f1_scores, color=['#ff6b6b', '#51cf66'], edgecolor='black', linewidth=1.5)
axes[1].set_ylabel('F1 Score (Weighted)')
axes[1].set_title('F1 Score Comparison')
axes[1].set_ylim([0, 1])
for i, v in enumerate(f1_scores):
    axes[1].text(i, v + 0.02, f'{v:.4f}', ha='center', fontweight='bold')
axes[1].grid(axis='y', alpha=0.3)

# Match rate comparison
match_rates = [(len(base_match_scores)/len(test_ds)*100), (len(lora_match_scores)/len(test_ds)*100)]
axes[2].bar(models, match_rates, color=['#ff6b6b', '#51cf66'], edgecolor='black', linewidth=1.5)
axes[2].set_ylabel('Successful Match Rate (%)')
axes[2].set_title('Prediction Parsing Success Rate')
axes[2].set_ylim([0, 105])
for i, v in enumerate(match_rates):
    axes[2].text(i, v + 2, f'{v:.1f}%', ha='center', fontweight='bold')
axes[2].grid(axis='y', alpha=0.3)

plt.tight_layout()
plt.show()

print("\nSample Predictions (First 10 Matched Examples)")
print("-" * 80)

# Get matched indices only
matched_count = 0
for i in range(len(test_ds)):
    if matched_count >= 10:
        break
    
    # Find the i-th matched prediction
    matched_idx = 0
    for j in range(len(base_predictions)):
        if base_predictions[j] != -1:
            if matched_idx == i:
                example = test_ds[j]
                base_pred_label_id = base_predictions[j]
                lora_pred_label_id = lora_predictions[j]
                print(f"\nExample {matched_count + 1}:")
                print(f"  Tweet: {example['text'][:80]}")
                print(f"  Ground Truth: {id2label[example['label']]}")
                print(f"  Base Model: {id2label[base_pred_label_id] if base_pred_label_id != -1 else 'UNMATCHED'}")
                print(f"  LoRA Model: {id2label[lora_pred_label_id] if lora_pred_label_id != -1 else 'UNMATCHED'}")
                matched_count += 1
                break
            matched_idx += 1
