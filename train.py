import argparse

import evaluate
import numpy as np
from datasets import load_dataset
from torchvision.transforms import (Compose, Normalize, RandomHorizontalFlip,
                                    RandomResizedCrop, RandomRotation,
                                    RandomVerticalFlip, ToTensor)
#from huggingface_hub import HfFolder
from transformers import (AutoImageProcessor, AutoModelForImageClassification,
                          DefaultDataCollator, Trainer, TrainingArguments)


def get_id2label_and_label2id(dataset):
    """To make it easier for the model  to get the label name from the label id, 
    create a dictionary that maps the label name to an integer and vice versa
    
    Args:
        dataset : Hf dataset to retrieve the labels to ids mapping

    Returns:
        tuple(dict, dict): two dictionary mapping from id to label for the first one,
        and from label to id for the second.
        
    """
    labels = dataset['train'].features['label'].names
    label2id, id2label = dict(),dict()
    
    for idx, label in enumerate(labels):
        label2id[label] = str(idx)
        id2label[str(idx)] = label
        
    return id2label, label2id

# Apply the transformations and convert the image to (200, 200, 3) 
# Return it as 'pixel_values'- the inputs of the model- of the image.
def transforms(examples):
    examples["pixel_values"] = [_transforms(img.convert("RGB").resize((200,200))) for img in examples["image"]]
    del examples["image"]
    return examples

# Load the evaluate function
f1 = evaluate.load("f1")
def compute_metrics(eval_pred):
    predictions, labels = eval_pred
    predictions = np.argmax(predictions, axis=1)
    return f1.compute(predictions=predictions, references=labels, average="weighted")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    
    
    ## Required parameters
    parser.add_argument("--data_dir", default="./data/dataset", type=str, required=False,
                        help="The input dataset directory. Should contain an eponymous folder for each pokemon, containing more than 50 images of said pokemon.")
    parser.add_argument("--model_dir", default="./model", type=str, required=False,
                        help="The output directory where the model checkpoints will be written.")
    parser.add_argument("--data_augmentation", action='store_true',
                        help="Specify whether or not to use data augmentation on training data.")
    
    
    args = parser.parse_args()
    
    
    # Load dataset
    dataset = load_dataset("imagefolder", data_dir=args.data_dir, download_mode="force_redownload")
    print("Training data loaded!")
    # Split dataset into training and test set
    dataset = dataset['train'].train_test_split(test_size=0.2, seed=42)
    
    # Get the list of label classes
    labels = dataset['train'].features['label'].names
    # Retrieve the mapping dictionaries from id to label, and reverse 
    id2label, label2id = get_id2label_and_label2id(dataset)
    
    # Load Image Preprocessor
    checkpoint = "google/vit-base-patch16-224-in21k"
    image_processor = AutoImageProcessor.from_pretrained(checkpoint)


    # Apply some image transformations to the images to make the model more robust against overfitting.
    normalize = Normalize(mean=image_processor.image_mean, std=image_processor.image_std)
    size = (
        image_processor.size["shortest_edge"]
        if "shortest_edge" in image_processor.size
        else (image_processor.size["height"], image_processor.size["width"])
    )
    
    if args.data_augmentation:
        # Do training with data augmentation
        _transforms = Compose([RandomResizedCrop(size), RandomHorizontalFlip(p=0.2), RandomVerticalFlip(p=0.2),
                        RandomRotation((-45,45)), ToTensor(), normalize])
    else:
        # Do training with minimal data augmentation
        _transforms = Compose([RandomResizedCrop(size), ToTensor(), normalize])
    
    # Apply the preprocessing function over the entire dataset:
    dataset = dataset.with_transform(transforms)
    print("Training data preprocessed!")
    
    # Define the DataCollator to process the data in batches.
    # Unlike other data collators in 🤗 Transformers, the DefaultDataCollator does not apply additional preprocessing such as padding.
    data_collator = DefaultDataCollator()

    # Load the pretrained model
    model = AutoModelForImageClassification.from_pretrained(
        checkpoint,
        num_labels=len(labels),
        id2label=id2label,
        label2id=label2id,
    )
    print("Pretrained model loaded!")
    
    
    # Define the training arguments (with hyperparameters selected with anterior hp search)
    training_args = TrainingArguments(
        output_dir= args.model_dir,
        remove_unused_columns=False,
        evaluation_strategy="epoch",
        save_strategy="epoch",
        learning_rate=6.56462271373806e-05,
        per_device_train_batch_size=4,
        gradient_accumulation_steps=4,
        per_device_eval_batch_size=16,
        num_train_epochs=9,
        warmup_ratio=0.1,
        logging_steps=10,
        optim="adamw_torch_fused", # improved optimizer
        # logging & evaluation strategies
        logging_dir=f"{args.model_dir}/logs",
        logging_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="f1",
        # push to hub parameters
        #report_to="tensorboard",
        #push_to_hub=True,
        #hub_strategy="every_save",
        #hub_token=HfFolder.get_token(),
        #hub_model_id=repository_id
    )

    # Define the Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        data_collator=data_collator,
        train_dataset=dataset["train"],
        eval_dataset=dataset["test"],
        tokenizer=image_processor,
        compute_metrics=compute_metrics,
    )

    # Train the model 
    trainer.train()
    
    print("Training done!")