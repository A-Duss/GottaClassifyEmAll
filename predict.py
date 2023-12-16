import argparse

import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image
from transformers import AutoImageProcessor, AutoModelForImageClassification


def preprocess(processor: AutoImageProcessor, image):
    return processor(image.convert("RGB").resize((200,200)), return_tensors="pt")

def predict(model: AutoModelForImageClassification, inputs, k=5):
    
    # Forward the image to the model and retrieve the logits 
    with torch.no_grad():
        logits = model(**inputs).logits
    
    # Convert the retrieved logits into a vector of probabilities for each class
    probabilities = torch.softmax(logits[0], dim=0).tolist()
    
    # Discriminate wether or not the inputted image was an image of a Pokemon
    # Compute the variance of the vector of probabilities 
        # The spread of the probability values is a good represent of the confusion of the model
        # Or in other words, its confidence => the greater the spread, the lower its confidence 
    variance = np.var(probabilities)
    
    # Too great of a spread: it is likely the image provided did not correspond to any known classes
    if variance < 0.001: #not a pokemon
       predicted_label = 'not a pokemon' 
       probability = -1
       (top_k_labels, top_k_probability) = '_', '_'
    else: # it is a pokemon
        # Retrieve the predicted class (pokemon)
        predicted_id = logits.argmax(-1).item()
        predicted_label = model.config.id2label[predicted_id]
        # Retrieve the probability for the predicted class, and format it to 2 decimals
        probability = round(probabilities[predicted_id]*100,2)
        # Retrieve the top 5 classes and their probabilities
        #top_k_labels = [model.config.id2label[key] for key in np.argpartition(logits.numpy(), -k)[-k:]]
        #top_k_probability = [round(prob*100,2) for prob in np.sort(probabilities.numpy())[-k:]]
        
    return predicted_label, probability #, top_k_labels, top_k_probability


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    
    
    ## Required parameters
    parser.add_argument("--img_path", default="./data/sample_imgs/01abra.jpg", type=str, required=False,
                        help="The path to the image to classify.")
    parser.add_argument("--model_dir", default="./model", type=str, required=False,
                        help="The directory where the model checkpoints are stored.")
    parser.add_argument("--load_from_hf", action='store_true',
                        help="If specified, load the model used in HuggingFace Space 'GottaClassifyEmAll'.")
    
    
    args = parser.parse_args()


    # Loading the finetuned image classifier model and its processor
    if args.load_from_hf:
        # Loading the model used in HuggingFace Space 'GottaClassifyEmAll' from HuggingFace
        image_processor = AutoImageProcessor.from_pretrained("Dusduo/Pokemon-classification-1stGen")
        model = AutoModelForImageClassification.from_pretrained("Dusduo/Pokemon-classification-1stGen")
    else:
        image_processor = AutoImageProcessor.from_pretrained(args.model_dir)
        model = AutoModelForImageClassification.from_pretrained(args.model_dir)

    # Load the image to classify
    image = Image.open(args.img_path)

    # Preprocess the image for the model
    model_inputs = preprocess(image_processor, image)

    # Get prediction
    prediction, probability = predict(model, model_inputs, 5) #, (top_k_labels, top_k_probability)
    
    # If image was classified as not a pokemon
    if probability==-1:
        print('''I am sorry I am having trouble finding a matching pokemon. 
**Potential explanations:**
    - The image provided is a Pokemon but not from the 1st Generation. 
    - The image provided is not a Pokemon. 
    - There are too many entities on the image.''')
    
    # If the image was classified as a pokemon
    else:
        print(f"It is a(n) {prediction} image.")
        plt.imshow(image)
        plt.title(f"Model guess: {prediction} \n Confidence: {probability}%")
        plt.show()