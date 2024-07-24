# Persephone - AI Flower Classifier

This is my final project for the _AI Programming with Python - Bertelsmann_ course. It's an AI developed and trained to recognize and catalog different types of flowers.

## Table of Contents

    Overview
    Project Structure
    Installation
    Usage
    	Model Training
    	Prediction

## Overview

This project involves training a neural network to classify images of flowers into different categories. The model is trained on a dataset of flower images and can predict the type of flower from new images provided by the user. With it, you can create your own AI using VGG or Densenet architecture, tweaking some settings to get the best result. During my tests, I've managed to get up to 92% of accuracy.
This project consists of two parts: The first one is a Jupyter Notebook with the final assignment. The second part consists of a Command Line App where you can pass some parameters to customize your model.
The AI needs some images to train, another set of different images to validate, and - finally - different images to test. You can find the images on your own, or you can download the set of images I've used here: [Flowers](https://drive.google.com/file/d/17D5HcQd9XaPKQUxjJT8sKsOoo4XkOcOr/view?usp=sharing "Flowers") .
Just download it and paste the directory in the root of this project.

I tried to organize the code the best I could in the hope other people could better understand how exactly AI training works. Nonetheless, you need to know that AI programming sums up to two words: **headache and fun**.

_Cheers!_

## Project StructureProject Structure

```bash
_persephone/_
├── assets/
├── cat_to_name.json
├── checkpoints/
├── classifier.py
├── flowers/
│ ├── manual_test/
│ │ ├── authurium.jpeg
│ │ ├── cautleya-spicata.jpg
│ │ ├── clematis.jpeg
│ │ ├── foxglove.jpeg
│ │ ├── hard-leaved-pocket-orchid.jpg
│ │ ├── orange-dahlia.jpg
│ │ └── sword-lily.jpeg
│ ├── test/
│ ├── train/
│ ├── valid/
├── Image Classifier Project.ipynb
├── loader_manager.py
├── model_manager.py
├── predict.py
├── requirements.txt
└── train.py
```

## Installation

###1. Clone this repository:

```bash
git clone https://github.com/your_username/ai-flower-classifier.git
cd ai-flower-classifier
```

###2. Install the required dependencies:

```
pip install -r requirements.txt
```

###3. (Optional) Download the flower image package and extract it in your root directory.
[Download your flowers here](https://drive.google.com/file/d/17D5HcQd9XaPKQUxjJT8sKsOoo4XkOcOr/view?usp=sharing "Flowers")

## Usage

###Model Training

To train the model, use the train.py script:

```
python train.py --data_dir [flowers_directory/train/]
```

This will train a new network on the dataset and save the model as a checkpoint. You can see how to spice things up using:

```
python train.py --help
```

**IMPORTANT:** If you decide to use your own images, you must add them in their respective folders inside `./flowers/`. There is one directory for each: "test", "valid" and "train". You can specify your own "train" directory, but you need to add images to "valid" and "test", as you can't change their location.

###Prediction

To predict the class of a flower, use the predict.py script as such:

```
python predict.py [path/to/flower_image.jpg] [path/to/checkpoints/checkpoint.pth]
```

This will output the top 5 most likely flower classes and their probabilities. You can see how to spice things up using:

```
python predict.py --help
```
