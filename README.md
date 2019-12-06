# Data Science App

In this app, I'll be deploying NLP, Computer Vision, Speech Processing and Generative models in a single application on Streamlit. 

## What's inside?

The idea is to leverage *pre-trained* models as well as algorithm built on my side. It's a good opportunity for me to become more familiar with all pre-trained models available in the open-source community. 

The algorithms covered are:
- Natural Language Processing
	- Part-Of-Speech Tagging with `SpaCy`
	- Named Entity Recognition with `SpaCy`
	- Sentiment Classification with `Flair`
	- Question Answering with `AllenNLP`
- Computer Vision
	- Object Detection with `Yolov3`
	- Face Detection with `Haar Cascade Classifier`
	- Face Recognition with `facerecognition`
- Speech Processing
	- Voice Activity Detection
	- Speaker Identification
- Generative Models
	- A GAN to generate digits
	- A GAN to generate faces
	- Text-to-images

I integrate and deploy everything using Streamlit and Render.com. 

## How to use it?

The application looks like this:

![image](images/screen_home.png)

It is currently hosted by Render.com and each push leads to a new build of the app. I am using the standard plan (2Gb RAM). If you would like to contribute, feel free to submit a PR.

To run it locally, clone this project and run :

```bash
pip install -r requirements
```

Rrun the app:

```bash
streamlit run app.py
```
