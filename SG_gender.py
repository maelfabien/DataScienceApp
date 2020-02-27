import streamlit as st

# Data manipulation
import numpy as np
import matplotlib.pyplot as plt

# Feature extraction
import scipy
import librosa
import python_speech_features as mfcc
import os
from scipy.io.wavfile import read

# Model training
from sklearn.mixture import GaussianMixture as GMM
from sklearn import preprocessing
import pickle

# Live recording
import sounddevice as sd
import soundfile as sf

def add_gender():
	st.title("Voice Gender Detection")
	st.write("This application demonstrates a simple Voice Gender Detection. Voice gender identification relies on three important steps.")
	st.write("- Extracting from the training set MFCC features (13 usually) for each gender")
	st.write("- Train a GMM on those features")
	st.write("- In prediction, compute the likelihood of each gender using the trained GMM, and pick the most likely gender")


	st.subheader("Ready to try it on your voice?")

	st.sidebar.title("Parameters")
	duration = st.sidebar.slider("Recording duration", 0.0, 10.0, 3.0)

	def get_MFCC(sr,audio):
	    """
	    Extracts the MFCC audio features from a file
	    """
	    features = mfcc.mfcc(audio, sr, 0.025, 0.01, 13, appendEnergy = False)
	    features = preprocessing.scale(features)
	    return features

	def record_and_predict(gmm_male, gmm_female, sr=16000, channels=1, duration=3, filename='pred_record.wav'):
	    """
	    Records live voice and returns the identified gender
	    """ 
	    recording = sd.rec(int(duration * sr), samplerate=sr, channels=channels).reshape(-1)
	    sd.wait()
	    
	    features = get_MFCC(sr,recording)
	    scores = None

	    log_likelihood_male = np.array(gmm_male.score(features)).sum()
	    log_likelihood_female = np.array(gmm_female.score(features)).sum()

	    if log_likelihood_male >= log_likelihood_female:
	        return("Male")
	    else:
	        return("Female")

	gmm_male = pickle.load(open('male.gmm','rb'))
	gmm_female = pickle.load(open('female.gmm','rb'))


	if st.button("Start Recording"):
	    with st.spinner("Recording..."):
	        gender = record_and_predict(gmm_male, gmm_female, duration=duration)
	        st.write("The identified gender is: " + gender)

	st.subheader("Limitations")
	st.write("The accuracy in test remains to improve (63%). When a user plays with his or her voice and tries to immitate the other gender, the GMM gets fooled and predicts the wrong gender. This is also due to the training data that it has seen so far which were extracted from AudioSet and Youtube.")
	st.write("I am currenlty thinking about adding a re-train button to this application and letting the user re-train the model by incorporating his self-annotated data.")
	st.write("For more information, check this article: https://appliedmachinelearning.blog/2017/06/14/voice-gender-detection-using-gmms-a-python-primer/")
