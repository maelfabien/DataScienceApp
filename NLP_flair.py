import streamlit as st
import flair
global flair_sentiment
import numpy as np

def load_flair():
	return flair.models.TextClassifier.load('en-sentiment')

def add_flair():

	flair_sentiment = load_flair()

	st.title("Sentiment Detection")
	st.write("Sentiment Detection from text is a classical problem. This is used when you try to predict the sentiment of comments on a restaurant review website for example, or when you receive customer support messages and want to classify them. This task usually involves Deep Learning algorithms such as Long Short-Term Memory (LSTMs). This algorithm relies on Flair, a library developped by Zalando (shopping site) research team.")
	
	input_sent = st.text_input("Input Sentence", "Although quite poorly rated, the story was interesting and I enjoyed it.")

	s = flair.data.Sentence(input_sent)
	flair_sentiment.predict(s)

	st.write('Your sentence is ', str(s.labels[0]).split()[0].lower(), " with ", str(np.round(float(str(s.labels[0]).split()[1][1:-1]),3)*100), " % probability.")