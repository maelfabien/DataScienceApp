import streamlit as st
import spacy

def return_pos(value):
	nlp = spacy.load("en_core_web_sm")
	doc = nlp(value)
	return [(X.text, X.pos_) for X in doc]

def add_pos():
	st.title("Part-Of-Speech Tagging")
	st.write("Part-Of-Speech Tagging is the process by which tag each word of a sentence with its correspondding grammatical function (determinant, noun, ajective...) using a mix of deep learning (Long Short-Term Memory networks) and probabilitstic approach (Conitional Random Fields). Just like Named Entity Recognition, this type of algorithm is generally trained on large corpuses such as Wikipedia. This algorithm relies on SpaCy, a state-of-the-art library which implements natural language processing models in English and French.")
	nlp = spacy.load("en_core_web_sm")

	input_sent = st.text_input("Input Sentence", "Your input sentence goes here")

	for res in return_pos(input_sent):
		st.write(res[0], ":", res[1])
