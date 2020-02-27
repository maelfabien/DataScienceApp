import streamlit as st
import spacy

def return_NER(value):
	nlp = spacy.load("en_core_web_sm")
	doc = nlp(value)
	return [(X.text, X.label_) for X in doc.ents]

def add_ner():
	
	st.title("Named Entity Recognition")

	st.write("Named Entity Recognition is the process by which we identify named entities (persons, organisations, governments, money...) using a mix of deep learning (Long Short-Term Memory networks) and probabilitstic approach (Conitional Random Fields). This requires to train an algorithm to make a difference between Apple (a fruit) and Apple (the brand) based on contextual information. This type of algorithm is generally trained on large corpuses such as Wikipedia. This algorithm relies on SpaCy, a state-of-the-art library which implements natural language processing models in English and French.")
	nlp = spacy.load("en_core_web_sm")

	input_sent = st.text_input("Input Sentence", "Orange sells 1 million Apple's phones each year.")

	for res in return_NER(input_sent):
		st.write(res[0], ":", res[1])
