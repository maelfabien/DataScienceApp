import streamlit as st
import pandas as pd
import numpy as np
import altair as alt
import spacy
from allennlp import pretrained
import matplotlib.pyplot as plt
import flair
import seaborn as sns
from vega_datasets import data
global flair_sentiment
import cv2
import pylab
import imageio
#import dlib
from imutils import face_utils


import NLP_ner
import NLP_pos
import NLP_flair
import NLP_qa

import CV_face
import CV_yolo

import SG_activity

st.sidebar.title("Category")

topic = st.sidebar.radio("Pick a topic", ["Natural Language", "Computer Vision", "Speech Processing","Data Visualization", "Generative Models"])

# I. NLP

if topic == "Natural Language":

	sub_topic = st.sidebar.radio("Algorithm", ["Named Entity Detection", "Part-Of-Speech Tagging", "Sentiment Detection", "Question Answering"])

	if sub_topic == "Named Entity Detection":

		NLP_ner.add_ner()

	elif sub_topic == "Part-Of-Speech Tagging":

		NLP_pos.add_pos()

	elif sub_topic == "Sentiment Detection":

		NLP_flair.add_flair()

	elif sub_topic == "Question Answering":

		NLP_qa.add_qa()

# II. Computer Vision

elif topic == "Computer Vision":

	sub_topic = st.sidebar.radio("Algorithm", ["Face Detection", "Object Detection"])
	# To add: "Face Recognition", "Sentiment Detection"

	if sub_topic == "Face Detection":

		CV_face.add_face()

	elif sub_topic == "Object Detection":

		CV_yolo.add_yolo()


# III. Speech Processing

elif topic == "Speech Processing":
	sub_topic = st.sidebar.radio("Algorithm", ["Voice Activity Detection", "Gender Identification"])

	# To add : "Speaker Identification", "Text to Speech", "Speech to Text"

	if sub_topic == "Voice Activity Detection":
		SG_activity.add_activity()

	elif sub_topic == "Voice Activity Detection":
		SG_gender.add_gender()

# IV. Data Visualization

elif topic == "Data Visualization":

    sub_topic = st.sidebar.radio("Project", ["Dataset Explorer", "New-York Uber"])
    page = st.sidebar.selectbox("Choose a page", ["Table", "Exploration"])

    if sub_topic == "Dataset Explorer":

    	df = data.cars()

    	def visualize_data(df, x_axis, y_axis):
    		graph = alt.Chart(df).mark_circle(size=60).encode(
				x=x_axis,
				y=y_axis,
				color='Origin',
				tooltip=['Name', 'Origin', 'Horsepower', 'Miles_per_Gallon']).interactive()
    		st.write(graph)

    	if page == "Table":
    		st.header("Explore the raw table.")
    		st.write("Please select a page on the left.")
    		st.write(df)
    	elif page == "Exploration":
	    	st.title("Data Exploration")
	    	x_axis = st.selectbox("Choose a variable for the x-axis", df.columns, index=3)
	    	y_axis = st.selectbox("Choose a variable for the y-axis", df.columns, index=4)
	    	visualize_data(df, x_axis, y_axis)

    elif sub_topic == "New-York Uber":
    	DATE_TIME = "date/time"
    	DATA_URL = ("http://s3-us-west-2.amazonaws.com/streamlit-demo-data/uber-raw-data-sep14.csv.gz")
    	st.title("Uber Pickups in New York City")
    	st.write("Uber pickups geographical distribution in New York City.")

    	@st.cache(persist=True)
    	def load_data(nrows):
    		data = pd.read_csv(DATA_URL, nrows=nrows)
    		lowercase = lambda x: str(x).lower()
    		data.rename(lowercase, axis="columns", inplace=True)
    		data[DATE_TIME] = pd.to_datetime(data[DATE_TIME])
    		return data

    	data = load_data(100000)
    	hour = st.slider("Hour to look at", 0, 23)
    	data = data[data[DATE_TIME].dt.hour == hour]
    	st.subheader("Geo data between %i:00 and %i:00" % (hour, (hour + 1) % 24))
    	midpoint = (np.average(data["lat"]), np.average(data["lon"]))
    	st.deck_gl_chart(viewport={"latitude": midpoint[0], "longitude": midpoint[1], "zoom": 11, "pitch": 50,}, layers=[{"type": "HexagonLayer", "data": data, "radius": 100, "elevationScale": 4, "elevationRange": [0, 1000], "pickable": True, "extruded": True,}],)
    	st.subheader("Breakdown by minute between %i:00 and %i:00" % (hour, (hour + 1) % 24))
    	filtered = data[(data[DATE_TIME].dt.hour >= hour) & (data[DATE_TIME].dt.hour < (hour + 1))]
    	hist = np.histogram(filtered[DATE_TIME].dt.minute, bins=60, range=(0, 60))[0]
    	chart_data = pd.DataFrame({"minute": range(60), "pickups": hist})
    	st.write(alt.Chart(chart_data, height=150).mark_area(interpolate='step-after',line=True).encode(x=alt.X("minute:Q", scale=alt.Scale(nice=False)),y=alt.Y("pickups:Q"),tooltip=['minute', 'pickups']))
    	if st.checkbox("Show raw data", False):
    		st.subheader("Raw data by minute between %i:00 and %i:00" % (hour, (hour + 1) % 24))
    		st.write(data)

# V. Generative Models

elif topic == "Generative Models":

	sub_topic = st.sidebar.radio("Algorithm", ["Generate a face", "Generate art", "Text to image"])

