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

def load_flair():
	return flair.models.TextClassifier.load('en-sentiment')

st.sidebar.title("Category")

topic = st.sidebar.radio("Pick a topic", ["Natural Language", "Computer Vision", "Speech Processing","Data Visualization", "Generative Models"])

# I. NLP

if topic == "Natural Language":

	sub_topic = st.sidebar.radio("Algorithm", ["Named Entity Detection", "Part-Of-Speech Tagging", "Sentiment Detection", "Question Answering"])

	if sub_topic == "Named Entity Detection":

		st.title("Named Entity Recognition")

		st.write("Named Entity Recognition is the process by which we identify named entities (persons, organisations, governments, money...) using a mix of deep learning (Long Short-Term Memory networks) and probabilitstic approach (Conitional Random Fields). This requires to train an algorithm to make a difference between Apple (a fruit) and Apple (the brand) based on contextual information. This type of algorithm is generally trained on large corpuses such as Wikipedia. This algorithm relies on SpaCy, a state-of-the-art library which implements natural language processing models in English and French.")
		nlp = spacy.load("en_core_web_sm")

		def return_NER(value):
		    doc = nlp(value)
		    return [(X.text, X.label_) for X in doc.ents]

		input_sent = st.text_input("Input Sentence", "Orange sells 1 million Apple's phones each year.")

		for res in return_NER(input_sent):
			st.write(res[0], ":", res[1])

	elif sub_topic == "Part-Of-Speech Tagging":

		st.title("Part-Of-Speech Tagging")

		st.write("Part-Of-Speech Tagging is the process by which tag each word of a sentence with its correspondding grammatical function (determinant, noun, ajective...) using a mix of deep learning (Long Short-Term Memory networks) and probabilitstic approach (Conitional Random Fields). Just like Named Entity Recognition, this type of algorithm is generally trained on large corpuses such as Wikipedia. This algorithm relies on SpaCy, a state-of-the-art library which implements natural language processing models in English and French.")
		nlp = spacy.load("en_core_web_sm")

		def return_pos(value):
		    doc = nlp(value)
		    return [(X.text, X.pos_) for X in doc]

		input_sent = st.text_input("Input Sentence", "Your input sentence goes here")

		for res in return_pos(input_sent):
			st.write(res[0], ":", res[1])

	elif sub_topic == "Sentiment Detection":

		flair_sentiment = load_flair()

		st.title("Sentiment Detection")
		st.write("Sentiment Detection from text is a classical problem. This is used when you try to predict the sentiment of comments on a restaurant review website for example, or when you receive customer support messages and want to classify them. This task usually involves Deep Learning algorithms such as Long Short-Term Memory (LSTMs). This algorithm relies on Flair, a library developped by Zalando (shopping site) research team.")
		
		input_sent = st.text_input("Input Sentence", "Although quite poorly rated, the story was interesting and I enjoyed it.")

		s = flair.data.Sentence(input_sent)
		flair_sentiment.predict(s)

		st.write('Your sentence is ', str(s.labels[0]).split()[0].lower(), " with ", str(np.round(float(str(s.labels[0]).split()[1][1:-1]),3)*100), " % probability.")

	elif sub_topic == "Question Answering":

		st.title("Question Answering")
		st.write("Question Answering is a state-of-the-art research topic that has been arising with the evolution of Deep Learning algorithms. You write a query regarding a long input text, the algorithm goes through the text and identifies the region of the text which is the most likely to contain the answer. The graph below displays 'attention', the process by which neural networks learn to focus on certain parts of the long text. The darker the cell, the most important the information was to identify the answer.")
		
		predictor = st.cache(
		       pretrained.bidirectional_attention_flow_seo_2017,
		       ignore_hash=True  # the Predictor is not hashable
		)()

		article_choice = st.sidebar.selectbox("Article to query", ["Netflix", "Italy"])

		if article_choice == "Netflix":
			passage = st.text_area("Article", """Netflix, Inc. is an American media-services provider and production company headquartered in Los Gatos, California, founded in 1997 by Reed Hastings and Marc Randolph in Scotts Valley, California. The company's primary business is its subscription-based streaming service which offers online streaming of a library of films and television programs, including those produced in-house. As of April 2019, Netflix had over 148 million paid subscriptions worldwide, including 60 million in the United States, and over 154 million subscriptions total including free trials. It is available worldwide except in mainland China (due to local restrictions), Syria, North Korea, and Crimea (due to US sanctions). The company also has offices in the Netherlands, Brazil, India, Japan, and South Korea. Netflix is a member of the Motion Picture Association (MPA).
				Netflix's initial business model included DVD sales and rental by mail, but Hastings abandoned the sales about a year after the company's founding to focus on the initial DVD rental business. Netflix expanded its business in 2010 with the introduction of streaming media while retaining the DVD and Blu-ray rental business. The company expanded internationally in 2010 with streaming available in Canada, followed by Latin America and the Caribbean. Netflix entered the content-production industry in 2012, debuting its first series Lilyhammer.
				Since 2012, Netflix has taken more of an active role as producer and distributor for both film and television series, and to that end, it offers a variety of "Netflix Original" content through its online library. By January 2016, Netflix services operated in more than 190 countries. Netflix released an estimated 126 original series and films in 2016, more than any other network or cable channel. Their efforts to produce new content, secure the rights for additional content, and diversify through 190 countries have resulted in the company racking up billions in debt: $21.9 billion as of September 2017, up from $16.8 billion from the previous year. $6.5 billion of this is long-term debt, while the remaining is in long-term obligations. In October 2018, Netflix announced it would raise another $2 billion in debt to help fund new content.
				""")
			question = st.text_input("Question", "Where are the headquarters of Netflix?")
		elif article_choice == "Italy":
			passage = st.text_area("Passage", "Italy, officially the Italian Republic is a European country consisting of a peninsula delimited by the Alps and surrounded by several islands. Italy is located in south-central Europe, and it is also considered a part of western Europe. The country covers a total area of 301,340 km2 (116,350 sq mi) and shares land borders with France, Switzerland, Austria, Slovenia, and the enclaved microstates of Vatican City and San Marino. Italy has a territorial exclave in Switzerland (Campione) and a maritime exclave in the Tunisian Sea (Lampedusa). With around 60 million inhabitants, Italy is the fourth-most populous member state of the European Union. Due to its central geographic location in Southern Europe and the Mediterranean, Italy has historically been home to myriad peoples and cultures. In addition to the various ancient peoples dispersed throughout modern-day Italy, the most predominant being the Indo-European Italic peoples who gave the peninsula its name, beginning from the classical era, Phoenicians and Carthaginians founded colonies mostly in insular Italy, Greeks established settlements in the so-called Magna Graecia of Southern Italy, while Etruscans and Celts inhabited central and northern Italy respectively. An Italic tribe known as the Latins formed the Roman Kingdom in the 8th century BC, which eventually became a republic with a government of the Senate and the People. The Roman Republic initially conquered and assimilated its neighbours on the peninsula, eventually expanding and conquering parts of Europe, North Africa and Asia. By the first century BC, the Roman Empire emerged as the dominant power in the Mediterranean Basin and became a leading cultural, political and religious centre, inaugurating the Pax Romana, a period of more than 200 years during which Italy's law, technology, economy, art, and literature developed. Italy remained the homeland of the Romans and the metropole of the empire, whose legacy can also be observed in the global distribution of culture, governments, Christianity and the Latin script.")
			question = st.text_input("Question", "How large is Italy?")
		
		result = predictor.predict(question, passage)

		# From the result, we want "best_span", "question_tokens", and "passage_tokens"
		start, end = result["best_span"]
		
		question_tokens = result["question_tokens"]
		passage_tokens = result["passage_tokens"]
		mds = [f"**{token}**" if start <= i <= end else token if start - 10 <= i <= end + 10 else "" for i, token in enumerate(passage_tokens)]
		st.markdown(" ".join(mds))

		attention = result["passage_question_attention"]

		plt.figure(figsize=(12,12))
		sns.heatmap(attention, cmap="YlGnBu")
		plt.autoscale(enable=True, axis='x')
		plt.xticks(np.arange(len(question_tokens)), labels=question_tokens)
		st.pyplot()

# II. Computer Vision

elif topic == "Computer Vision":

	sub_topic = st.sidebar.radio("Algorithm", ["Object Detection", "Face Detection", "Face Recognition", "Sentiment Detection"])

	if sub_topic == "Face Detection":

		st.title("Face Detection")
		st.write("Face detection is a central algorithm in computer vision. The algorithm implemented below is a Haar Cascade Classifier. It detects several faces using classical methods, and not deep learning. There are however important parameters to choose.")
		
		font = cv2.FONT_HERSHEY_SIMPLEX
		cascPath = "/usr/local/lib/python3.7/site-packages/cv2/data/haarcascade_frontalface_default.xml"
		faceCascade = cv2.CascadeClassifier(cascPath)
		gray = cv2.imread('faces.jpeg', 0)

		st.markdown("*Original image:*")
		plt.figure(figsize=(12,8))
		plt.imshow(gray, cmap='gray')
		st.pyplot()

		scaleFactor = st.sidebar.slider("Scale Factor", 1.02, 1.15, 1.1, 0.01)
		minNeighbors = st.sidebar.slider("Number of neighblrs", 1, 15, 5, 1)
		minSize = st.sidebar.slider("Minimum size", 10, 200, 20, 1)
		
		# Detect faces
		faces = faceCascade.detectMultiScale(
		gray,
		scaleFactor=scaleFactor,
		minNeighbors=minNeighbors,
		flags=cv2.CASCADE_SCALE_IMAGE
		)

		# For each face
		for (x, y, w, h) in faces: 
		    # Draw rectangle around the face
		    if w > minSize:
		    	cv2.rectangle(gray, (x, y), (x+w, y+h), (0, 0, 0), 5)
		st.markdown("*Detected faces:*")
		plt.figure(figsize=(12,8))
		plt.imshow(gray, cmap='gray')
		st.pyplot()

	elif sub_topic == "Object Detection":

		st.title("Object Detection")

		st.write("Object Detection is a field which consists in identifying objects in an image or a video feed. This task involves convolutional neural networks (CNNs), a special type of deep learning architecture. The algorithm presented below is YOLO (You Only Look Once), a state-of-the-art algorithm trained to identify thousands of object types.")
		# This sidebar UI lets the user select parameters for the YOLO object detector.
		def object_detector_ui():
		    st.sidebar.markdown("# Model")
		    confidence_threshold = st.sidebar.slider("Confidence threshold", 0.0, 1.0, 0.5, 0.01)
		    return confidence_threshold #overlap_threshold

		# Draws an image with boxes overlayed to indicate the presence of cars, pedestrians etc.
		def draw_image_with_boxes(image, boxes):
		    LABEL_COLORS = [0, 255, 0]
		    image_with_boxes = image.astype(np.float64)
		    for _, (xmin, ymin, xmax, ymax) in boxes.iterrows():
		        image_with_boxes[int(ymin):int(ymax),int(xmin):int(xmax),:] += LABEL_COLORS
		        image_with_boxes[int(ymin):int(ymax),int(xmin):int(xmax),:] /= 2

		    st.image(image_with_boxes.astype(np.uint8), use_column_width=True)

		@st.cache(show_spinner=False)
		def load_present_image(img):
		    image = cv2.imread(img, cv2.IMREAD_COLOR)
		    image = image[:, :, [2, 1, 0]] # BGR -> RGB
		    return image

		def yolo_v3(image, confidence_threshold=0.5, overlap_threshold=0.3):
		    #@st.cache()allow_output_mutation=True
		    def load_network(config_path, weights_path):
		        net = cv2.dnn.readNetFromDarknet(config_path, weights_path)
		        output_layer_names = net.getLayerNames()
		        output_layer_names = [output_layer_names[i[0] - 1] for i in net.getUnconnectedOutLayers()]
		        return net, output_layer_names

		    net, output_layer_names = load_network("yolov3.cfg", "yolov3.weights")

		    blob = cv2.dnn.blobFromImage(image, 1 / 255.0, (416, 416), swapRB=True, crop=False)
		    net.setInput(blob)
		    layer_outputs = net.forward(output_layer_names)

		    boxes, confidences, class_IDs = [], [], []
		    H, W = image.shape[:2]

		    for output in layer_outputs:
		        for detection in output:
		            scores = detection[5:]
		            classID = np.argmax(scores)
		            confidence = scores[classID]
		            if confidence > confidence_threshold:
		                box = detection[0:4] * np.array([W, H, W, H])
		                centerX, centerY, width, height = box.astype("int")
		                x, y = int(centerX - (width / 2)), int(centerY - (height / 2))
		                boxes.append([x, y, int(width), int(height)])
		                confidences.append(float(confidence))
		                class_IDs.append(classID)

		    f = open("classes.txt", "r")
		    f = f.readlines()
		    f = [line.rstrip('\n') for line in list(f)]

		    try:
		    	st.subheader("Detected objects: " + ', '.join(list(set([f[obj] for obj in class_IDs]))))
		    except IndexError:
		    	st.write("Nothing detected")

		    indices = cv2.dnn.NMSBoxes(boxes, confidences, confidence_threshold, overlap_threshold)

		    xmin, xmax, ymin, ymax, labels = [], [], [], [], []
		    if len(indices) > 0:

		        for i in indices.flatten():

		            x, y, w, h = boxes[i][0], boxes[i][1], boxes[i][2], boxes[i][3]
		            xmin.append(x)
		            ymin.append(y)
		            xmax.append(x+w)
		            ymax.append(y+h)

		    boxes = pd.DataFrame({"xmin": xmin, "ymin": ymin, "xmax": xmax, "ymax": ymax})
		    return boxes[["xmin", "ymin", "xmax", "ymax"]]

		confidence_threshold = object_detector_ui()
		img_type = st.sidebar.selectbox("Select image type?", ['Cars', 'People', 'Animals', "Meeting"])

		if img_type == 'People':
		    image_url = "people.jpg"
		elif img_type == 'Cars':
		    image_url = "cars.jpg"
		elif img_type == 'Animals':
		    image_url = "animal.jpg"
		elif img_type == 'Meeting':
		    image_url = "meeting.jpg"

		image = load_present_image(image_url)

		# Get the boxes for the objects detected by YOLO by running the YOLO model.
		yolo_boxes = yolo_v3(image, confidence_threshold)
		draw_image_with_boxes(image, yolo_boxes)

	elif sub_topic == "Sentiment Detection":

		st.title("Sentiment Detection")

# III. Speech Processing

elif topic == "Speech Processing":
	sub_topic = st.sidebar.radio("Algorithm", ["Voice Activity Detection", "Speaker Identification", "Text to Speech", "Speech to Text"])

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

