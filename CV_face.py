import streamlit as st
import matplotlib.pyplot as plt
import cv2

def add_face():

	st.title("Face Detection")
	st.write("Face detection is a central algorithm in computer vision. The algorithm implemented below is a Haar Cascade Classifier. It detects several faces using classical methods, and not deep learning. There are however important parameters to choose.")
	
	font = cv2.FONT_HERSHEY_SIMPLEX
	cascPath = "facedetect/haarcascade_frontalface_default.xml"
	faceCascade = cv2.CascadeClassifier(cascPath)
	gray = cv2.imread('images/Women.jpg', 0)

	st.markdown("*Original image:*")
	plt.figure(figsize=(12,8))
	plt.imshow(gray, cmap='gray')
	st.pyplot()

	scaleFactor = st.sidebar.slider("Scale Factor", 1.02, 1.15, 1.1, 0.01)
	minNeighbors = st.sidebar.slider("Number of neighbors", 1, 15, 5, 1)
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