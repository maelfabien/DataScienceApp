import streamlit as st
import matplotlib.pyplot as plt
import cv2
import numpy as np
import pandas as pd

def add_yolo():

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

	    net, output_layer_names = load_network("yolov3/yolov3.cfg", "yolov3.weights")

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

	    f = open("yolov3/classes.txt", "r")
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
	    image_url = "images/Group.jpg"
	elif img_type == 'Cars':
	    image_url = "images/cars.jpg"
	elif img_type == 'Animals':
	    image_url = "images/animal.jpg"
	elif img_type == 'Meeting':
	    image_url = "images/Men.jpg"

	image = load_present_image(image_url)

	# Get the boxes for the objects detected by YOLO by running the YOLO model.
	yolo_boxes = yolo_v3(image, confidence_threshold)
	draw_image_with_boxes(image, yolo_boxes)
