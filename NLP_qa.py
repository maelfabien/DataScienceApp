import streamlit as st
from allennlp import pretrained
import matplotlib.pyplot as plt

def add_qa():

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