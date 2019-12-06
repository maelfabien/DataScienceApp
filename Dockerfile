FROM ubuntu:latest
EXPOSE 8501
WORKDIR /app
COPY . .
RUN sudo pip install -r requirements.txt
CMD 
CMD streamlit run app.py