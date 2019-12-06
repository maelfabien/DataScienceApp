FROM python:3.8
EXPOSE 8501
WORKDIR /app
COPY . .
RUN apt-get update && \
    apt-get install -y --no-install-recommends gcc && \
    python setup.py install && \
    apt-get remove -y gcc && apt-get -y autoremove
RUN pip install -r requirements.txt
CMD 
CMD streamlit run app.py