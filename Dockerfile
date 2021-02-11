FROM python:3.8
LABEL maintainer="Bryan Patrick Wood <bpw1621@gmail.com>"

WORKDIR /usr/src/app
COPY .. .
RUN pip install -U pip && pip install --no-cache-dir -e .
EXPOSE 8501
ENTRYPOINT streamlit run ./streamlit_topic_modeling/app.py