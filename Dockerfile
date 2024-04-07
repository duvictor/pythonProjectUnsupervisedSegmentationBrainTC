FROM python:3.7
LABEL maintainer='Paulo Victor - Unsupervised Segmentation Brain'
RUN echo "Starting coping files"
COPY . /app
WORKDIR /app
RUN echo "Starting install requirements"

RUN apt-get update && apt-get install ffmpeg libsm6 libxext6  -y


RUN pip install --upgrade pip \
  && pip install --no-cache-dir -r requirements.txt
RUN echo "Starting run train and validation"
CMD ['ls']
CMD ["python","main.py"]