FROM python:3.12-slim-bookworm

RUN apt-get -y update
RUN apt-get install -y ffmpeg

RUN pip install --upgrade pip

RUN mkdir /soundfiles

WORKDIR /app

COPY . .

RUN pip install -r /app/requirements.txt

ENV PYTHONPATH="${PYTHONPATH}:/app"