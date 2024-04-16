FROM python:3.12-slim

RUN pip install --upgrade pip

RUN apt-get update && apt-get -y mpg123

WORKDIR /app

COPY . .
RUN pip install -r /app/requirements.txt

ENV PYTHONPATH="${PYTHONPATH}:/app"