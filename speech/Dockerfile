FROM python:3.12.3-alpine

RUN pip install --upgrade pip

RUN mkdir /soundfiles

WORKDIR /app

COPY . .
RUN pip install -r /app/requirements.txt

ENV PYTHONPATH="${PYTHONPATH}:/app"
