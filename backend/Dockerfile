FROM python:3.12.3-alpine

RUN pip install --upgrade pip

WORKDIR /app

COPY . .
RUN pip install -r /app/requirements.txt

ENV PYTHONPATH="${PYTHONPATH}:/app"
