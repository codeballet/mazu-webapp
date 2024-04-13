FROM tensorflow/tensorflow:2.10.1-gpu

RUN pip install --upgrade pip

WORKDIR /app

COPY . .
RUN pip install -r /app/requirements.txt

ENV PYTHONPATH="${PYTHONPATH}:/app"
