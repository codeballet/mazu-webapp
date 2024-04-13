FROM tensorflow/tensorflow:2.10.1-gpu

RUN pip install --upgrade pip

WORKDIR /app

COPY ./requirements.txt /app
RUN pip install -r /app/requirements.txt

COPY /notebooks/. /app/notebooks

ENV PYTHONPATH="${PYTHONPATH}:/app"