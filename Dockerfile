# syntax=docker/dockerfile:1

FROM python:3.10-slim-buster

# Install the necessary system dependencies
RUN apt-get update && apt-get install -y libgl1-mesa-glx

WORKDIR /app

COPY requirements.txt requirements.txt

RUN pip3 install -r requirements.txt

COPY . . 

CMD ["gunicorn", "wsgi:app"]

EXPOSE 3000
