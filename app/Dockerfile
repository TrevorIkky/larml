FROM ubuntu:bionic

ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update -y && apt-get install -y build-essential cmake \
    libsm6 libxext6 libxrender-dev \
    python3 python3-pip python3-dev python3-opencv

COPY ./requirements.txt ./requirements.txt
WORKDIR /
RUN pip3 install --upgrade pip && pip3 install -r requirements.txt

COPY . .
EXPOSE 5000
CMD ["gunicorn", "--workers=4",  "--bind=0.0.0.0:5000", "wsgi:app"]
