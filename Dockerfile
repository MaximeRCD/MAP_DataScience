FROM ubuntu:latest

RUN mkdir /app
COPY ./modules/ /app/modules/
COPY ./requirements.txt /app
COPY ./main.py /app
WORKDIR /app

RUN apt-get -y update && \
    apt-get install -y python3-pip

RUN pip install -r requirements.txt

ENV AWS_S3_ENDPOINT="minio.lab.sspcloud.fr"

CMD ["uvicorn", "main:app", "--reload", "--host", "0.0.0.0", "--port", "8000"]
