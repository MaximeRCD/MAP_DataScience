FROM ubuntu:latest

RUN mkdir /app
COPY . /app
WORKDIR /app

RUN apt-get -y update && \
    apt-get install -y python3-pip

RUN pip install -r requirements.txt
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "5000"]
