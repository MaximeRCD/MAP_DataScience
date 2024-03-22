FROM python:3.10

RUN mkdir /app
COPY ./requirements.txt /app

WORKDIR /app

RUN pip install -r requirements.txt

COPY ./modules/ /app/modules/
COPY ./main.py /app
COPY ./.env /app


CMD ["uvicorn", "main:app", "--reload", "--host", "0.0.0.0", "--port", "8000"]
