from tensorflow/tensorflow:2.10.0-gpu

RUN apt update && \
    apt install -y git && \
    apt install -y python3.8-venv

WORKDIR /app

# copy the requirements file into the image
COPY . .
RUN pip install --requirement requirements.txt

CMD ["sh", "./scripts/train.sh"]