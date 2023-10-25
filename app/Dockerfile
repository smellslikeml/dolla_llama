FROM ubuntu:20.04

# Dependencies
RUN apt-get update && apt-get install -y \
    git \
    g++ \
    make \
    curl \
    python3 \
    python3-pip \
    bash


# Clone whisper.cpp
RUN git clone https://github.com/ggerganov/whisper.cpp /whisper
WORKDIR /whisper

# Download the Whisper model
RUN bash ./models/download-ggml-model.sh base.en

# Build the main example
RUN make

# Add and run the Gradio app
WORKDIR /app

COPY requirements.txt /app
RUN pip install -r requirements.txt

COPY app.py /app
COPY entrypoint.sh /app/entrypoint.sh
RUN chmod +x /app/entrypoint.sh
ENTRYPOINT ["./entrypoint.sh"]
