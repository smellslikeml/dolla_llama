# Using an official Python runtime as a parent image
FROM python:3.9-slim-buster

# Set environment variables for non-interactive (silent) installation
ENV DEBIAN_FRONTEND=noninteractive

# System dependencies
RUN apt-get update && \
    apt-get install -y --no-install-recommends gcc libpq-dev && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

# Setting working directory in container
WORKDIR /app

# Copying requirements.txt and config.json to the container
COPY requirements.txt /app

# Installing Python dependencies
RUN pip install -r requirements.txt

COPY config.json /app
COPY main.py /app

# This command will not actually run when using Docker Compose to start the services
CMD ["python", "main.py"]

