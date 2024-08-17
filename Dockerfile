# Specify the base image with platform
FROM --platform=linux/amd64 python:3.9-slim

# Set the working directory in the container
WORKDIR /app

# Copy the current directory contents into the container at /app
COPY . /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Install Python packages individually
RUN pip3 install torch torchvision --index-url https://download.pytorch.org/whl/cpu
RUN pip3 install transformers tqdm numpy scikit-learn scipy nltk sentencepiece

RUN pip install --no-cache-dir openai
RUN pip install --no-cache-dir unstructured
RUN pip install --no-cache-dir sentence-transformers
RUN pip install --no-cache-dir Flask
RUN pip install --no-cache-dir Flask-CORS

# Make port 5000 available to the world outside this container
EXPOSE 5000

# Define environment variable
ENV FLASK_APP=app.py

# Run app.py when the container launches
CMD ["python", "-m", "flask", "run", "--host", "0.0.0.0"]