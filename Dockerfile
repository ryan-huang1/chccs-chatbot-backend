# Use an official Python runtime as a base image
FROM python:3.10

# Set the working directory in the container
WORKDIR /app

# Copy the current directory contents into the container at /app
COPY . /app

# Install Pillow first
RUN pip install --no-cache-dir pillow

# Install Python packages individually
RUN pip3 install torch torchvision --index-url https://download.pytorch.org/whl/cpu
RUN pip3 install transformers tqdm numpy scikit-learn scipy nltk sentencepiece

RUN pip install --no-cache-dir openai
RUN pip install --no-cache-dir pdfminer.six
RUN pip install --no-cache-dir pillow-heif
RUN pip install --no-cache-dir unstructured
RUN pip install --no-cache-dir sentence-transformers
RUN pip install --no-cache-dir Flask
RUN pip install --no-cache-dir Flask-CORS

# Set environment variables
ENV OPENAI_API_KEY=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

# Make port 80 available to the world outside this container
EXPOSE 80

CMD python3 main.py