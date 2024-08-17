# Use an official Python runtime as a base image
FROM python:3.10

# Set the working directory in the container
WORKDIR /app

# Copy the current directory contents into the container at /app
COPY . /app

# Install any needed packages specified in requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Set environment variables
ENV OPENAI_API_KEY=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

# Make port 80 available to the world outside this container
EXPOSE 80

CMD python3 main.py