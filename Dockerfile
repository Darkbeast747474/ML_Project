# Use an official Python runtime as a parent image
FROM python:3.8-slim-buster

# Set the working directory in the container
WORKDIR /app

# Copy the current directory contents into the container at /app
COPY . /app

# Update and install necessary packages
RUN apt-get update -y && apt install awscli -y

# Print the contents of requirements.txt for debugging purposes
RUN cat requirements.txt

# Install any needed packages specified in requirements.txt

RUN pip install --upgrade pip --default-timeout=200 && pip install -r requirements.txt --no-cache-dir

# Make port 80 available to the world outside this container
EXPOSE 80

# Define environment variable
ENV NAME World

# Run application when the container launches
CMD ["python3", "application.py"]
