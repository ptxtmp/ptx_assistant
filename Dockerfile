FROM python:3.10-slim

# Set the working dir
WORKDIR /app

# List the contents of the working dir
RUN ls -al

# Copy the files from project dir to working dir
COPY ./assistant /app/

# List the contents of the working dir
RUN ls -al

# Update the docker container aptitude
RUN apt-get update -y

# Install the necessary Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Move into the app directory
RUN cd /app/

# Expose the app port
EXPOSE 80

# Run the app
CMD ["chainlit", "run", "app.py", "-h", "--port", "80"]
