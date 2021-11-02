# Use python as base image
FROM python:3.6-stretch

# Use working directory /app/model
WORKDIR /app/model

# Copy and install required packages
COPY requirements.txt .
RUN pip install --trusted-host pypi.python.org -r requirements.txt

# Copy all the content of current directory to working directory
COPY ./saved_weights.pt ./
COPY ./main.py ./

# Set env variables for Cloud Run
#ENV PORT 5000
#ENV HOST 0.0.0.0

# Open port 5000
EXPOSE 5000:5000

# Run flask app
CMD ["python","main.py"]