# Use an official Python runtime as a parent image
FROM python:3.9

# Set the working directory inside the container
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y espeak

# Copy the requirements file
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir --upgrade -r requirements.txt

# Ensure /tmp/flagged directory exists and is writable
RUN mkdir -p /tmp/flagged && chmod 777 /tmp/flagged

# Copy the application code
COPY . .

# Expose the port FastAPI runs on
EXPOSE 8000

# Command to run FastAPI app with Uvicorn
CMD ["uvicorn", "new:app", "--host", "0.0.0.0", "--port", "8000", "--reload"]
