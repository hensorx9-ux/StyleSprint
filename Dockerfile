# Use an official NVIDIA CUDA runtime as a base image for GPU support
FROM python:3.10-slim

# Set the working directory in the container to /app
WORKDIR /app

# Copy dependency definition first
COPY requirements.txt .

# Install dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy remaining project files
COPY . .

# Make port 80 available to the world outside this container
EXPOSE 80

# Run app.py when the container launches
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "80"]