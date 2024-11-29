# Base image with CUDA 11.8 and pre-installed Python 3.9
FROM nvidia/cuda:11.8.0-devel-ubuntu20.04

# Set the working directory inside the container
WORKDIR /app

# Install necessary system dependencies
RUN apt-get update && apt-get install -y \
    ffmpeg libsm6 libxext6 libfontconfig1 libxrender1 libgl1-mesa-glx \
    python3-tk \
    bash \
    curl \
    && rm -rf /var/lib/apt/lists/*

RUN pip install opencv-python-headless

# Copy the application code and required files to the container
COPY . /app

# Install Python dependencies from requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Make the necessary bash scripts executable
RUN chmod +x env_install.sh scripts/test_xportrait.sh

# Run environment setup
RUN bash env_install.sh

# Expose the port FastAPI will run on
EXPOSE 8000

# Command to run the FastAPI server
CMD ["uvicorn", "index:app", "--host", "0.0.0.0", "--port", "8000"]
