# Base image with Python 3.9 and CUDA 11.8
FROM nvidia/cuda:11.8.0-devel-ubuntu20.04

# Set the working directory inside the container
WORKDIR /app

# Install system dependencies for Python and tkinter
RUN apt-get update && apt-get install -y \
    python3.9 \
    python3.9-venv \
    python3.9-dev \
    python3-pip \
    python3-tk \
    bash \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Set Python 3.9 as the default
RUN update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.9 1
RUN update-alternatives --install /usr/bin/pip3 pip3 /usr/bin/pip3 1

# Copy the application code and required files to the container
COPY . /app

# Install Python dependencies from requirements.txt
RUN pip3 install --no-cache-dir -r requirements.txt

# Make the necessary bash scripts executable
RUN chmod +x env_install.sh scripts/test_xportrait.sh

# Run environment setup
RUN bash env_install.sh

# Expose the port FastAPI will run on
EXPOSE 8000

# Command to run the FastAPI server
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
