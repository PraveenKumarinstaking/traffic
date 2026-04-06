# Use an official Python runtime as a parent image
FROM python:3.10-slim

# Set the working directory in the container
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Setup a non-root user for Hugging Face Spaces (UID 1000)
RUN useradd -m -u 1000 user
ENV HOME=/home/user \
	PATH=/home/user/.local/bin:$PATH

WORKDIR $HOME/app

# Copy the requirements and lock files first for caching
COPY --chown=user requirements.txt pyproject.toml uv.lock ./
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the application code
COPY --chown=user . $HOME/app

# Change ownership and install the project as a package
RUN pip install --no-cache-dir -e .

# Expose the default Gradio/FastAPI port
EXPOSE 7860

# Define environment variables
ENV PYTHONUNBUFFERED=1

# Start the application using the entry point defined in pyproject.toml
CMD ["server"]
