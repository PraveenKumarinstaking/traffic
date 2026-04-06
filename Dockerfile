# Use an official Python runtime as a parent image
FROM python:3.10-slim

# Set the working directory in the container
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Copy the requirements file into the container
# Since we don't have many dependencies, we'll install them directly
RUN pip install --no-cache-dir pydantic numpy openai gradio fastapi uvicorn

# Setup a non-root user for Hugging Face Spaces (UID 1000)
RUN useradd -m -u 1000 user
USER user
ENV HOME=/home/user \
    PATH=/home/user/.local/bin:$PATH
WORKDIR $HOME/app

# Copy the current directory contents into the container at $HOME/app
# Ensure the user has ownership
COPY --chown=user . $HOME/app

# Expose the port the app runs on (for Gradio/Hugging Face)
EXPOSE 7860

# Define environment variables
ENV PYTHONUNBUFFERED=1

# Run the app with Uvicorn
CMD ["uvicorn", "server:app", "--host", "0.0.0.0", "--port", "7860"]
