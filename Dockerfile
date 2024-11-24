# Use an official Python image.
FROM mambaorg/micromamba:2.0.3

# Set environment variables to avoid buffering and specify the app port.
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PORT=8000

# Set the working directory.
WORKDIR /raft_kb_server

# Copy the application code.
COPY .. .

# Install Python environment and dependencies
RUN micromamba install -y -n base -f server_env.yaml && \
    micromamba clean --all --yes


# Expose the port Gunicorn will run on.
EXPOSE $PORT

# Start Gunicorn.
CMD ["gunicorn", "-w", "4", "-b", "0.0.0.0:8000", "wsgi:app"]

# Build the image in the console with the tag raftkb
#   docker build -t raftkb .

# Run the container using the image
#   docker run -p 8000:8000 raftkb
