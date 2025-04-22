# Use an official Python image.
FROM mambaorg/micromamba:2.0.3

# Set environment variables to avoid buffering and specify the app port.
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PORT=8000

# Set the name working directory.
WORKDIR /raft-knowledge-base-website

# Copy the application code.
USER root
COPY . .

# Install Python environment and dependencies
RUN micromamba install -y -n base -f server_env.yaml && \
    micromamba clean --all --yes


# Expose the port Gunicorn will run on.
EXPOSE $PORT

# Start Gunicorn.
CMD ["gunicorn", "-w", "1", "-b", "0.0.0.0:8000", "wsgi:app"]
