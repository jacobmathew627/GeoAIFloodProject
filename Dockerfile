# Use the official Python Streamlit image
FROM python:3.10-slim

WORKDIR /app

# Install git and git-lfs for Hugging Face
RUN apt-get update && apt-get install -y \
    build-essential \
    software-properties-common \
    git \
    git-lfs \
    && rm -rf /var/lib/apt/lists/*
RUN git lfs install

# Copy requirements and install
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy all the massive GeoAI_New files and code
COPY . .

# Expose Streamlit port
EXPOSE 8501

# Command to run the app
CMD ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0"]
