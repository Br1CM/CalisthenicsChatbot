FROM python:3.11.9-slim

# Install system libs (as before)
RUN apt-get update && apt-get install -y \
    build-essential pkg-config libffi-dev libhdf5-dev \
    libjpeg-dev zlib1g-dev libgl1-mesa-glx libglib2.0-0 \
    libsm6 libxext6 libgomp1 ffmpeg \
  && rm -rf /var/lib/apt/lists/*

WORKDIR /app

RUN python -m pip install --no-cache-dir pip==25.1.1
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY chatbot_functions.py excercise_classification.py webapp.py ./
COPY Images/ ./Images/
COPY Models/ ./Models/
COPY Videos/ ./Videos/

EXPOSE 8501

CMD ["streamlit", "run", "webapp.py", "--server.enableXsrfProtection=false", "--server.port=8501", "--server.address=0.0.0.0"]
