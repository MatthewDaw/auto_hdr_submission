FROM python:3.11-slim

RUN pip install --no-cache-dir \
    numpy \
    opencv-python-headless \
    Pillow \
    scipy \
    scikit-learn \
    PyWavelets

WORKDIR /app
COPY solution.py .
COPY autohdr/ ./autohdr/

CMD ["python", "solution.py"]
