FROM python:3.11-slim

RUN pip install --no-cache-dir numpy opencv-python-headless Pillow

WORKDIR /app
COPY solution.py .

CMD ["python", "solution.py"]
