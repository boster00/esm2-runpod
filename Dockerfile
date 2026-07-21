FROM pytorch/pytorch:2.3.1-cuda12.1-cudnn8-runtime
WORKDIR /app
RUN pip install --no-cache-dir runpod transformers
COPY handler.py .
CMD ["python", "-u", "handler.py"]
