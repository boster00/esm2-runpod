FROM pytorch/pytorch:2.3.1-cuda12.1-cudnn8-runtime
WORKDIR /app
RUN pip install --no-cache-dir runpod transformers scikit-learn numpy
COPY handler.py .
COPY data.csv .
# Pre-download ESM-2 650M model weights so workers start immediately (no download latency)
RUN python3 -c "\
from transformers import AutoTokenizer, AutoModel; \
tok = AutoTokenizer.from_pretrained('facebook/esm2_t33_650M_UR50D'); \
m = AutoModel.from_pretrained('facebook/esm2_t33_650M_UR50D'); \
print('ESM-2 650M cached:', tok.__class__.__name__); \
"
CMD ["python", "-u", "handler.py"]
