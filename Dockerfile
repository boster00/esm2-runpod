FROM pytorch/pytorch:2.3.1-cuda12.1-cudnn8-runtime
# Ensure conda python/pip is used
ENV PATH=/opt/conda/bin:$PATH
WORKDIR /app
RUN pip install --no-cache-dir runpod transformers scikit-learn numpy
COPY handler.py .
COPY data.csv .
# Pre-download ESM-2 650M model weights so workers start immediately (no cold-start latency)
RUN python3 -c "\
from transformers import AutoTokenizer, AutoModel; \
tok = AutoTokenizer.from_pretrained('facebook/esm2_t33_650M_UR50D'); \
m = AutoModel.from_pretrained('facebook/esm2_t33_650M_UR50D'); \
print('ESM-2 650M cached OK, hidden_size:', m.config.hidden_size); \
"
CMD ["python", "-u", "handler.py"]
