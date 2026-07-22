FROM pytorch/pytorch:2.3.1-cuda12.1-cudnn8-runtime
# Ensure conda python/pip is used
ENV PATH=/opt/conda/bin:$PATH
WORKDIR /app
RUN pip install --no-cache-dir runpod transformers scikit-learn numpy
COPY handler.py .
COPY data.csv .
# Pre-download ESM-2 650M model weights using huggingface_hub (no GPU needed at build time)
# snapshot_download fetches the raw files into HF cache without instantiating the model,
# so it works even though GitHub Actions runners have no GPU / CUDA.
RUN python -c "from huggingface_hub import snapshot_download; snapshot_download(repo_id='facebook/esm2_t33_650M_UR50D'); print('ESM-2 650M weights cached OK')"
CMD ["python", "-u", "handler.py"]
