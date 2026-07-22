FROM pytorch/pytorch:2.3.1-cuda12.1-cudnn8-runtime
# Ensure conda python/pip is used
ENV PATH=/opt/conda/bin:$PATH
WORKDIR /app
# Pin runpod to known-good version; use fair-esm (proven to work in conda pytorch env)
RUN pip install --no-cache-dir "runpod>=1.7,<2.0" fair-esm scikit-learn numpy
COPY handler.py .
COPY run_proteome_plm.py .
COPY entrypoint.sh .
COPY data.csv .
RUN chmod +x entrypoint.sh
# Pre-download ESM-2 650M model weights via fair-esm
RUN python3 -c "\
import esm; \
m, alpha = esm.pretrained.esm2_t33_650M_UR50D(); \
print('ESM-2 650M cached OK via fair-esm'); \
"
CMD ["bash", "entrypoint.sh"]
