FROM python:3.11-slim

RUN pip install --no-cache-dir \
    pandas \
    numpy \
    scipy \
    matplotlib \
    scikit-learn \
    Pillow

# CPU-only PyTorch to keep image size manageable (~200MB vs ~2GB)
RUN pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu

# TensorFlow (~500MB). Remove this line to skip 45 TF tasks and shrink the image.
RUN pip install --no-cache-dir tensorflow
