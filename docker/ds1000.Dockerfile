FROM python:3.11-slim

RUN pip install --no-cache-dir \
    pandas \
    numpy \
    scipy \
    matplotlib \
    scikit-learn
