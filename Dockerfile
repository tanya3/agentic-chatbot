# Dockerfile

FROM python:3.11-slim as base

ENV PYTHONUNBUFFERED 1

ENV PIP_NO_CACHE_DIR 1

WORKDIR /app

ENV HF_HOME=/app/.cache/huggingface
ENV SENTENCE_TRANSFORMERS_HOME=/app/.cache/huggingface/sentence_transformers
RUN mkdir -p ${HF_HOME} && chmod -R 777 ${HF_HOME}

COPY requirements.txt .

RUN pip install --upgrade pip && \
    pip install -r requirements.txt

COPY ./app /app/app
COPY ./configs /app/configs
COPY ./assets /app/assets

EXPOSE 8501

CMD ["streamlit", "run", "app/ui/streamlit_app.py", "--server.port=8501", "--server.address=0.0.0.0", "--server.fileWatcherType", "none", "--browser.serverAddress=localhost"]