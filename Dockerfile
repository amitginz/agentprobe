FROM python:3.12-slim
WORKDIR /app
COPY pyproject.toml README.md ./
COPY src ./src
RUN pip install --no-cache-dir ".[postgres]"
EXPOSE 8000
CMD ["uvicorn", "agentprobe.api:app_factory", "--factory", "--host", "0.0.0.0", "--port", "8000"]
