# Stage 1: Build
FROM python:3.10-slim as builder
WORKDIR /app

# Install dependencies into /root/.local (copied into runtime image)
COPY requirements.txt .
RUN pip install --no-cache-dir --user -r requirements.txt

# Stage 2: Runtime
FROM python:3.10-slim
WORKDIR /app

# Bring in site packages from builder
COPY --from=builder /root/.local /root/.local
ENV PATH=/root/.local/bin:$PATH

# App code
COPY src/ src/
COPY pyproject.toml .

# Install the package without re-installing dependencies
RUN pip install --no-cache-dir --no-deps -e .

EXPOSE 8000
CMD ["uvicorn", "cv200.api:app", "--host", "0.0.0.0", "--port", "8000"]








