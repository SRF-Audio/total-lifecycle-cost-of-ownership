FROM python:latest

# Install minimal deps
RUN pip install --no-cache-dir pandas

# App code
WORKDIR /app
COPY src/ src/
COPY app/ app/

# Run the CLI: reads /work/cars.json, writes to /work/data (mounted via compose)
ENV PYTHONUNBUFFERED=1
ENTRYPOINT ["python", "app/main.py", "--cars", "/work/cars.json", "--outdir", "/work/data"]
