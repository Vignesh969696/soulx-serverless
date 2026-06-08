FROM runpod/pytorch:1.0.2-cu1281-torch280-ubuntu2404

WORKDIR /app

COPY dual-mode-worker/requirements.txt /tmp/worker_requirements.txt
RUN pip install --break-system-packages --no-cache-dir -r /tmp/worker_requirements.txt

COPY . .

CMD ["python", "-u", "dual-mode-worker/handler.py"]
