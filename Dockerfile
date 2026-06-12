FROM runpod/pytorch:1.0.2-cu1281-torch280-ubuntu2404

WORKDIR /workspace/soulx-serverless

COPY . .

RUN pip install --break-system-packages --no-cache-dir \
    -r requirements.txt

RUN pip install --break-system-packages --no-cache-dir \
    runpod

RUN pip install --break-system-packages --no-cache-dir \
    hf_transfer

RUN chmod +x docker-entrypoint.sh

EXPOSE 8000

CMD ["./docker-entrypoint.sh"]
