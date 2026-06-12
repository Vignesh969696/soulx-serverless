FROM runpod/pytorch:1.0.2-cu1281-torch280-ubuntu2404

WORKDIR /app

COPY requirements_no_xformers.txt .
COPY docker-entrypoint.sh .

RUN pip install --break-system-packages --no-cache-dir \
    xfuser==0.4.5 --no-deps

RUN pip install --break-system-packages --no-cache-dir \
    torch==2.7.1+cu128 \
    torchvision==0.22.1+cu128 \
    torchaudio==2.7.1+cu128 \
    --index-url https://download.pytorch.org/whl/cu128

RUN pip install --break-system-packages --no-cache-dir \
    -r requirements_no_xformers.txt

RUN pip install --break-system-packages --no-cache-dir \
    xformers==0.0.31

RUN pip install --break-system-packages --no-cache-dir \
    hf_transfer

RUN chmod +x docker-entrypoint.sh

COPY . .

EXPOSE 8000

CMD ["./docker-entrypoint.sh"]
