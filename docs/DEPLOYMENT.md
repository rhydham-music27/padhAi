# Production Deployment Guide

## Table of Contents
1. Prerequisites
2. Server Requirements
3. Installation
4. Configuration
5. Running in Production
6. Monitoring and Logging
7. Backup and Recovery
8. Security Considerations
9. Scaling
10. Troubleshooting

## 1. Prerequisites
- Ubuntu 20.04+ or similar Linux
- Python 3.10/3.11
- NVIDIA GPU 16–24GB VRAM
- CUDA 12.4+ and cuDNN
- 32GB+ RAM, 100GB+ disk

## 2. Server Requirements
- Minimum (single user): 16GB VRAM, 16GB RAM, 4 CPU cores
- Recommended (multi-user): 24GB VRAM, 32GB RAM, 8 CPU cores

## 3. Installation
```bash
sudo apt update && sudo apt upgrade -y
sudo apt install -y python3.10 python3.10-venv python3-pip git
# Install CUDA following NVIDIA docs

git clone https://github.com/your-org/explainableAi.git
cd explainableAi
python3.10 -m venv .venv && source .venv/bin/activate
pip install torch==2.4.0 --index-url https://download.pytorch.org/whl/cu124
pip install -r requirements.txt
# Adapters
cp -r /path/to/trained/adapters models/adapters/final
```

## 4. Configuration
```bash
cp .env.example .env
```
Edit `.env`:
```bash
GRADIO_SERVER_PORT=7860
GRADIO_SERVER_NAME=0.0.0.0
GRADIO_SHARE=false
MODEL_NAME=mistralai/Mistral-7B-Instruct-v0.1
ADAPTER_PATH=./models/adapters/final
LOAD_IN_4BIT=true
VISION_MODEL_NAME=Salesforce/blip2-flan-t5-xl
VISION_MODEL_LOAD_IN_4BIT=true
MAX_SLIDES_CONTEXT=5
EXPLANATION_TEMPERATURE=0.7
QUIZ_TEMPERATURE=0.4
HF_TOKEN=your_token_here
```

## 5. Running in Production
Systemd unit `/etc/systemd/system/explainable-ai.service`:
```ini
[Unit]
Description=Explainable AI Tutor
After=network.target

[Service]
Type=simple
User=your-user
WorkingDirectory=/path/to/explainableAi
Environment="PATH=/path/to/explainableAi/.venv/bin"
ExecStart=/path/to/explainableAi/.venv/bin/python -m src.app --port 7860 --host 0.0.0.0
Restart=always
RestartSec=10

[Install]
WantedBy=multi-user.target
```
Enable/start:
```bash
sudo systemctl daemon-reload
sudo systemctl enable explainable-ai
sudo systemctl start explainable-ai
sudo systemctl status explainable-ai
```

Docker alternative:
```dockerfile
FROM nvidia/cuda:12.4.0-runtime-ubuntu22.04
RUN apt-get update && apt-get install -y python3.10 python3-pip
WORKDIR /app
COPY . /app
RUN pip install -r requirements.txt
EXPOSE 7860
CMD ["python", "-m", "src.app"]
```

Nginx reverse proxy (HTTPS):
```nginx
server {
    listen 80;
    server_name your-domain.com;
    location / {
        proxy_pass http://localhost:7860;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;
        proxy_http_version 1.1;
        proxy_set_header Upgrade $http_upgrade;
        proxy_set_header Connection "upgrade";
    }
}
```

## 6. Monitoring and Logging
- `journalctl -u explainable-ai -f`
- `watch -n 1 nvidia-smi`
- Add file logging in app if needed

## 7. Backup and Recovery
Backup:
```bash
mkdir -p /backups/explainable-ai
DATE=$(date +%Y%m%d)
tar -czf /backups/explainable-ai/models-$DATE.tar.gz models/adapters/final/
cp .env /backups/explainable-ai/.env-$DATE
```
Recovery:
```bash
# restore and restart service
```

## 8. Security Considerations
- Add Gradio auth in `src/app.py` launch
- Always use HTTPS
- Set upload limits, rate limiting
- Enable firewall (ufw)

## 9. Scaling
- Horizontal: multiple instances + LB, shared models
- Vertical: larger GPU, more RAM
- Queue: adjust concurrency limit in app

## 10. Troubleshooting
See `docs/TROUBLESHOOTING.md`.
