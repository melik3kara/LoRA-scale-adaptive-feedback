# RunPod Setup

## 1. Pod ayarları
- Template: `runpod/pytorch:2.4.0-py3.11-cuda12.4.1`
- GPU: RTX 3090 / A40 / A5000 (min 16 GB VRAM)
- Disk: 50 GB
- Start Jupyter Lab

## 2. Repo'yu klonla
Terminal'de (Jupyter → File → New → Terminal):
```bash
cd /workspace
git clone https://github.com/YOUR_USERNAME/LoRA-scale-adaptive-feedback.git
cd LoRA-scale-adaptive-feedback
```

## 3. LoRA dosyalarını yükle
Google Drive'daki LoRA'ları indir:
```bash
mkdir -p data/loras

# Google Drive'dan indirmek için (gdown):
pip install gdown
gdown "https://drive.google.com/uc?id=HERMIONE_FILE_ID" -O data/loras/hermonie.safetensors
gdown "https://drive.google.com/uc?id=DAENERYS_FILE_ID" -O data/loras/daenerys.safetensors
```

Ya da Jupyter'in sol panelinden sürükle-bırak:
- `data/loras/hermonie.safetensors`
- `data/loras/daenerys.safetensors`

## 4. Reference ve pose dosyaları
Repo'da zaten var olmalı (`data/reference_faces/`, `data/pose_images/`).  
Eğer yoksa, Drive'dan aynı şekilde indir.

## 5. Notebook'u aç ve çalıştır
`notebooks/adaptive_loop.ipynb` → cell'leri sırayla çalıştır.

## Notlar
- İlk çalıştırmada SDXL (~7 GB) + ControlNet (~1.5 GB) HF'den indirilir — 5–10 dk sürer
- Sonraki runlarda `/workspace/cache/huggingface` kullanılır → hızlı
- Pod kapanınca `/workspace` dışı silinir, ama `/workspace` kalır
