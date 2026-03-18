# Setup Guide

## Done So Far
- `src/pipeline.py` — loads SDXL + ControlNet OpenPose + 2 LoRAs (Hermione & Daenerys)
- `src/generate_pose.py` — generates multi-person pose skeletons for ControlNet
- `src/extract_reference.py` — generates reference face images from each LoRA
- `configs/identities.yaml` — identity config (trigger words, paths)
- `data/pose_images/two_person_pose.png` — synthetic 2-person pose (ready)

## To Do
- Get LoRA files from shared Drive → place in `data/loras/`
- Generate or add reference face images (see below)
- Implement `src/scorer.py`, `src/adaptive_loop.py`, `src/sweep.py`, `src/evaluate.py`

## Reference Face Images
Place one image per identity in `data/reference_faces/`:
- `hermione.png`
- `daenerys.png`

**Good reference image:**
- Single person, face clearly visible
- Frontal or slight angle (max ~30°)
- Realistic style, 512x512+ resolution
- Matches what the LoRA produces (movie stills work)

**Bad reference image:**
- Group photos, side profiles, sunglasses, heavy shadows, blurry, cartoon/anime style

Or just run `python src/extract_reference.py` on a GPU to auto-generate them from the LoRAs.
