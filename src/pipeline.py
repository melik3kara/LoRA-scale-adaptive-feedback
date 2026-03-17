# src/pipeline.py
import yaml
import torch
from diffusers import StableDiffusionXLControlNetPipeline, ControlNetModel

def load_identities(config_path="configs/identities.yaml"):
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    return config["identities"]

def build_pipeline(identities):
    # Load ControlNet
    controlnet = ControlNetModel.from_pretrained(
        "thibaud/controlnet-openpose-sdxl-1.0",
        torch_dtype=torch.float16
    )

    # Load base SDXL pipeline
    pipe = StableDiffusionXLControlNetPipeline.from_pretrained(
        "stabilityai/stable-diffusion-xl-base-1.0",
        controlnet=controlnet,
        torch_dtype=torch.float16
    ).to("cuda")

    # Load each LoRA by name
    for name, identity in identities.items():
        pipe.load_lora_weights(
            identity["lora_path"],
            adapter_name=name
        )

    return pipe