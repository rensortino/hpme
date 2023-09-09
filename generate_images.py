from diffusers import DPMSolverMultistepScheduler, StableDiffusionPipeline
import torch
from pathlib import Path
from torchvision.utils import save_image
import json


model_id = "stabilityai/stable-diffusion-2-1-base"
scheduler = DPMSolverMultistepScheduler.from_pretrained(model_id, subfolder="scheduler")
device = "cuda" if torch.cuda.is_available() else "cpu"
out_dir = Path("generated")
pipe = StableDiffusionPipeline.from_pretrained(
    model_id,
    scheduler=scheduler,
    torch_dtype=torch.float16,
    revision="fp16",
    )
pipe = pipe.to(device)

image_length = 512

with open('prompts/tinyimagenet.json') as f:
    learned_prompts = json.load(f)

with open('data/tinyimagenet/class_mapping.json') as f:
    class_mapping = json.load(f)

for class_id, learned_prompt in learned_prompts.items():

    prompt = f"a quality photo of {learned_prompt}"

    num_images = 1
    guidance_scale = 9
    num_inference_steps = 25

    class_name = f"{class_id}_{class_mapping[class_id]}"
    save_dir = out_dir / class_name
    save_dir.mkdir(exist_ok=True, parents=True)

    for i in range(500):

        images = pipe(
            prompt,
            num_images_per_prompt=num_images,
            guidance_scale=guidance_scale,
            num_inference_steps=num_inference_steps,
            height=image_length,
            width=image_length,
            ).images
        
        save_path = save_dir / f"{i}.png"
        # save_image(images[0], save_dir)
        images[0].save(save_path)