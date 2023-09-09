import sys
from PIL import Image
import json
from pathlib import Path


if len(sys.argv) < 2:
  sys.exit("""Usage: python run.py path-to-image [path-to-image-2 ...]
Passing multiple images will optimize a single prompt across all passed images, useful for style transfer.
""")

config_path = "sample_config.json"

# image_paths = sys.argv[1]
dataset_path = sys.argv[1]
prompt_file_name = Path(sys.argv[2])

prompts = {}

if prompt_file_name.exists():
  with open(prompt_file_name, "r") as f:
    prompts = json.load(f)

dataset_name = dataset_path.split("/")[-1]

# with open('prompt_' + dataset_name + ".txt", "w") as f:
#   f.write(dataset_name + "\n")

for image_dir in sorted(Path(dataset_path).iterdir()):
  image_paths = list(image_dir.glob('*.png'))
  dirname = image_dir.name

  # load the target image
  images = [Image.open(image_path) for image_path in image_paths]

  # defer loading other stuff until we confirm the images loaded
  import argparse
  import open_clip
  from optim_utils import *

  print("Initializing...")

  # load args
  args = argparse.Namespace()
  args.__dict__.update(read_json(config_path))

  # You may modify the hyperparamters here
  args.print_new_best = True

  # load CLIP model
  device = "cuda" if torch.cuda.is_available() else "cpu"
  model, _, preprocess = open_clip.create_model_and_transforms(args.clip_model, pretrained=args.clip_pretrain, device=device)

  print(f"Running for {args.iter} steps.")
  if getattr(args, 'print_new_best', False) and args.print_step is not None:
    print(f"Intermediate results will be printed every {args.print_step} steps.")

  # optimize prompt
  learned_prompt = optimize_prompt(model, preprocess, args, device, target_images=images)
  print(f"Learned prompt for class {dirname}: {learned_prompt}")
  prompts[dirname] = learned_prompt

with open(prompt_file_name, "w") as f:
  json.dump(prompts, f)
