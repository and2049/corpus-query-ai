from transformers import AutoProcessor, LlavaForConditionalGeneration
import torch
from PIL import Image
import os

local_model_path = "models/llava-1.5-7b-hf"
hub_model_id = "llava-hf/llava-1.5-7b-hf"

if os.path.isdir(local_model_path):
    model_source = local_model_path
    print(f"Found local model at: {model_source}")
else:
    model_source = hub_model_id
    print(f"Local model not found. Downloading from Hugging Face Hub: {model_source}")

device = "cuda" if torch.cuda.is_available() else "cpu"

model = LlavaForConditionalGeneration.from_pretrained(
    model_source,
    torch_dtype=torch.float16,
    low_cpu_mem_usage=True,
).to(device)
processor = AutoProcessor.from_pretrained(model_source)

image_file = "papers/sample_figure.png"
raw_image = Image.open(image_file).convert("RGB")

prompt = "USER: <image>\nWhat is this graph showing? ASSISTANT:"
inputs = processor(prompt, images=raw_image, return_tensors="pt").to(device)

output = model.generate(**inputs, max_new_tokens=200)
description = processor.decode(output[0][2:], skip_special_tokens=True)

print("\n--- VLM Description ---")
print(description)