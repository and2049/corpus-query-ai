from transformers import AutoProcessor, LlavaForConditionalGeneration
import torch
from PIL import Image
import requests

device = "cuda" if torch.cuda.is_available() else "cpu"

# Load the LLaVA model and processor
model_id = "llava-hf/llava-1.5-7b-hf"
model = LlavaForConditionalGeneration.from_pretrained(
    model_id,
    torch_dtype=torch.float16,
    low_cpu_mem_usage=True,
).to(device)
processor = AutoProcessor.from_pretrained(model_id)

image_file = "papers/sample_fig.jpg"
raw_image = Image.open(image_file).convert("RGB")

prompt = "USER: <image>\nWhat is this graph showing? ASSISTANT:"
inputs = processor(prompt, images=raw_image, return_tensors="pt").to(device)

output = model.generate(**inputs, max_new_tokens=200)
description = processor.decode(output[0][2:], skip_special_tokens=True)

print("--- VLM Description ---")
print(description)