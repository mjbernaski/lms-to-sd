import torch
from diffusers import StableDiffusionXLPipeline, StableDiffusionXLImg2ImgPipeline
from PIL import Image

# Choose device
device = "mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu"

# Load base SDXL pipeline
base = StableDiffusionXLPipeline.from_pretrained(
    "stabilityai/stable-diffusion-xl-base-1.0",
    torch_dtype=torch.float32,
    variant="fp16",
    use_safetensors=True
).to(device)

# Load SDXL Refiner pipeline
refiner = StableDiffusionXLImg2ImgPipeline.from_pretrained(
    "stabilityai/stable-diffusion-xl-refiner-1.0",
    torch_dtype=torch.float32,
    variant="fp16",
    use_safetensors=True
).to(device)

prompt = "A close-up portrait of a woman with intricate jewelry, photorealistic, high detail"
negative_prompt = "blurry, deformed, ugly, low quality, pixelated, duplicate, watermark"

# Generate base image (latent output)
print("Generating base image...")
base_result = base(
    prompt=prompt,
    negative_prompt=negative_prompt,
    num_inference_steps=40,
    guidance_scale=7.5,
    width=1024,
    height=1024,
    output_type="latent"
)

# Decode and save base image for comparison
with torch.no_grad():
    latents = base_result.images[0]
    if latents.ndim == 3:
        latents = latents.unsqueeze(0)  # Add batch dimension if missing
    scaling_factor = base.vae.config.scaling_factor if hasattr(base.vae.config, "scaling_factor") else 0.18215
    image = base.vae.decode(latents / scaling_factor, return_dict=False)[0]
    base_image = base.image_processor.postprocess(image, output_type="pil")[0]
    base_image.save("base_output.png")
print("Saved base_output.png")

# Refine the image
print("Refining image...")
refined_result = refiner(
    image=base_result.images[0],
    prompt=prompt,
    negative_prompt=negative_prompt,
    num_inference_steps=30,
    guidance_scale=7.5,
)

refined_result.images[0].save("refined_output.png")
print("Saved refined_output.png")