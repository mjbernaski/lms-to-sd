import requests
import json
from diffusers import StableDiffusionXLPipeline, StableDiffusion3Pipeline, StableDiffusionPipeline, DPMSolverMultistepScheduler, EulerDiscreteScheduler, LMSDiscreteScheduler, PNDMScheduler, StableDiffusionXLImg2ImgPipeline
from diffusers import DDIMScheduler, HeunDiscreteScheduler, UniPCMultistepScheduler, EulerAncestralDiscreteScheduler, DEISMultistepScheduler, KDPM2DiscreteScheduler, KDPM2AncestralDiscreteScheduler
import torch
from PIL import Image
import io
from datetime import datetime
import re
import os
import time
import psutil
import platform
import sys
import traceback
import subprocess
import argparse
import logging

def debug_torch_device():
    print("\nDebug information:")
    print(f"PyTorch version: {torch.__version__}")
    print(f"MPS available: {torch.backends.mps.is_available()}")
    print(f"MPS built: {torch.backends.mps.is_built()}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    print(f"Platform: {platform.platform()}")
    print(f"Python version: {sys.version}")
    print(f"Current working directory: {os.getcwd()}")
    print(f"Python executable: {sys.executable}")

class ImageGenerator:
    def __init__(self, model_path=None, pipeline_class=None, model_name_for_prompt=None, use_sdxl=False, model_id=None):
        try:
            self.use_sdxl = use_sdxl
            self.model_name_for_prompt = model_name_for_prompt or ("Stable Diffusion XL" if use_sdxl else "Stable Diffusion 3.5 Medium")
            if pipeline_class is None:
                if use_sdxl:
                    pipeline_class = StableDiffusionXLPipeline
                else:
                    pipeline_class = StableDiffusion3Pipeline
            self.pipeline_class = pipeline_class
            if self.use_sdxl:
                print("\nConfiguring for Stable Diffusion XL 1.0 pipeline...")
                self.refiner = None
                print("SDXL Refiner is not applicable and will not be loaded.")
            elif self.model_name_for_prompt == "Stable Diffusion 3.5 Medium":
                print("\nConfiguring for Stable Diffusion 3.5 Medium pipeline (default)...")
                self.refiner = None
                print("SDXL Refiner is not applicable for Stable Diffusion 3.5 and will not be loaded.")
            else:
                print(f"\nConfiguring for {self.model_name_for_prompt} pipeline...")
                self.refiner = None
                print("SDXL Refiner is not applicable and will not be loaded.")

            print("\nInitializing Stable Diffusion pipeline...")
            print("This may take a few minutes on first run as models are downloaded...")
            
            # Debug device information
            debug_torch_device()
            
            # Create outputs directory if it doesn't exist
            self.output_dir = "outputs"
            os.makedirs(self.output_dir, exist_ok=True)
            
            # Initialize random seed
            self.current_seed = torch.randint(0, 2**32 - 1, (1,)).item()
            print(f"\nInitial seed: {self.current_seed}")
            
            # Initialize number of steps (default 35 for SD3.5)
            self.num_steps = 35
            print(f"\nInitial number of steps: {self.num_steps}")
            
            # Set default and current dimensions based on model
            if model_id in ["sd15", "sd14"]:
                self.default_dimensions = (512, 512)
            else:
                self.default_dimensions = (1024, 1024)
            self.current_dimensions = self.default_dimensions
            print(f"Initial dimensions: {self.current_dimensions[0]}x{self.current_dimensions[1]}")
            
            # Initialize detail mode
            self.show_detail = False
            
            # Initialize idea history
            self.idea_history = []
            
            # Track if next generation should reuse the previous seed
            self.use_same_seed_next = False
            
            # Added: Common prompt rules
            self.common_prompt_rules = """Rules for prompt creation:
            1. Always include relevant details from previous prompts
            2. Keep prompts under 60 words
            3. Focus on visual elements, style, mood, lighting, and key details
            4. Be specific about composition, colors, and textures
            5. Maintain consistency in the scene and subject
            6. When modifying a prompt, preserve the successful elements
            7. Include artistic style references when appropriate
            8. Specify lighting conditions and atmosphere
            
            Format your responses as two lines:
            Line 1: The positive prompt (include all relevant details)
            Line 2: The negative prompt starting with "Negative: "
            """
            
            print("\nLoading Base Diffusion pipeline...")
            # Set dtype for memory efficiency and MPS compatibility
            if torch.backends.mps.is_available():
                dtype = torch.float32
                print("Using torch.float32 for pipeline on MPS (Mac) to avoid black images.")

            if model_path:
                print(f"Loading base model from: {model_path}")
                if model_path.endswith('.safetensors'):
                    self.pipeline = self.pipeline_class.from_single_file(
                        model_path, 
                        torch_dtype=dtype
                    )
                else:
                    self.pipeline = self.pipeline_class.from_pretrained(
                        model_path,
                        torch_dtype=dtype,
                        use_safetensors=True
                    )
            else:
                print(f"Loading base model from default ID: {self.model_name_for_prompt}")
                self.pipeline = self.pipeline_class.from_pretrained(
                    self.model_name_for_prompt,
                    torch_dtype=dtype,
                    use_safetensors=True
                )
            print("Base pipeline loaded successfully.")

            # After pipeline is loaded, force CPU for all operations
            self.pipeline = self.pipeline.to('cpu')
            self.device = 'cpu'
            print("Pipeline forced to CPU for inference (for memory compatibility).")
            
            # Enable memory-saving techniques for MPS or CUDA if not CPU
            if self.device != "cpu":
                try:
                    print("Attempting to enable model CPU offload...")
                    # self.pipeline.enable_model_cpu_offload() # Temporarily disabled for MPS/CUDA error diagnosis
                    # print("Model CPU offload enabled.")
                    print("Model CPU offload TEMPORARILY DISABLED for MPS/CUDA error diagnosis.")
                except Exception as e:
                    print(f"Could not enable model CPU offload: {e}")
                
                try:
                    print("Attempting to enable attention slicing...")
                    self.pipeline.enable_attention_slicing()
                    print("Attention slicing enabled.")
                except Exception as e:
                    print(f"Could not enable attention slicing: {e}")
            
            # LMStudio endpoint
            self.lmstudio_url = "http://127.0.0.1:1234/v1/chat/completions"
            
            # Modified system prompt
            self.conversation_history = [
                {"role": "system", "content": f"""You are a creative AI assistant that helps craft detailed and effective prompts for {self.model_name_for_prompt}. 
                Your task is to create coherent, evolving prompts that maintain and build upon previous details.  When in doubt the goal is photorealism.   
                
                {self.common_prompt_rules}"""}
            ]
            
            # Set up logging
            self.logger = logging.getLogger("ImageGenerator")
            self.logger.setLevel(logging.INFO)
            os.makedirs(self.output_dir, exist_ok=True)
            log_file = os.path.join(self.output_dir, "generation.log")
            fh = logging.FileHandler(log_file)
            fh.setLevel(logging.INFO)
            formatter = logging.Formatter('%(asctime)s %(levelname)s %(message)s')
            fh.setFormatter(formatter)
            if not self.logger.handlers:
                self.logger.addHandler(fh)
            
            print("\nInitialization complete!")
            
        except Exception as e:
            print("\nError during initialization:")
            print(str(e))
            print("\nTraceback:")
            print(traceback.format_exc())
            raise

    def get_prompt_from_lmstudio(self, user_input):
        """Get an enhanced prompt from LMStudio, update dimensions if specified."""
        print("\nGetting creative prompt from LMStudio...")
        headers = {
            "Content-Type": "application/json"
        }
        
        # Check if user specified dimensions in input
        original_input_for_lm = user_input # Keep original for LMStudio call
        dimension_match = re.search(r'\[?(\d+)x(\d+)\]?', user_input)
        if dimension_match:
            width = int(dimension_match.group(1))
            height = int(dimension_match.group(2))
            # Update the persistent dimensions for the session
            self.current_dimensions = (width, height)
            print(f"Dimensions updated to: {self.current_dimensions[0]}x{self.current_dimensions[1]}")
            # Remove the dimension specification from the input passed to LM Studio
            original_input_for_lm = re.sub(r'\[?(\d+)x(\d+)\]?', '', user_input).strip()
        
        # Add /nothink to prevent thinking output
        lm_studio_input = f"/nothink {original_input_for_lm}"
        
        # Add the user's input (without dimensions) to the conversation history
        self.conversation_history.append({
            "role": "user",
            "content": lm_studio_input
        })
        
        data = {
            "messages": self.conversation_history,
            "temperature": 0.7,
            "max_tokens": 120
        }
        
        # Define junk tokens to filter out
        JUNK_TOKENS = {"<think>", "Line 1", "<thinking>", "think", "thinking", "", "<|im_start|>", "<|im_end|>", "</think>"}
        
        try:
            response = requests.post(self.lmstudio_url, headers=headers, json=data)
            if response.status_code == 200:
                response_json = response.json()
                message = response_json["choices"][0]["message"]
                
                # Add the assistant's response to the conversation history
                self.conversation_history.append({
                    "role": "assistant",
                    "content": message.get("content", "")
                })
                
                # Get content from the response
                content_parts = message.get("content", "").strip().split("\n")
                content_parts = [p.strip() for p in content_parts if p.strip()]
                
                # Extract prompt and negative prompt
                prompt = None
                negative_prompt = None
                
                # Look for the description/prompt, skipping junk tokens
                for part in content_parts:
                    part_lower = part.lower().strip()
                    if part_lower in JUNK_TOKENS:
                        continue  # skip junk
                    if part.startswith("Description:"):
                        prompt = part.replace("Description:", "").strip()
                        break
                    elif not part.startswith("Negative:"):
                        prompt = part.strip()
                        break
                
                # Look for negative prompt
                for part in content_parts:
                    if part.startswith("Negative:"):
                        negative_prompt = part.replace("Negative:", "").strip()
                        break
                
                # If no prompt found or it's junk, use original input
                if not prompt or prompt.lower().strip() in JUNK_TOKENS:
                    prompt = original_input_for_lm.replace("/nothink ", "")
                
                # Ensure the prompt isn't too long
                if len(prompt.split()) > 77:
                    prompt = " ".join(prompt.split()[:77])
                
                return prompt, negative_prompt
            else:
                print(f"\nError from LM Studio: {response.status_code}")
                # Fallback to original input (without /nothink)
                return original_input_for_lm, None
        except Exception as e:
            print(f"\nError getting prompt from LM Studio: {str(e)}")
            # Fallback to original input (without /nothink)
            return original_input_for_lm, None

    def reset_conversation(self):
        """Reset the conversation history and generate a new seed"""
        # Modified system prompt
        self.conversation_history = [
            {"role": "system", "content": f"""You are a creative AI assistant that helps craft detailed and effective prompts for {self.model_name_for_prompt}. 
            Your task is to create coherent, evolving prompts that maintain and build upon previous details.
            
            {self.common_prompt_rules}"""}
        ]
        self.current_seed = torch.randint(0, 2**32 - 1, (1,)).item()
        print("\nConversation history has been reset.")
        print(f"New seed: {self.current_seed}")
        self.idea_history = [] # Reset idea history
        self.current_dimensions = self.default_dimensions # Reset dimensions to default
        print(f"Dimensions reset to default: {self.current_dimensions[0]}x{self.current_dimensions[1]}")

    def get_resource_usage(self):
        """Get current CPU and GPU usage"""
        cpu_percent = psutil.cpu_percent()
        if self.device == "mps":
            # For MPS, we can only get CPU usage
            return f"CPU: {cpu_percent}% | Device: Apple Metal"
        elif self.device == "cuda":
            gpu_memory = torch.cuda.memory_allocated() / 1024**2  # Convert to MB
            gpu_percent = torch.cuda.utilization()
            return f"CPU: {cpu_percent}% | GPU: {gpu_percent}% ({gpu_memory:.1f}MB)"
        return f"CPU: {cpu_percent}%"

    def generate_image(self, original_idea: str, prompt, negative_prompt=None, num_inference_steps=50, guidance_scale=7.5):
        """Generate an image using Stable Diffusion XL and refine it, using current dimensions."""
        self.idea_history.append(original_idea)
        print("\nGenerating image... This may take a few minutes...")
        print(f"\nParameters:")
        print(f"- Idea Chain: {' -> '.join(self.idea_history)}")
        print(f"- Positive Prompt: {prompt}")
        if negative_prompt:
            print(f"- Negative Prompt: {negative_prompt}")
        print(f"- Steps: {num_inference_steps}")
        print(f"- Guidance Scale: {guidance_scale}")
        print(f"- Device: {self.device.upper()}")
        print(f"- Seed: {self.current_seed}")
        width, height = self.current_dimensions
        print(f"- Dimensions: {width}x{height} pixels")
        
        # Determine output_type for the base pipeline
        output_type = "latent" if self.refiner else "pil"
        print(f"- Base pipeline output type: {output_type}")

        # Print sampler name if available
        sampler_name = getattr(self.pipeline.scheduler, '__class__', type('Unknown', (), {})).__name__
        print(f"- Sampler: {sampler_name}")
        if self.show_detail:
            print(f"- Detail Mode: ON (intermediate images every step)")
        # Remove "Positive" from the beginning of the prompt if it exists
        if prompt.lower().startswith("positive"):
            prompt = prompt[8:].strip()
        negative_prompt = negative_prompt or "Negative: "
        try:
            # Record start time
            start_time = time.time()
            
            # Create a unique folder for this generation
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            generation_dir = os.path.join(self.output_dir, f"generation_{timestamp}")
            os.makedirs(generation_dir, exist_ok=True)
            self.current_generation_dir = generation_dir # Added for callback access
            print(f"\nIntermediate images will be saved in: {generation_dir}")
            
            # Prepare callback parameters if not on CPU
            callback_params = {}
            if self.device != "cpu": 
                callback_params["callback_on_step_end"] = self.callback_on_step_end
                callback_params["callback_on_step_end_tensor_inputs"] = ["latents"]

            # Generate image with specified dimensions, negative prompt, and seed
            image_result = self.pipeline(
                prompt=prompt,
                negative_prompt=negative_prompt,
                num_inference_steps=num_inference_steps,
                guidance_scale=guidance_scale,
                height=height,
                width=width,
                generator=torch.Generator(device=self.device).manual_seed(self.current_seed),
                output_type=output_type,
                **callback_params 
            )
            base_image = image_result.images[0]
            
            # Robust: Always decode if not a PIL image
            from PIL import Image
            if not isinstance(base_image, Image.Image):
                print("\nDecoding latent output to PIL image (auto-fix for latent output)...")
                with torch.no_grad():
                    scaling_factor = getattr(self.pipeline.vae.config, "scaling_factor", 0.18215)
                    if hasattr(base_image, 'ndim') and base_image.ndim == 3:
                        base_image = base_image.unsqueeze(0)
                    decoded = self.pipeline.vae.decode(base_image / scaling_factor, return_dict=False)[0]
                    base_image = self.pipeline.image_processor.postprocess(decoded, output_type="pil")[0]
            
            # Refine the image if applicable
            if self.refiner and output_type == "latent":
                print("\nRefining image with SDXL Refiner...")
                refined_image = self.refiner(
                    image=base_image, 
                    prompt=prompt, 
                    negative_prompt=negative_prompt,
                    num_inference_steps=30,
                    guidance_scale=guidance_scale,
                    generator=torch.Generator(device=self.device).manual_seed(self.current_seed),
                ).images[0]
            elif output_type == "pil": 
                print("\nUsing base image directly (already PIL).")
                refined_image = base_image
            else: 
                print("\nDecoding latent base image as refiner is not used/available...")
                with torch.no_grad():
                    # VAE decoding might need scaling factor depending on the model/pipeline
                    # Assuming pipe.decode_latents handles this or VAE on pipeline is configured
                    if hasattr(self.pipeline, 'vae') and hasattr(self.pipeline.vae, 'config') and hasattr(self.pipeline.vae.config, 'scaling_factor'):
                         decoded_image = self.pipeline.vae.decode(base_image / self.pipeline.vae.config.scaling_factor, return_dict=False)[0]
                    else: # Fallback if scaling_factor is not obviously available, decode_latents is preferred
                         decoded_image = self.pipeline.decode_latents(base_image)[0] # decode_latents is safer
                refined_image = self.pipeline.image_processor.postprocess(decoded_image, output_type="pil")[0]
            
            # Calculate duration
            duration = time.time() - start_time
            
            # Get resource usage
            resource_usage = self.get_resource_usage()
            
            print(f"\nGeneration complete!")
            print(f"Time taken: {duration:.1f} seconds")
            print(f"Resource usage: {resource_usage}")
            print(f"Image dimensions: {refined_image.size[0]}x{refined_image.size[1]} pixels")
            print(f"Intermediate images saved in: {generation_dir}")

            # Log the generation
            output_file = self.create_filename(prompt)
            self.logger.info(f"Model: {self.model_name_for_prompt}, Sampler: {sampler_name}, Prompt: {prompt}, Negative: {negative_prompt}, Steps: {num_inference_steps}, Guidance: {guidance_scale}, Seed: {self.current_seed}, Dimensions: {width}x{height}, Output: {output_file}")
            
            return refined_image
        except Exception as e:
            print(f"Error generating image: {str(e)}")
            return None

    def create_filename(self, prompt):
        """Create a unique filename based on timestamp and prompt"""
        # Get current timestamp
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Extract first few meaningful words from prompt
        words = re.findall(r'\b\w+\b', prompt)
        short_desc = '_'.join(words[:3]).lower()
        
        # Create filename
        filename = f"img_{timestamp}_{short_desc}.png"
        return os.path.join(self.output_dir, filename)

    def save_image(self, image, prompt):
        """Save the generated image with a unique filename"""
        if image:
            try:
                filename = self.create_filename(prompt)
                image.save(filename)
                print(f"\nImage saved as {filename}")
                
                # Open the image after saving
                if platform.system() == 'Darwin':  # macOS
                    subprocess.run(['open', filename])
                elif platform.system() == 'Windows':
                    os.startfile(filename)
                elif platform.system() == 'Linux':
                    subprocess.run(['xdg-open', filename])
            except Exception as e:
                print(f"Error saving image: {str(e)}")
        else:
            print("No image to save")

    # Converted to a class method
    def callback_on_step_end(self, pipe, step, timestep, callback_kwargs):
        progress = (step + 1) / self.num_steps * 100
        # Simplified: print each progress update on a new line (no carriage return/overwrite)
        print(f"Progress: {progress:.1f}% (Step {step + 1}/{self.num_steps})")
        return callback_kwargs

def main():
    # Set up argument parser
    parser = argparse.ArgumentParser(description='Generate images using Stable Diffusion')
    parser.add_argument('--idea', type=str, help='Direct image generation idea')
    parser.add_argument('--steps', type=int, help='Number of inference steps (default: 50 for SDXL, or as set in generator)')
    parser.add_argument('--sampler', type=str, choices=['dpm', 'euler', 'lms', 'pndm', 'ddim', 'heun', 'unipc', 'euler_a', 'deis', 'k_dpm2', 'k_dpm2_a'], default='dpm',
                      help='Sampler to use: dpm (DPMSolverMultistep), euler (EulerDiscrete), lms (LMSDiscrete), pndm (PNDM), ddim (DDIM), heun (HeunDiscrete), unipc (UniPCMultistep), euler_a (EulerAncestral), deis (DEISMultistep), k_dpm2 (KDPM2Discrete), k_dpm2_a (KDPM2Ancestral)')
    parser.add_argument('--guidance', type=float, default=7.5,
                      help='Guidance scale (default: 7.5, higher values make the image more closely follow the prompt)')
    parser.add_argument('--detail', action='store_true',
                      help='Show detailed intermediate images throughout the entire generation process')
    parser.add_argument('--skip-refiner', action='store_true', help='Skip the SDXL refiner step (if SDXL 1.0 is used)')
    parser.add_argument('--model', type=str, help='Path to a local model directory or .safetensors file')
    parser.add_argument('--model_id', type=str, choices=['sd35medium', 'sdxl', 'sd15', 'sd14'], default='sd35medium', help='Select the model: sd35medium (default), sdxl, sd15, sd14')
    parser.add_argument('--all', action='store_true', help='Run the prompt through all supported models (requires --idea)')
    parser.add_argument('--all_samplers', action='store_true', help='Run the selected model with all supported samplers (requires --idea)')
    args = parser.parse_args()

    # Map model_id to Hugging Face repo and pipeline class
    model_map = {
        'sd35medium': ('stabilityai/stable-diffusion-3.5-medium', StableDiffusion3Pipeline, 'Stable Diffusion 3.5 Medium'),
        'sdxl': ('stabilityai/stable-diffusion-xl-base-1.0', StableDiffusionXLPipeline, 'Stable Diffusion XL 1.0'),
        'sd15': ('runwayml/stable-diffusion-v1-5', StableDiffusionPipeline, 'Stable Diffusion 1.5'),
        'sd14': ('CompVis/stable-diffusion-v1-4', StableDiffusionPipeline, 'Stable Diffusion 1.4'),
    }
    # Set SDXL as the default model
    selected_model_id = args.model_id if args.model_id else 'sdxl'
    model_path, pipeline_class, model_name_for_prompt = model_map[selected_model_id]

    print("\nStarting Image Generator...")
    print(f"Using {model_name_for_prompt} model.")
    print("Make sure LMStudio is running at http://127.0.0.1:1234")
    
    # Only SDXL should set use_sdxl True
    use_sdxl = (selected_model_id == 'sdxl')
    generator = ImageGenerator(
        model_path=args.model if args.model else model_path,
        pipeline_class=pipeline_class,
        model_name_for_prompt=model_name_for_prompt,
        use_sdxl=use_sdxl,
        model_id=selected_model_id
    )
    # Set device: use CPU for SD 3.5 Medium, MPS for all others (if available)
    if selected_model_id == 'sd35medium':
        generator.device = 'cpu'
        print("\nForcing CPU for SD 3.5 Medium (default)")
    elif torch.backends.mps.is_available():
        generator.device = 'mps'
        print("\nUsing MPS for this model")
    else:
        generator.device = 'cpu'
        print("\nMPS not available, using CPU")
    generator.pipeline = generator.pipeline.to(generator.device)
    
    # If --steps is provided, set the number of steps
    if args.steps:
        generator.num_steps = args.steps
        print(f"\nNumber of steps set to: {generator.num_steps}")
    
    # Set the sampler based on the argument
    sampler_map = {
        'dpm': DPMSolverMultistepScheduler,
        'euler': EulerDiscreteScheduler,
        'lms': LMSDiscreteScheduler,
        'pndm': PNDMScheduler,
        'ddim': DDIMScheduler,
        'heun': HeunDiscreteScheduler,
        'unipc': UniPCMultistepScheduler,
        'euler_a': EulerAncestralDiscreteScheduler,
        'deis': DEISMultistepScheduler,
        'k_dpm2': KDPM2DiscreteScheduler,
        'k_dpm2_a': KDPM2AncestralDiscreteScheduler,
    }
    if args.sampler in sampler_map:
        generator.pipeline.scheduler = sampler_map[args.sampler].from_config(generator.pipeline.scheduler.config)
        print(f"\nUsing {args.sampler.upper()} sampler")
    else:
        print(f"\nUnknown sampler: {args.sampler}")
    
    # Set initial guidance scale
    guidance_scale = args.guidance
    print(f"\nGuidance scale set to: {guidance_scale}")
    
    # Set detail mode
    generator.show_detail = args.detail
    if generator.show_detail:
        print("\nDetailed intermediate images will be shown throughout the entire generation process")
    
    if args.all:
        if not args.idea:
            print("Error: --all requires --idea to be specified.")
            return
        all_model_ids = ['sd35medium', 'sdxl', 'sd15', 'sd14']
        for model_id in all_model_ids:
            model_path, pipeline_class, model_name_for_prompt = model_map[model_id]
            print(f"\n--- Running with {model_name_for_prompt} ---")
            use_sdxl = (model_id == 'sdxl')
            generator = ImageGenerator(
                model_path=args.model if args.model else model_path,
                pipeline_class=pipeline_class,
                model_name_for_prompt=model_name_for_prompt,
                use_sdxl=use_sdxl,
                model_id=model_id
            )
            # Set device: use CPU for SD 3.5 Medium, MPS for all others (if available)
            if model_id == 'sd35medium':
                generator.device = 'cpu'
                print("\nForcing CPU for SD 3.5 Medium (default)")
            elif torch.backends.mps.is_available():
                generator.device = 'mps'
                print("\nUsing MPS for this model")
            else:
                generator.device = 'cpu'
                print("\nMPS not available, using CPU")
            generator.pipeline = generator.pipeline.to(generator.device)
            # If --steps is provided, set the number of steps
            if args.steps:
                generator.num_steps = args.steps
                print(f"\nNumber of steps set to: {generator.num_steps}")
            # Set the sampler based on the argument
            if args.sampler in sampler_map:
                generator.pipeline.scheduler = sampler_map[args.sampler].from_config(generator.pipeline.scheduler.config)
                print(f"\nUsing {args.sampler.upper()} sampler")
            else:
                print(f"\nUnknown sampler: {args.sampler}")
            guidance_scale = args.guidance
            print(f"\nGuidance scale set to: {guidance_scale}")
            generator.show_detail = args.detail
            if generator.show_detail:
                print("\nDetailed intermediate images will be shown throughout the entire generation process")
            try:
                enhanced_prompt, negative_prompt = generator.get_prompt_from_lmstudio(args.idea)
                print(f"\nGenerated prompt: {enhanced_prompt}")
                if negative_prompt:
                    print(f"\nNegative prompt: {negative_prompt}")
                image = generator.generate_image(args.idea, enhanced_prompt, negative_prompt, 
                                              num_inference_steps=generator.num_steps,
                                              guidance_scale=guidance_scale)
                generator.save_image(image, enhanced_prompt)
            except Exception as e:
                print(f"Error: {str(e)}")
        return

    if args.all_samplers:
        if not args.idea:
            print("Error: --all_samplers requires --idea to be specified.")
            return
        sampler_names = ['dpm', 'euler', 'lms', 'pndm', 'ddim', 'heun', 'unipc', 'euler_a', 'deis', 'k_dpm2', 'k_dpm2_a']
        if args.sampler and args.sampler not in sampler_names:
            print(f"Warning: {args.sampler} is not a recognized sampler, will use all supported samplers.")
        # Use the selected model only
        print(f"\n--- Running {model_name_for_prompt} with all samplers ---")
        for sampler in sampler_names:
            print(f"\n--- Using sampler: {sampler.upper()} ---")
            generator = ImageGenerator(
                model_path=args.model if args.model else model_path,
                pipeline_class=pipeline_class,
                model_name_for_prompt=model_name_for_prompt,
                use_sdxl=use_sdxl,
                model_id=selected_model_id
            )
            # Set device: use CPU for SD 3.5 Medium, MPS for all others (if available)
            if selected_model_id == 'sd35medium':
                generator.device = 'cpu'
                print("\nForcing CPU for SD 3.5 Medium (default)")
            elif torch.backends.mps.is_available():
                generator.device = 'mps'
                print("\nUsing MPS for this model")
            else:
                generator.device = 'cpu'
                print("\nMPS not available, using CPU")
            generator.pipeline = generator.pipeline.to(generator.device)
            if args.steps:
                generator.num_steps = args.steps
                print(f"\nNumber of steps set to: {generator.num_steps}")
            generator.pipeline.scheduler = sampler_map[sampler].from_config(generator.pipeline.scheduler.config)
            print(f"\nUsing {sampler.upper()} sampler")
            guidance_scale = args.guidance
            print(f"\nGuidance scale set to: {guidance_scale}")
            generator.show_detail = args.detail
            if generator.show_detail:
                print("\nDetailed intermediate images will be shown throughout the entire generation process")
            try:
                enhanced_prompt, negative_prompt = generator.get_prompt_from_lmstudio(args.idea)
                print(f"\nGenerated prompt: {enhanced_prompt}")
                if negative_prompt:
                    print(f"\nNegative prompt: {negative_prompt}")
                image = generator.generate_image(args.idea, enhanced_prompt, negative_prompt, 
                                              num_inference_steps=generator.num_steps,
                                              guidance_scale=guidance_scale)
                # Save with sampler name in filename
                if image:
                    orig_create_filename = generator.create_filename
                    def create_filename_with_sampler(prompt, sampler_name=sampler):
                        base = orig_create_filename(prompt)
                        base, ext = os.path.splitext(base)
                        return f"{base}_{sampler_name}{ext}"
                    generator.create_filename = create_filename_with_sampler
                    generator.save_image(image, enhanced_prompt)
                    generator.create_filename = orig_create_filename
            except Exception as e:
                print(f"Error: {str(e)}")
        return

    # If --idea is provided, generate one image and exit
    if args.idea and not args.all and not args.all_samplers:
        try:
            # Get prompts, dimensions are handled internally now
            enhanced_prompt, negative_prompt = generator.get_prompt_from_lmstudio(args.idea)
            print(f"\nGenerated prompt: {enhanced_prompt}")
            if negative_prompt:
                print(f"\nNegative prompt: {negative_prompt}")
            # Dimensions are no longer returned/needed here
            
            # Generate image using stored dimensions
            image = generator.generate_image(args.idea, enhanced_prompt, negative_prompt, 
                                          num_inference_steps=generator.num_steps,
                                          guidance_scale=guidance_scale)
            generator.save_image(image, enhanced_prompt)
            return
        except Exception as e:
            print(f"Error: {str(e)}")
            return
    
    # Interactive mode
    print("Type '/reset' to reset the conversation history")
    print("Type '/steps <number>' to change the number of inference steps (default: 75)")
    print("Type '/guidance <number>' to change the guidance scale (default: 7.5)")
    print("Type '/new-seed' to generate a new random seed for the next image")
    print("Type '/same-seed' to use the same seed for the next image only")
    print("Type '/quit' to exit")
    
    waiting_for_idea_after_same_seed = False

    while True:
        if not waiting_for_idea_after_same_seed:
            print("\nEnter your image idea (or 'quit' to exit):")
        user_input = input("> ").strip()

        if user_input.lower() in {'quit', '/quit'}:
            break
        elif user_input.lower() == '/reset':
            generator.reset_conversation()
            print("\nConversation history has been reset. You can start a new conversation.")
            continue
        elif user_input.lower() == '/new-seed':
            generator.current_seed = torch.randint(0, 2**32 - 1, (1,)).item()
            print(f"\nNew seed set: {generator.current_seed}")
            continue
        elif user_input.lower() == '/same-seed':
            generator.use_same_seed_next = True
            print("\nThe next image will use the same seed as the previous one.")
            waiting_for_idea_after_same_seed = True
            continue
        elif user_input.lower().startswith('/steps '):
            try:
                new_steps = int(user_input.split()[1])
                if new_steps < 1:
                    print("Number of steps must be at least 1")
                    continue
                generator.num_steps = new_steps
                print(f"\nNumber of steps set to: {generator.num_steps}")
                continue
            except (ValueError, IndexError):
                print("Invalid number of steps. Usage: /steps <number>")
                continue
        elif user_input.lower().startswith('/guidance '):
            try:
                new_guidance = float(user_input.split()[1])
                if new_guidance < 0:
                    print("Guidance scale must be positive")
                    continue
                guidance_scale = new_guidance
                print(f"\nGuidance scale set to: {guidance_scale}")
                continue
            except (ValueError, IndexError):
                print("Invalid guidance scale. Usage: /guidance <number>")
                continue

        # If it's not a command, treat it as an idea
        original_idea_for_generation = user_input

        # Seed logic: new seed unless /same-seed was used
        if generator.use_same_seed_next:
            print(f"\nUsing the same seed as previous: {generator.current_seed}")
            generator.use_same_seed_next = False
            waiting_for_idea_after_same_seed = False
        else:
            generator.current_seed = torch.randint(0, 2**32 - 1, (1,)).item()
            print(f"\nNew random seed for this generation: {generator.current_seed}")

        try:
            enhanced_prompt, negative_prompt = generator.get_prompt_from_lmstudio(original_idea_for_generation)
            print(f"\nGenerated prompt: {enhanced_prompt}")
            if negative_prompt:
                print(f"Negative prompt: {negative_prompt}")
            image = generator.generate_image(original_idea_for_generation, 
                                             enhanced_prompt, negative_prompt, 
                                             num_inference_steps=generator.num_steps,
                                             guidance_scale=guidance_scale)
            generator.save_image(image, enhanced_prompt)
        except Exception as e:
            print(f"Error: {str(e)}")
            print("Type '/reset' to start a new conversation or 'quit' to exit.")

if __name__ == "__main__":
    main()