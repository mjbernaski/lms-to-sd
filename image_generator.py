import requests
import json
from diffusers import StableDiffusionXLPipeline, DPMSolverMultistepScheduler, EulerDiscreteScheduler, LMSDiscreteScheduler, PNDMScheduler, StableDiffusionXLImg2ImgPipeline
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
    def __init__(self):
        try:
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
            
            # Initialize number of steps (default 50 for SDXL)
            self.num_steps = 50
            print(f"\nInitial number of steps: {self.num_steps}")
            
            # Initialize detail mode
            self.show_detail = False
            
            print("\nLoading Stable Diffusion pipeline...")
            # Initialize Stable Diffusion XL pipeline
            self.pipeline = StableDiffusionXLPipeline.from_pretrained(
                "stabilityai/stable-diffusion-xl-base-1.0",
                torch_dtype=torch.float32,
                variant="fp16",  # Use fp16 for better memory efficiency
                use_safetensors=True
            )
            print("Pipeline loaded successfully")

            # Load the refiner pipeline
            print("Loading SDXL Refiner pipeline...")
            self.refiner = StableDiffusionXLImg2ImgPipeline.from_pretrained(
                "stabilityai/stable-diffusion-xl-refiner-1.0",
                torch_dtype=torch.float32,
                variant="fp16",
                use_safetensors=True
            )
            print("Refiner loaded successfully")
            
            # Determine the best device to use
            if torch.backends.mps.is_available():
                print("\nAttempting to use Apple Metal (MPS) for inference...")
                try:
                    self.pipeline = self.pipeline.to("mps")
                    self.device = "mps"
                    print("Successfully moved pipeline to MPS")
                except Exception as e:
                    print(f"Error moving to MPS: {str(e)}")
                    print("Traceback:", traceback.format_exc())
                    print("Falling back to CPU")
                    self.pipeline = self.pipeline.to("cpu")
                    self.device = "cpu"
            elif torch.cuda.is_available():
                print("\nUsing CUDA GPU for inference")
                self.pipeline = self.pipeline.to("cuda")
                self.device = "cuda"
            else:
                print("\nUsing CPU for inference")
                self.pipeline = self.pipeline.to("cpu")
                self.device = "cpu"
            
            print(f"\nPipeline initialized successfully on {self.device}")
            
            # LMStudio endpoint
            self.lmstudio_url = "http://127.0.0.1:1234/v1/chat/completions"
            
            # Initialize conversation history with updated system prompt for SDXL
            self.conversation_history = [
                {"role": "system", "content": """You are a creative AI assistant that helps craft detailed and effective prompts for Stable Diffusion XL. 
                Your task is to create coherent, evolving prompts that maintain and build upon previous details.
                
                Rules for prompt creation:
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
                """}
            ]
            
            print("\nInitialization complete!")
            
        except Exception as e:
            print("\nError during initialization:")
            print(str(e))
            print("\nTraceback:")
            print(traceback.format_exc())
            raise

    def get_prompt_from_lmstudio(self, user_input):
        """Get an enhanced prompt from LMStudio"""
        print("\nGetting creative prompt from LMStudio...")
        headers = {
            "Content-Type": "application/json"
        }
        
        # Check if user specified dimensions in input
        dimensions = None
        dimension_match = re.search(r'\[?(\d+)x(\d+)\]?', user_input)
        if dimension_match:
            width = int(dimension_match.group(1))
            height = int(dimension_match.group(2))
            dimensions = (width, height)
            # Remove the dimension specification from the input
            user_input = re.sub(r'\[?(\d+)x(\d+)\]?', '', user_input).strip()
        
        # Add /nothink to prevent thinking output
        user_input = f"/nothink {user_input}"
        
        # Add the user's input to the conversation history instead of resetting it
        self.conversation_history.append({
            "role": "user",
            "content": user_input
        })
        
        data = {
            "messages": self.conversation_history,
            "temperature": 0.7,
            "max_tokens": 120
        }
        
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
                
                # Look for the description/prompt
                for part in content_parts:
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
                
                # If no prompt found, use original input
                if not prompt:
                    prompt = user_input.replace("/nothink ", "")
                
                # Ensure the prompt isn't too long
                if len(prompt.split()) > 77:
                    prompt = " ".join(prompt.split()[:77])
                
                print(f"\nGenerated prompt: {prompt}")
                if negative_prompt:
                    print(f"Negative prompt: {negative_prompt}")
                
                return prompt, dimensions, negative_prompt
            else:
                print(f"\nError from LM Studio: {response.status_code}")
                return user_input.replace("/nothink ", ""), dimensions, None
        except Exception as e:
            print(f"\nError getting prompt from LM Studio: {str(e)}")
            return user_input.replace("/nothink ", ""), dimensions, None

    def reset_conversation(self):
        """Reset the conversation history and generate a new seed"""
        self.conversation_history = [
            {"role": "system", "content": """You are a creative AI assistant that helps craft detailed and effective prompts for Stable Diffusion XL. 
            Your task is to create coherent, evolving prompts that maintain and build upon previous details.
            
            Rules for prompt creation:
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
            """}
        ]
        self.current_seed = torch.randint(0, 2**32 - 1, (1,)).item()
        print("\nConversation history has been reset.")
        print(f"New seed: {self.current_seed}")

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

    def generate_image(self, prompt, dimensions=None, negative_prompt=None, num_inference_steps=50, guidance_scale=7.5):
        """Generate an image using Stable Diffusion XL and refine it"""
        print("\nGenerating image... This may take a few minutes...")
        print(f"\nParameters:")
        print(f"- Steps: {num_inference_steps}")
        print(f"- Guidance Scale: {guidance_scale}")
        print(f"- Device: {self.device.upper()}")
        print(f"- Seed: {self.current_seed}")
        
        # Remove "Positive" from the beginning of the prompt if it exists
        if prompt.lower().startswith("positive"):
            prompt = prompt[8:].strip()
        
        # Use specified dimensions or default to 1024x1024 (SDXL's optimal resolution)
        width = dimensions[0] if dimensions else 1024
        height = dimensions[1] if dimensions else 1024
        print(f"- Dimensions: {width}x{height} pixels")
        print(f"- Prompt: {prompt}")
        if negative_prompt:
            print(f"- Negative Prompt: {negative_prompt}")
        
        try:
            # Record start time
            start_time = time.time()
            
            # Create a unique folder for this generation
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            generation_dir = os.path.join(self.output_dir, f"generation_{timestamp}")
            os.makedirs(generation_dir, exist_ok=True)
            print(f"\nIntermediate images will be saved in: {generation_dir}")
            
            # Define callback function to show progress and intermediate images
            def callback_on_step_end(pipe, step, timestep, callback_kwargs):
                progress = (step + 1) / num_inference_steps * 100
                print(f"\rProgress: {progress:.1f}% (Step {step + 1}/{num_inference_steps})", end="")
                
                # Show intermediate image based on progress and detail mode
                should_show_image = False
                if self.show_detail:
                    should_show_image = True  # Save every step in detail mode
                elif progress >= 75 and progress <= 95:
                    should_show_image = True  # Save steps between 75-95% in normal mode
                
                if should_show_image:
                    # Get the current latents
                    latents = callback_kwargs["latents"]
                    
                    # Create a temporary pipeline for denoising
                    temp_pipe = pipe
                    temp_pipe.scheduler.set_timesteps(num_inference_steps)
                    
                    # Denoise the latents
                    denoised_latents = temp_pipe.scheduler.step(
                        latents,
                        timestep,
                        latents,
                        return_dict=False
                    )[0]
                    
                    # Decode the denoised latents to an image
                    image = pipe.vae.decode(denoised_latents / pipe.vae.config.scaling_factor, return_dict=False)[0]
                    # Convert to PIL image
                    image = pipe.image_processor.postprocess(image, output_type="pil")[0]
                    
                    # Save the intermediate image with high quality
                    temp_path = os.path.join(generation_dir, f"step{step + 1}_{progress:.1f}%.png")
                    image.save(temp_path, quality=95)
                
                # Show resource usage every 10 steps
                if (step + 1) % 10 == 0:
                    print(f"\nResource usage: {self.get_resource_usage()}")
                
                return callback_kwargs
            
            # Generate image with specified dimensions, negative prompt, and seed
            image = self.pipeline(
                prompt=prompt,
                negative_prompt=negative_prompt,
                num_inference_steps=num_inference_steps,
                guidance_scale=guidance_scale,
                height=height,
                width=width,
                generator=torch.Generator(device=self.device).manual_seed(self.current_seed),
                output_type="latent"  # <-- get latents for refiner
            ).images[0]
            
            # Refine the image
            print("Refining image for better details...")
            refined_image = self.refiner(
                image=image,
                prompt=prompt,
                negative_prompt=negative_prompt,
                num_inference_steps=30,  # Fewer steps for refiner is typical
                guidance_scale=guidance_scale,
            ).images[0]
            
            # Calculate duration
            duration = time.time() - start_time
            
            # Get resource usage
            resource_usage = self.get_resource_usage()
            
            print(f"\nGeneration complete!")
            print(f"Time taken: {duration:.1f} seconds")
            print(f"Resource usage: {resource_usage}")
            print(f"Image dimensions: {refined_image.size[0]}x{refined_image.size[1]} pixels")
            print(f"Intermediate images saved in: {generation_dir}")
            
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

def main():
    # Set up argument parser
    parser = argparse.ArgumentParser(description='Generate images using Stable Diffusion')
    parser.add_argument('--idea', type=str, help='Direct image generation idea')
    parser.add_argument('--steps', type=int, help='Number of inference steps (default: 75)')
    parser.add_argument('--sampler', type=str, choices=['dpm', 'euler', 'lms', 'pndm'], default='dpm',
                      help='Sampler to use: dpm (DPMSolverMultistep), euler (EulerDiscrete), lms (LMSDiscrete), pndm (PNDM)')
    parser.add_argument('--guidance', type=float, default=7.5,
                      help='Guidance scale (default: 7.5, higher values make the image more closely follow the prompt)')
    parser.add_argument('--detail', action='store_true',
                      help='Show detailed intermediate images throughout the entire generation process')
    args = parser.parse_args()

    print("\nStarting Image Generator...")
    print("Make sure LMStudio is running at http://127.0.0.1:1234")
    generator = ImageGenerator()
    
    # If --steps is provided, set the number of steps
    if args.steps:
        generator.num_steps = args.steps
        print(f"\nNumber of steps set to: {generator.num_steps}")
    
    # Set the sampler based on the argument
    if args.sampler == 'dpm':
        generator.pipeline.scheduler = DPMSolverMultistepScheduler.from_config(generator.pipeline.scheduler.config)
    elif args.sampler == 'euler':
        generator.pipeline.scheduler = EulerDiscreteScheduler.from_config(generator.pipeline.scheduler.config)
    elif args.sampler == 'lms':
        generator.pipeline.scheduler = LMSDiscreteScheduler.from_config(generator.pipeline.scheduler.config)
    elif args.sampler == 'pndm':
        generator.pipeline.scheduler = PNDMScheduler.from_config(generator.pipeline.scheduler.config)
    print(f"\nUsing {args.sampler.upper()} sampler")
    
    # Set initial guidance scale
    guidance_scale = args.guidance
    print(f"\nGuidance scale set to: {guidance_scale}")
    
    # Set detail mode
    generator.show_detail = args.detail
    if generator.show_detail:
        print("\nDetailed intermediate images will be shown throughout the entire generation process")
    
    # If --idea is provided, generate one image and exit
    if args.idea:
        try:
            enhanced_prompt, dimensions, negative_prompt = generator.get_prompt_from_lmstudio(args.idea)
            print(f"\nGenerated prompt: {enhanced_prompt}")
            if negative_prompt:
                print(f"Negative prompt: {negative_prompt}")
            if dimensions:
                print(f"Requested dimensions: {dimensions[0]}x{dimensions[1]} pixels")
            
            image = generator.generate_image(enhanced_prompt, dimensions, negative_prompt, 
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
    print("Type '/quit' to exit")
    
    while True:
        print("\nEnter your image idea (or 'quit' to exit):")
        user_input = input("> ")
        
        if user_input.lower() == 'quit' or user_input.lower() == '/quit':
            break
        elif user_input.lower() == '/reset':
            generator.reset_conversation()
            print("\nConversation history has been reset. You can start a new conversation.")
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
            
        try:
            enhanced_prompt, dimensions, negative_prompt = generator.get_prompt_from_lmstudio(user_input)
            print(f"\nGenerated prompt: {enhanced_prompt}")
            if negative_prompt:
                print(f"Negative prompt: {negative_prompt}")
            if dimensions:
                print(f"Requested dimensions: {dimensions[0]}x{dimensions[1]} pixels")
            
            image = generator.generate_image(enhanced_prompt, dimensions, negative_prompt, 
                                          num_inference_steps=generator.num_steps,
                                          guidance_scale=guidance_scale)
            generator.save_image(image, enhanced_prompt)
            
        except Exception as e:
            print(f"Error: {str(e)}")
            print("Type '/reset' to start a new conversation or 'quit' to exit.")

if __name__ == "__main__":
    main()