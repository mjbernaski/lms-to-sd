import os
os.environ["DIFFUSERS_PROGRESS_BAR"] = "off"
os.environ["TRANSFORMERS_NO_TQDM"] = "1"
# Robustly suppress tqdm output globally
try:
    import tqdm
    class DummyTqdm:
        def __init__(self, *a, **k): pass
        def update(self, *a, **k): return self
        def close(self, *a, **k): return self
        def __enter__(self): return self
        def __exit__(self, *a): return False
        def write(self, *a, **k): pass
        def set_description(self, *a, **k): pass
        def set_postfix(self, *a, **k): pass
        def __iter__(self): return iter([])
    tqdm.tqdm = DummyTqdm
except ImportError:
    pass
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
import difflib
import textwrap
from rich import print
from rich.progress import Progress, BarColumn, TextColumn, TimeElapsedColumn, SpinnerColumn
from rich.traceback import install

install()  # Enable Rich tracebacks for better error messages

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

def wrap_console_text(text, words_per_line=13):
    words = text.split()
    lines = [" ".join(words[i:i+words_per_line]) for i in range(0, len(words), words_per_line)]
    return "\n".join(lines)

class ImageGenerator:
    def __init__(self, model_path=None, pipeline_class=None, model_name_for_prompt=None, use_sdxl=False, model_id=None):
        try:
            self.use_sdxl = use_sdxl
            self.model_name_for_prompt = model_name_for_prompt or ("Stable Diffusion XL" if use_sdxl else "Stable Diffusion 3.5 Medium")
            # Initialize conversation history for LM Studio
            self.conversation_history = [
                {
                    "role": "system",
                    "content": (
                        f"You are an expert prompt engineer for Stable Diffusion. For every reply, output exactly two lines and nothing else:\n"
                        "Line 1: A concise, visual description for an image (<60 words, no explanations, no extra lines, do not include the word 'prompt', do not explain or reference the prompt, do not use quotes).\n"
                        "Line 2: Negative: followed by a comma-separated list of visual flaws to avoid (e.g., ugly, blurry, deformed, watermark, text, extra limbs, asymmetrical eyes, etc.).\n"
                        "Never include explanations, never reference the prompt, never use quotes, never output anything except the two lines.\n"
                        "Examples:\n"
                        "A photorealistic portrait of a Roman emperor, dramatic lighting, laurel wreath, marble background\n"
                        "Negative: ugly, blurry, deformed, watermark, text, extra limbs, asymmetrical eyes\n"
                        "A lush forest landscape at sunrise, misty atmosphere, sunbeams through trees, high detail\n"
                        "Negative: blurry, lowres, cartoon, watermark, text, extra limbs, deformed\n"
                        "A futuristic city skyline at night, neon lights, reflections on water, cinematic\n"
                        "Negative: ugly, blurry, deformed, watermark, text, extra limbs, lowres, asymmetrical\n"
                    )
                }
            ]
            self.lmstudio_url = "http://127.0.0.1:1234/v1/chat/completions"
            if pipeline_class is None:
                if use_sdxl:
                    pipeline_class = StableDiffusionXLPipeline
                else:
                    pipeline_class = StableDiffusion3Pipeline
            self.pipeline_class = pipeline_class
            self.refiner = None
            # Only print model name once
            print(wrap_console_text(f"Model: {self.model_name_for_prompt}"))
            print()
            # Only print device info if needed
            # Debug device information (optional, comment out if not needed)
            # debug_torch_device()
            self.output_dir = "outputs"
            os.makedirs(self.output_dir, exist_ok=True)
            self.current_seed = torch.randint(0, 2**32 - 1, (1,)).item()
            self.num_steps = 35
            if model_id in ["sd15", "sd14"]:
                self.default_dimensions = (512, 512)
            else:
                self.default_dimensions = (1024, 1024)
            self.current_dimensions = self.default_dimensions
            self.show_detail = False
            self.idea_history = []
            self.use_same_seed_next = False
            self.common_prompt_rules = None  # Not printed
            self.show_diff = False
            # Loading pipeline
            dtype = torch.float32
            if torch.backends.mps.is_available():
                dtype = torch.float32
            if model_path:
                self.pipeline = self.pipeline_class.from_pretrained(
                    model_path,
                    torch_dtype=dtype,
                    use_safetensors=True
                )
            else:
                self.pipeline = self.pipeline_class.from_pretrained(
                    self.model_name_for_prompt,
                    torch_dtype=dtype,
                    use_safetensors=True
                )
            self.pipeline = self.pipeline.to('cpu')
            self.device = 'cpu'
            # Logging setup (unchanged)
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
        except Exception as e:
            print("Error during initialization:")
            print(str(e))
            print("Traceback:")
            print(traceback.format_exc())
            raise

    def get_prompt_from_lmstudio(self, user_input):
        """Get an enhanced prompt from LMStudio, update dimensions if specified."""
        print("\nGetting creative prompt from LMStudio...")
        print()
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
            print(wrap_console_text(f"Dimensions updated to: {self.current_dimensions[0]}x{self.current_dimensions[1]}"))
            print()
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
                # Post-processing: filter out lines with explanations, quotes, or irrelevant content
                filtered_parts = []
                for part in content_parts:
                    part_stripped = part.strip()
                    # Skip lines with quotes, explanations, or irrelevant phrases
                    if (
                        not part_stripped
                        or '"' in part_stripped
                        or "'" in part_stripped
                        or 'is a famous line' in part_stripped.lower()
                        or 'should not be associated' in part_stripped.lower()
                        or 'explanation' in part_stripped.lower()
                        or 'reference' in part_stripped.lower()
                        or part_stripped.lower().startswith('this prompt')
                        or part_stripped.lower().startswith('line 1:')
                        or part_stripped.lower().startswith('line 2:')
                    ):
                        continue
                    filtered_parts.append(part_stripped)
                # Now extract prompt and negative prompt from filtered parts
                for part in filtered_parts:
                    part_lower = part.lower().strip()
                    if part_lower in JUNK_TOKENS:
                        continue  # skip junk
                    if part.startswith("Description:"):
                        prompt = part.replace("Description:", "").strip()
                        break
                    elif part.startswith("Positive:"):
                        prompt = part.replace("Positive:", "").strip()
                        break
                    elif not part.startswith("Negative:"):
                        prompt = part.strip()
                        break
                for part in filtered_parts:
                    if part.startswith("Negative:"):
                        negative_prompt = part.replace("Negative:", "").strip()
                        break
                # If no prompt found or it's junk, use original input
                if not prompt or prompt.lower().strip() in JUNK_TOKENS:
                    prompt = original_input_for_lm.replace("/nothink ", "")
                # Ensure the prompt isn't too long
                if len(prompt.split()) > 77:
                    prompt = " ".join(prompt.split()[:77])
                # Always provide a standard negative prompt if missing
                DEFAULT_NEGATIVE_PROMPT = "ugly, blurry, deformed, mutated, extra limbs, extra digits, watermark, signature, text, out of frame, duplicate, lowres, jpeg artifacts, asymmetrical eyes, crossed eyes, lazy eye, off-center eyes, distorted eyes, missing eyes, extra eyes, hollow eyes, unrealistic eyes, eye deformation"
                if not negative_prompt or not negative_prompt.strip():
                    negative_prompt = DEFAULT_NEGATIVE_PROMPT
                # Fallback: if the prompt is not visually descriptive (e.g., contains 'famous line', 'should not be associated', or is too short), use the original input
                if (
                    'famous line' in prompt.lower()
                    or 'should not be associated' in prompt.lower()
                    or len(prompt.split()) < 3
                ):
                    prompt = original_input_for_lm.replace("/nothink ", "")
                
                # Show diff from last prompt unless reset or first prompt, and only if show_diff is set
                if getattr(self, 'show_diff', False) and hasattr(self, '_last_prompt') and self._last_prompt is not None:
                    diff = difflib.unified_diff(
                        self._last_prompt.split(),
                        prompt.split(),
                        lineterm='',
                        fromfile='previous',
                        tofile='current'
                    )
                    diff_lines = list(diff)
                    if diff_lines:
                        print(wrap_console_text("\nPrompt diff (previous → current):"))
                        print()
                        for line in diff_lines:
                            print(wrap_console_text(line))
                self._last_prompt = prompt
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
            {
                "role": "system",
                "content": (
                    f"You are an expert prompt engineer for Stable Diffusion. For every reply, output exactly two lines and nothing else:\n"
                    "Line 1: A concise, visual description for an image (<60 words, no explanations, no extra lines, do not include the word 'prompt', do not explain or reference the prompt, do not use quotes).\n"
                    "Line 2: Negative: followed by a comma-separated list of visual flaws to avoid (e.g., ugly, blurry, deformed, watermark, text, extra limbs, asymmetrical eyes, etc.).\n"
                    "Never include explanations, never reference the prompt, never use quotes, never output anything except the two lines.\n"
                    "Examples:\n"
                    "A photorealistic portrait of a Roman emperor, dramatic lighting, laurel wreath, marble background\n"
                    "Negative: ugly, blurry, deformed, watermark, text, extra limbs, asymmetrical eyes\n"
                    "A lush forest landscape at sunrise, misty atmosphere, sunbeams through trees, high detail\n"
                    "Negative: blurry, lowres, cartoon, watermark, text, extra limbs, deformed\n"
                    "A futuristic city skyline at night, neon lights, reflections on water, cinematic\n"
                    "Negative: ugly, blurry, deformed, watermark, text, extra limbs, lowres, asymmetrical\n"
                )
            }
        ]
        self.current_seed = torch.randint(0, 2**32 - 1, (1,)).item()
        print(wrap_console_text("\nConversation history has been reset."))
        print(wrap_console_text(f"New seed: {self.current_seed}"))
        self.idea_history = [] # Reset idea history
        self.current_dimensions = self.default_dimensions # Reset dimensions to default
        print(wrap_console_text(f"Dimensions reset to default: {self.current_dimensions[0]}x{self.current_dimensions[1]}"))
        self._last_prompt = None

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
        self.idea_history.append(original_idea)
        # print(wrap_console_text(f"\nGenerating: {prompt[:80]}{'...' if len(prompt) > 80 else ''}"))
        # print()
        width, height = self.current_dimensions
        # Only print key parameters
        print(wrap_console_text(f"Seed: {self.current_seed} | Steps: {num_inference_steps} | Size: {width}x{height}"))
        print()
        try:
            start_time = time.time()
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            generation_dir = os.path.join(self.output_dir, f"generation_{timestamp}")
            os.makedirs(generation_dir, exist_ok=True)
            self.current_generation_dir = generation_dir
            callback_params = {}
            if self.device != "cpu":
                callback_params["callback_on_step_end"] = self.callback_on_step_end
                callback_params["callback_on_step_end_tensor_inputs"] = ["latents"]
            # Use Rich progress bar for image generation
            with Progress(
                SpinnerColumn(),
                TextColumn("[progress.description]{task.description}"),
                BarColumn(),
                "[progress.percentage]{task.percentage:>3.0f}%",
                TimeElapsedColumn(),
                transient=True,
            ) as progress:
                task = progress.add_task("Generating image...", total=num_inference_steps)
                # Only update progress bar if needed (handled by diffusers if callback is set)
                image_result = self.pipeline(
                    prompt=prompt,
                    negative_prompt=negative_prompt or "Negative: ",
                    num_inference_steps=num_inference_steps,
                    guidance_scale=guidance_scale,
                    height=height,
                    width=width,
                    generator=torch.Generator(device=self.device).manual_seed(self.current_seed),
                    output_type="pil",
                    progress_bar=False,
                    **callback_params
                )
            base_image = image_result.images[0]
            refined_image = base_image
            duration = time.time() - start_time
            print(wrap_console_text(f"\nDone in {duration:.1f}s."))
            print()
            return refined_image
        except Exception as e:
            print(f"[bold red]Error generating image:[/bold red] {str(e)}")
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
        if image:
            try:
                filename = self.create_filename(prompt)
                image.save(filename)
                print(wrap_console_text(f"Saved: {filename}"))
                print()
                if platform.system() == 'Darwin':
                    subprocess.run(['open', filename])
                elif platform.system() == 'Windows':
                    os.startfile(filename)
                elif platform.system() == 'Linux':
                    subprocess.run(['xdg-open', filename])
            except Exception as e:
                print(f"[bold red]Error saving image:[/bold red] {str(e)}")
        else:
            print("[yellow]No image to save[/yellow]")

    # Converted to a class method
    def callback_on_step_end(self, pipe, step, timestep, callback_kwargs):
        # No-op: Rich progress bar already handles progress display on one line
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
    parser.add_argument('--model_id', type=str, choices=['sdxl', 'sd15', 'sd14'], default='sdxl', help='Select the model: sdxl (default), sd15, sd14')
    parser.add_argument('--all', action='store_true', help='Run the prompt through all supported models (requires --idea)')
    parser.add_argument('--all_samplers', action='store_true', help='Run the selected model with all supported samplers (requires --idea)')
    parser.add_argument('--diff', action='store_true', help='Show the diff between the last and new prompt')
    args = parser.parse_args()

    # Map model_id to Hugging Face repo and pipeline class
    model_map = {
        'sdxl': ('stabilityai/stable-diffusion-xl-base-1.0', StableDiffusionXLPipeline, 'Stable Diffusion XL 1.0'),
        'sd15': ('runwayml/stable-diffusion-v1-5', StableDiffusionPipeline, 'Stable Diffusion 1.5'),
        'sd14': ('CompVis/stable-diffusion-v1-4', StableDiffusionPipeline, 'Stable Diffusion 1.4'),
    }
    # Set SDXL as the default model
    selected_model_id = args.model_id if args.model_id else 'sdxl'
    model_path, pipeline_class, model_name_for_prompt = model_map[selected_model_id]

    print(wrap_console_text("\nStarting Image Generator..."))
    print()
    print(wrap_console_text(f"Using {model_name_for_prompt} model."))
    print()
    print(wrap_console_text("Make sure LMStudio is running at http://127.0.0.1:1234"))
    print()
    
    # Only SDXL should set use_sdxl True
    use_sdxl = (selected_model_id == 'sdxl')
    generator = ImageGenerator(
        model_path=args.model if args.model else model_path,
        pipeline_class=pipeline_class,
        model_name_for_prompt=model_name_for_prompt,
        use_sdxl=use_sdxl,
        model_id=selected_model_id
    )
    generator.show_diff = args.diff
    # Set device: use MPS for all models if available, otherwise CPU
    if torch.backends.mps.is_available():
        generator.device = 'mps'
        print(wrap_console_text("\nUsing MPS for this model"))
        print()
    else:
        generator.device = 'cpu'
        print(wrap_console_text("\nMPS not available, using CPU"))
        print()
    generator.pipeline = generator.pipeline.to(generator.device)
    
    # If --steps is provided, set the number of steps
    if args.steps:
        generator.num_steps = args.steps
        print(wrap_console_text(f"\nNumber of steps set to: {generator.num_steps}"))
        print()
    
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
        print(wrap_console_text(f"\nUsing {args.sampler.upper()} sampler"))
        print()
    else:
        print(wrap_console_text(f"\nUnknown sampler: {args.sampler}"))
    
    # Set initial guidance scale
    guidance_scale = args.guidance
    print(wrap_console_text(f"\nGuidance scale set to: {guidance_scale}"))
    print()
    
    # Set detail mode
    generator.show_detail = args.detail
    if generator.show_detail:
        print(wrap_console_text("\nDetailed intermediate images will be shown throughout the entire generation process"))
        print()
    
    if args.all:
        if not args.idea:
            print("Error: --all requires --idea to be specified.")
            return
        all_model_ids = ['sdxl', 'sd15', 'sd14']
        for model_id in all_model_ids:
            model_path, pipeline_class, model_name_for_prompt = model_map[model_id]
            print(wrap_console_text(f"\n--- Running with {model_name_for_prompt} ---"))
            print()
            use_sdxl = (model_id == 'sdxl')
            generator = ImageGenerator(
                model_path=args.model if args.model else model_path,
                pipeline_class=pipeline_class,
                model_name_for_prompt=model_name_for_prompt,
                use_sdxl=use_sdxl,
                model_id=model_id
            )
            # Set device: use MPS for all models if available, otherwise CPU
            if torch.backends.mps.is_available():
                generator.device = 'mps'
                print(wrap_console_text("\nUsing MPS for this model"))
                print()
            else:
                generator.device = 'cpu'
                print(wrap_console_text("\nMPS not available, using CPU"))
                print()
            generator.pipeline = generator.pipeline.to(generator.device)
            # If --steps is provided, set the number of steps
            if args.steps:
                generator.num_steps = args.steps
                print(wrap_console_text(f"\nNumber of steps set to: {generator.num_steps}"))
                print()
            # Set the sampler based on the argument
            if args.sampler in sampler_map:
                generator.pipeline.scheduler = sampler_map[args.sampler].from_config(generator.pipeline.scheduler.config)
                print(wrap_console_text(f"\nUsing {args.sampler.upper()} sampler"))
                print()
            else:
                print(wrap_console_text(f"\nUnknown sampler: {args.sampler}"))
            print(wrap_console_text(f"\nGuidance scale set to: {guidance_scale}"))
            print()
            generator.show_detail = args.detail
            if generator.show_detail:
                print(wrap_console_text("\nDetailed intermediate images will be shown throughout the entire generation process"))
                print()
            try:
                enhanced_prompt, negative_prompt = generator.get_prompt_from_lmstudio(args.idea)
                print(wrap_console_text(f"\nGenerated prompt: {enhanced_prompt}"))
                if negative_prompt:
                    print(wrap_console_text(f"\nNegative prompt: {negative_prompt}"))
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
        print(wrap_console_text(f"\n--- Running {model_name_for_prompt} with all samplers ---"))
        print()
        for sampler in sampler_names:
            print(wrap_console_text(f"\n--- Using sampler: {sampler.upper()} ---"))
            print()
            generator = ImageGenerator(
                model_path=args.model if args.model else model_path,
                pipeline_class=pipeline_class,
                model_name_for_prompt=model_name_for_prompt,
                use_sdxl=use_sdxl,
                model_id=selected_model_id
            )
            # Set device: use MPS for all models if available, otherwise CPU
            if torch.backends.mps.is_available():
                generator.device = 'mps'
                print(wrap_console_text("\nUsing MPS for this model"))
                print()
            else:
                generator.device = 'cpu'
                print(wrap_console_text("\nMPS not available, using CPU"))
                print()
            generator.pipeline = generator.pipeline.to(generator.device)
            if args.steps:
                generator.num_steps = args.steps
                print(wrap_console_text(f"\nNumber of steps set to: {generator.num_steps}"))
                print()
            generator.pipeline.scheduler = sampler_map[sampler].from_config(generator.pipeline.scheduler.config)
            print(wrap_console_text(f"\nUsing {sampler.upper()} sampler"))
            print()
            guidance_scale = args.guidance
            print(wrap_console_text(f"\nGuidance scale set to: {guidance_scale}"))
            print()
            generator.show_detail = args.detail
            if generator.show_detail:
                print(wrap_console_text("\nDetailed intermediate images will be shown throughout the entire generation process"))
                print()
            try:
                enhanced_prompt, negative_prompt = generator.get_prompt_from_lmstudio(args.idea)
                print(wrap_console_text(f"\nGenerated prompt: {enhanced_prompt}"))
                if negative_prompt:
                    print(wrap_console_text(f"\nNegative prompt: {negative_prompt}"))
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
            print(wrap_console_text(f"\nGenerated prompt: {enhanced_prompt}"))
            if negative_prompt:
                print(wrap_console_text(f"\nNegative prompt: {negative_prompt}"))
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
    print(wrap_console_text("Type '/reset' to reset the conversation history"))
    print()
    print(wrap_console_text("Type '/steps <number>' to change the number of inference steps (default: 75)"))
    print()
    print(wrap_console_text("Type '/guidance <number>' to change the guidance scale (default: 7.5)"))
    print()
    print(wrap_console_text("Type '/new-seed' to generate a new random seed for the next image"))
    print()
    print(wrap_console_text("Type '/same-seed' to use the same seed for the next image only"))
    print()
    print(wrap_console_text("Type '/quit' to exit"))
    print()
    
    waiting_for_idea_after_same_seed = False

    while True:
        if not waiting_for_idea_after_same_seed:
            print("\nEnter your image idea (or 'quit' to exit):")
        user_input = input("> ").strip()

        if user_input.lower() in {'quit', '/quit'}:
            break
        elif user_input.lower() == '/reset':
            generator.reset_conversation()
            print(wrap_console_text("\nConversation history has been reset. You can start a new conversation."))
            continue
        elif user_input.lower() == '/new-seed':
            generator.current_seed = torch.randint(0, 2**32 - 1, (1,)).item()
            print(wrap_console_text(f"\nNew seed set: {generator.current_seed}"))
            continue
        elif user_input.lower() == '/same-seed':
            generator.use_same_seed_next = True
            print(wrap_console_text("\nThe next image will use the same seed as the previous one."))
            waiting_for_idea_after_same_seed = True
            continue
        elif user_input.lower().startswith('/steps '):
            try:
                new_steps = int(user_input.split()[1])
                if new_steps < 1:
                    print(wrap_console_text("Number of steps must be at least 1"))
                    continue
                generator.num_steps = new_steps
                print(wrap_console_text(f"\nNumber of steps set to: {generator.num_steps}"))
                continue
            except (ValueError, IndexError):
                print(wrap_console_text("Invalid number of steps. Usage: /steps <number>"))
                continue
        elif user_input.lower().startswith('/guidance '):
            try:
                new_guidance = float(user_input.split()[1])
                if new_guidance < 0:
                    print(wrap_console_text("Guidance scale must be positive"))
                    continue
                guidance_scale = new_guidance
                print(wrap_console_text(f"\nGuidance scale set to: {guidance_scale}"))
                continue
            except (ValueError, IndexError):
                print(wrap_console_text("Invalid guidance scale. Usage: /guidance <number>"))
                continue

        # If it's not a command, treat it as an idea
        original_idea_for_generation = user_input

        # Seed logic: new seed unless /same-seed was used
        if generator.use_same_seed_next:
            print(wrap_console_text(f"\nUsing the same seed as previous: {generator.current_seed}"))
            generator.use_same_seed_next = False
            waiting_for_idea_after_same_seed = False
        else:
            generator.current_seed = torch.randint(0, 2**32 - 1, (1,)).item()
            print(wrap_console_text(f"\nNew random seed for this generation: {generator.current_seed}"))

        try:
            enhanced_prompt, negative_prompt = generator.get_prompt_from_lmstudio(original_idea_for_generation)
            print(wrap_console_text(f"\nGenerated prompt: {enhanced_prompt}"))
            if negative_prompt:
                print(wrap_console_text(f"Negative prompt: {negative_prompt}"))
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