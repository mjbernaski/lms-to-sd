# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

LMtoSD (Language Model to Stable Diffusion) is a Python-based image generation tool that combines a local language model (LMStudio) with Stable Diffusion models to create images from text prompts. The tool enhances user prompts through LMStudio before passing them to various SD models.

## Key Commands

### Running the Application
```bash
# Standard run with Python
python image_generator.py [options]

# Using shell wrapper (handles venv)
./run_image_generator.sh [options]
# or
./lmsd [options]

# Common options:
# --model_id [sd-3.5-medium|sdxl-1.0|sd-1.5|sd-1.4]
# --idea "your prompt"
# --width 1024 --height 1024
# --steps 50
# --seed 42
```

### Development Commands
```bash
# Install dependencies
pip install -r requirements.txt

# Test LMStudio connection
python test_lmscall.py

# Test SDXL refiner
python test_sdxlrefiner.py
```

## Architecture & Structure

### Core Components
- **ImageGenerator class** (image_generator.py): Main application logic
  - Handles model loading and caching
  - Manages prompt enhancement via LMStudio
  - Implements image generation with various SD models
  - Provides interactive CLI interface

### Key Integration Points
- **LMStudio API**: Expects local LM running at `http://127.0.0.1:1234/v1/chat/completions`
- **Model Loading**: Uses Hugging Face diffusers, loads models to CPU by default
- **Output Management**: Saves images to `outputs/` with timestamp-based filenames

### Interactive Mode Features
- Continuous prompt loop with conversation history
- `/reset` command to clear conversation context
- `/quit` to exit
- Dynamic dimension setting with `[width]x[height]` format
- `/compress` to toggle intelligent prompt compression vs simple truncation

### Important Implementation Notes
1. The application suppresses tqdm progress bars for cleaner CLI output
2. Implements cumulative idea tracking - each prompt builds on previous ones
3. Resource monitoring shows CPU/GPU usage during generation
4. Supports multiple schedulers (DPM, Euler, LMS, etc.) configurable per model

## Working with the Code

When modifying this codebase:
1. Ensure LMStudio is running locally before testing
2. Generated images are saved in `outputs/` directory
3. The main entry point is the `main()` function in image_generator.py
4. Model initialization is cached to avoid repeated loading
5. Error handling includes specific checks for LMStudio connectivity
6. CLIP token limit (77 tokens) is automatically handled with:
   - Intelligent compression using LMStudio (default) - preserves key information
   - Simple truncation as fallback - use `/compress` command to toggle modes