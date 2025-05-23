# LMtoSD - Language Model to Stable Diffusion Image Generator

A Python application that uses a local language model (via LMStudio) to generate prompts for Stable Diffusion image generation.

## Features

- Integration with LMStudio for creative prompt generation
- Stable Diffusion v1.4 image generation
- Support for both positive and negative prompts
- Conversation history maintenance
- Automatic image opening after generation
- Consistent seed handling for reproducible results
- Support for custom image dimensions
- Resource usage monitoring

## Requirements

- Python 3.9+
- PyTorch
- diffusers
- Pillow
- requests
- LMStudio running locally

## Installation

1. Clone this repository:
```bash
git clone https://github.com/yourusername/LMtoSD.git
cd LMtoSD
```

2. Create and activate a virtual environment:
```bash
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

## Usage

1. Start LMStudio and ensure it's running at http://127.0.0.1:1234
2. Run the image generator:
```bash
python image_generator.py [--model_id MODEL] [other options]
```

3. Enter your image idea when prompted
4. Use the following commands:
   - `/reset` - Reset the conversation history
   - `/quit` - Exit the program

### Model Selection

You can select which Stable Diffusion model to use with the `--model_id` argument:

| Model Name                | Identifier (use with --model_id)         |
|--------------------------|------------------------------------------|
| SD 3.5 Medium (default)  | sd35medium                               |
| SDXL 1.0                 | sdxl                                     |
| SD 1.5                   | sd15                                     |
| SD 1.4                   | sd14                                     |

**Example usage:**
```bash
python image_generator.py --idea "A Maine coffee shop in the fall [768x768]" --model_id sdxl
python image_generator.py --idea "A Maine coffee shop in the fall [768x768]" --model_id sd15
python image_generator.py --idea "A Maine coffee shop in the fall [768x768]" --model_id sd14
# Default (no argument): SD 3.5 Medium
```

## Configuration

The script uses the following default settings:
- Stable Diffusion 3.5 Medium (default)
- 512x512 image dimensions (customizable)
- 35 inference steps (customizable)
- 7.5 guidance scale (customizable)

## License

MIT License # lms-to-sd
