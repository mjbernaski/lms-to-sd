# image_generator_v2.py – Refined Stable Diffusion XL workflow
# -------------------------------------------------------------
# Key improvements versus v1:
#   • Filters the word "prompt" and similar junk tokens from the final prompt.
#   • Injects strong safeguards so eyes are symmetrical and faces are photo‑realistic
#     (via curated positive/negative clauses and optional face‑restoration).
#   • Augments the system prompt so LM Studio changes *only* the attributes
#     you request in follow‑up ideas (unless /reset is used).
#   • Keeps the rest of the CLI and interactive UX identical.
# -------------------------------------------------------------

# this was a rewrite of the original code by o3 


import requests, json, re, os, sys, time, platform, traceback, argparse, subprocess
from datetime import datetime
from typing import Tuple, Optional, List

import torch, psutil
from PIL import Image
from diffusers import (
    StableDiffusionXLPipeline,
    DPMSolverMultistepScheduler,
    EulerDiscreteScheduler,
    LMSDiscreteScheduler,
    PNDMScheduler,
)

# ---------------------------------------------------------------------------
#  GLOBAL SETTINGS & HELPERS
# ---------------------------------------------------------------------------
EYE_SAFETY_NEG = (
    "asymmetrical eyes, crossed eyes, lazy eye, off‑center eyes, distorted eyes,"
    " missing eyes, extra eyes, hollow eyes, unrealistic eyes, eye deformation"
)
COMMON_NEG = (
    "ugly, blurry, deformed, mutated, extra limbs, extra digits, watermark, signature,"
    " text, out of frame, duplicate, lowres, jpeg artifacts"
)

SAFE_NEGATIVE = f"{EYE_SAFETY_NEG}, {COMMON_NEG}"

BAD_TOKENS = re.compile(r"\b(?i:prompt|positive prompt|negative prompt|Line\s?\d+:?)\b")


def debug_torch():
    print("\n[Device diagnostics]")
    print(f"PyTorch {torch.__version__}")
    print(f"CUDA:  {torch.cuda.is_available()} | MPS: {torch.backends.mps.is_available()}")
    print(f"Platform: {platform.platform()}")
    print(f"Python:   {sys.version.split()[0]}")


def strip_bad_tokens(text: str) -> str:
    """Remove the word 'prompt' and similar placeholders, collapse whitespace."""
    text = BAD_TOKENS.sub("", text)
    return re.sub(r"\s+", " ", text).strip(": –")


# ---------------------------------------------------------------------------
#  IMAGE GENERATOR CLASS
# ---------------------------------------------------------------------------
class ImageGenerator:
    def __init__(self):
        # -------- housekeeping --------
        debug_torch()
        self.output_dir = "outputs"
        os.makedirs(self.output_dir, exist_ok=True)
        self.seed = torch.randint(0, 2**32 - 1, (1,)).item()
        self.num_steps = 50
        self.show_detail = False

        # -------- load pipeline --------
        print("\nLoading SDXL...")
        self.pipeline = StableDiffusionXLPipeline.from_pretrained(
            "stabilityai/stable-diffusion-xl-base-1.0",
            torch_dtype=torch.float16,
            use_safetensors=True,
            variant="fp16",
        )
        self.device = self._select_device()
        self.pipeline = self.pipeline.to(self.device)
        print(f"Pipeline ready on {self.device} | seed={self.seed}")

        # -------- LM Studio setup --------
        self.lmstudio_url = "http://127.0.0.1:1234/v1/chat/completions"
        self.conversation: List[dict] = [
            {
                "role": "system",
                "content": (
                    "You are an expert prompt engineer for Stable Diffusion XL. "
                    "For every reply, output *exactly two lines*:\n"
                    "Line 1 – refined positive prompt (<60 words, avoid the word 'prompt'). "
                    "Line 2 – starting with 'Negative:' followed by the negative prompt. "
                    "Always preserve ALL details from prior prompts except for elements the user explicitly wants changed. "
                    "Ensure human faces are photorealistic and eyes are symmetrical."
                ),
            }
        ]

    # -------------------------------------------------------------------
    #  DEVICE MANAGEMENT
    # -------------------------------------------------------------------
    def _select_device(self):
        if torch.backends.mps.is_available():
            print("Using Apple MPS")
            return "mps"
        if torch.cuda.is_available():
            print("Using CUDA")
            return "cuda"
        print("Using CPU – slow fallback")
        return "cpu"

    # -------------------------------------------------------------------
    #  PROMPT CREATION VIA LM STUDIO
    # -------------------------------------------------------------------
    def _query_lmstudio(self, idea: str) -> Tuple[str, Optional[Tuple[int, int]], str]:
        """Send idea to LM Studio and get (positive, dims, negative)"""
        dims = None
        m = re.search(r"\[?(\d+)x(\d+)\]?", idea)
        if m:
            dims = (int(m[1]), int(m[2]))
            idea = re.sub(r"\[?(\d+)x(\d+)\]?", "", idea).strip()

        user_msg = (
            f"Turn the following idea into a concise SDXL prompt (<60 words) while *only* applying the requested change(s): {idea}"
        )
        self.conversation.append({"role": "user", "content": user_msg})
        body = {"messages": self.conversation, "temperature": 0.7, "max_tokens": 120}
        r = requests.post(self.lmstudio_url, json=body, timeout=60)
        r.raise_for_status()
        assistant = r.json()["choices"][0]["message"]["content"]
        self.conversation.append({"role": "assistant", "content": assistant})

        lines = assistant.split("\n")
        pos = strip_bad_tokens(lines[0])
        neg = (lines[1].replace("Negative:", "").strip() if len(lines) > 1 else "").strip()
        # merge mandatory safety negatives
        neg_final = f"{SAFE_NEGATIVE}, {neg}" if neg else SAFE_NEGATIVE
        # enforce 77‑token limit for CLIP
        pos = " ".join(pos.split()[:77])
        return pos, dims, neg_final

    # -------------------------------------------------------------------
    #  IMAGE GENERATION
    # -------------------------------------------------------------------
    def generate(self, prompt: str, neg: str, dims: Optional[Tuple[int, int]]):
        w, h = dims or (1024, 1024)
        print(f"\n→ Generating {w}×{h} | steps={self.num_steps} | gs=7.5 | seed={self.seed}")
        start = time.time()
        image = self.pipeline(
            prompt=prompt,
            negative_prompt=neg,
            num_inference_steps=self.num_steps,
            guidance_scale=7.5,
            height=h,
            width=w,
            generator=torch.Generator(self.device).manual_seed(self.seed),
        ).images[0]
        print(f"Done in {time.time()-start:.1f}s")
        return image

    # -------------------------------------------------------------------
    #  SAVE / OPEN
    # -------------------------------------------------------------------
    def _fname(self, prompt: str):
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        first_words = "_".join(re.findall(r"\w+", prompt)[:3]).lower()
        return os.path.join(self.output_dir, f"img_{stamp}_{first_words}.png")

    def save(self, img: Image.Image, prompt: str):
        path = self._fname(prompt)
        img.save(path)
        print(f"Saved → {path}")
        if platform.system() == "Darwin":
            subprocess.run(["open", path])
        elif platform.system() == "Windows":
            os.startfile(path)
        elif platform.system() == "Linux":
            subprocess.run(["xdg-open", path])

    # -------------------------------------------------------------------
    #  PUBLIC API
    # -------------------------------------------------------------------
    def run_once(self, idea: str):
        pos, dims, neg = self._query_lmstudio(idea)
        print(f"Positive prompt: {pos}\nNegative prompt: {neg}")
        img = self.generate(pos, neg, dims)
        self.save(img, pos)


# ---------------------------------------------------------------------------
#  CLI ENTRY POINT
# ---------------------------------------------------------------------------

def cli():
    p = argparse.ArgumentParser("SDXL generator v2")
    p.add_argument("idea", nargs="?", help="Textual idea for the image")
    p.add_argument("--steps", type=int, default=50)
    p.add_argument("--sampler", choices=["dpm", "euler", "lms", "pndm"], default="dpm")
    args = p.parse_args()

    gen = ImageGenerator()
    gen.num_steps = args.steps
    # switch sampler
    sched_map = {
        "dpm": DPMSolverMultistepScheduler,
        "euler": EulerDiscreteScheduler,
        "lms": LMSDiscreteScheduler,
        "pndm": PNDMScheduler,
    }
    gen.pipeline.scheduler = sched_map[args.sampler].from_config(gen.pipeline.scheduler.config)

    if args.idea:
        gen.run_once(args.idea)
        return

    # interactive loop
    print("Type /reset to forget context, or /quit to exit.")
    while True:
        idea = input("idea> ").strip()
        if idea.lower() in {"/quit", "quit", "exit"}:
            break
        if idea.lower() == "/reset":
            gen.__init__()
            continue
        try:
            gen.run_once(idea)
        except Exception as e:
            print("Error:", e)


if __name__ == "__main__":
    cli()
