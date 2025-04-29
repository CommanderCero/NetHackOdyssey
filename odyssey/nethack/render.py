import numpy as np
import os
from PIL import Image, ImageDraw, ImageFont

DEFAULT_FONT_PATH = os.path.join(os.path.dirname(__file__), "TheSansMono-Plain.otf")
DEFAULT_FONT = ImageFont.truetype(DEFAULT_FONT_PATH, 12)

def render_tty_chars(tty_chars: np.ndarray, font: ImageFont=DEFAULT_FONT) -> np.ndarray:
    lines = ["".join([chr(c) for c in line]) for line in tty_chars]
    full_text = "\n".join(lines)

    # Determine the text bounding box
    dummy_img = Image.new('RGB', (1, 1))
    left, top, width, height = ImageDraw.Draw(dummy_img).multiline_textbbox((0, 0), full_text, font=font)

    # Make sure height is divisible by 2
    if height % 2 != 0:
        height += 1  # Increase height by 1 pixel if not divisible by 2

    # Render the image
    image = Image.new('RGB', (width, height), color='black')
    canvas = ImageDraw.Draw(image)
    canvas.multiline_text((0, 0), full_text, font=font, fill="white")

    return np.array(image)