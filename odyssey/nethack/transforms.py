from odyssey.nethack.constants import BLSTATS_DUNGEON_NUMBER_INDEX, BLSTATS_LEVEL_NUMBER_INDEX, MAP_HEIGHT, MAP_WIDTH, SOLID_STONE_GLYPH
from odyssey.nethack.render import render_tty_chars, ImageFont, DEFAULT_FONT

from tensordict import TensorDict
from torchrl.envs import Transform
from torchrl.data import Composite, BoundedContinuous
import torch

import numpy as np
import os
from PIL import Image, ImageDraw, ImageFont

from typing import Optional

DEFAULT_FONT_PATH = os.path.join(os.path.dirname(__file__), "TheSansMono-Plain.otf")
DEFAULT_FONT = ImageFont.truetype(DEFAULT_FONT_PATH, 12)

def render_tty_chars_transform(data: TensorDict, out_key: str="pixels", font: ImageFont=DEFAULT_FONT) -> TensorDict:
    data.set(out_key, render_tty_chars(data["tty_chars"], font=font))
    return data

class MapCoverageRewardTransform(Transform):
    def __init__(self, 
        out_key: str = "reward",
        reward_scale: float = 0.1,
        glyphs_key: str = "glyphs",
        blstats_key: str = "blstats"
    ):
        super().__init__()
        self.out_key = out_key
        self.reward_scale = reward_scale
        self.glyphs_key = glyphs_key
        self.blstats_key = blstats_key

        self.previous_level = None
        self.register_buffer("level_coverage_map", torch.zeros((MAP_HEIGHT, MAP_WIDTH), dtype=bool))

    def _reset(self, *args, **kwargs):
        self.previous_level = None
        self.level_coverage_map[...] = False
        return super()._reset(*args, **kwargs)

    def _call(self, data: TensorDict) -> TensorDict:
        blstats = data[self.blstats_key]
        glyphs = data[self.glyphs_key]
        
        current_level = (blstats[BLSTATS_DUNGEON_NUMBER_INDEX].item(), blstats[BLSTATS_LEVEL_NUMBER_INDEX].item())
        if current_level != self.previous_level:
            # No reward when we change levels, just update our coverage map
            self.previous_level = current_level
            self.level_coverage_map = glyphs != SOLID_STONE_GLYPH
        else:
            # Reward proportional to the number of new tiles uncovered
            new_coverage = glyphs != SOLID_STONE_GLYPH
            coverage_diff = torch.sum(new_coverage & ~self.level_coverage_map) / (MAP_HEIGHT * MAP_WIDTH)
            reward = self.reward_scale * coverage_diff
            self.level_coverage_map |= new_coverage

            data[self.out_key] += reward

        return data