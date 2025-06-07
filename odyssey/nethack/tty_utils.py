from odyssey.nethack.constants import TTY_WIDTH, TTY_BOTTOM_BAR_HEIGHT

import numpy as np

def censor_bottom_bar(tty_chars, tty_colors, censor_ratio: float=0.5, inplace=False):
    if not inplace:
        tty_chars = np.copy(tty_chars)
        tty_colors = np.copy(tty_colors)

    censor_mask = np.random.rand(*tty_chars.shape[:-2], TTY_BOTTOM_BAR_HEIGHT, TTY_WIDTH) < censor_ratio
    tty_chars[..., -TTY_BOTTOM_BAR_HEIGHT:, :][censor_mask] = ord(" ")  # Replace with space character
    tty_colors[..., -TTY_BOTTOM_BAR_HEIGHT:, :][censor_mask] = 0  # Replace with default color (usually black)
    return tty_chars, tty_colors