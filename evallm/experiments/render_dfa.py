import io

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

from evallm.enumerate_dfa.pack_dfa import unpack_dfa


def render_pdfa(pdfa):
    """
    Render a pdfa as a PIL image.
    """
    return Image.open(
        io.BytesIO(unpack_dfa(pdfa).show_diagram().draw(format="png"))
    ).convert("RGB")


def render_pdfas(pdfas, name):
    """
    Render a list of pdfas as a grid of images, using matplotlib.
    """
    n_cols = int(len(pdfas) ** 0.5)
    n_rows = (len(pdfas) + n_cols - 1) // n_cols
    size_each = 1
    _, axs = plt.subplots(
        n_rows,
        n_cols,
        figsize=(n_cols * size_each * 3, n_rows * size_each),
        tight_layout=True,
    )
    axs = list(axs.flatten())
    for ax in axs:
        ax.axis("off")
    for pdfa, ax in zip(pdfas, axs):
        ax.imshow(render_pdfa(pdfa))
    plt.suptitle(name)
