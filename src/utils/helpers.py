"""Funciones auxiliares: E/S de imágenes, guardado de figuras vectoriales (SVG/PDF)."""
import matplotlib.pyplot as plt

def save_vector_fig(fig: plt.Figure, path_no_ext: str) -> None:
    """Guarda la figura en SVG y PDF (el enunciado exige calidad vectorial)."""
    fig.savefig(f"{path_no_ext}.svg", bbox_inches="tight")
    fig.savefig(f"{path_no_ext}.pdf", bbox_inches="tight")
