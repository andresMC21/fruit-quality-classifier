"""CNN pequeña desde cero: 2-3 bloques Conv+Pool + capas densas. Si hay transfer learning: congelar capas y añadir las propias.

Arquitectura (entrada 3 x 128 x 128):
    3 bloques [Conv 3x3 -> BatchNorm -> ReLU -> MaxPool 2x2] con 32, 64 y 128 filtros  -> 128 x 16 x 16
    Global Average Pooling                                                              -> 128
    Dense 128 -> 64, ReLU, Dropout(p) y Dense 64 -> n_classes (logits; softmax va en la perdida)
Unos 100 mil parametros. Global Average Pooling en vez de aplanar 128*16*16 = 32768 valores evita una capa
densa de millones de pesos, que con unos cientos de imagenes sobreajustaria.
"""
import torch
from torch import nn


def conv_block(c_in, c_out):
    return nn.Sequential(
        nn.Conv2d(c_in, c_out, kernel_size=3, padding=1, bias=False),
        nn.BatchNorm2d(c_out),
        nn.ReLU(inplace=True),
        nn.MaxPool2d(2),
    )


class SmallCNN(nn.Module):
    def __init__(self, n_classes=3, dropout=0.3):
        super().__init__()
        self.features = nn.Sequential(conv_block(3, 32), conv_block(32, 64), conv_block(64, 128))
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.head = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128, 64),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(64, n_classes),
        )

    def forward(self, x):
        return self.head(self.pool(self.features(x)))


def build_cnn(input_shape=(128, 128, 3), n_classes=3, dropout=0.3):
    if input_shape[2] != 3:
        raise ValueError("La red espera imagenes RGB (3 canales).")
    return SmallCNN(n_classes, dropout)


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def augment(x):
    """Aumento de datos por lote (solo en entrenamiento), en la GPU si x esta alli.

    Volteos y rotaciones de 90 grados: una manzana vista desde arriba no tiene orientacion preferida.
    Brillo +-10 %: simula cambios de luz sin cambiar el color de los defectos.
    """
    if torch.rand(1).item() < 0.5:
        x = torch.flip(x, dims=[3])
    if torch.rand(1).item() < 0.5:
        x = torch.flip(x, dims=[2])
    x = torch.rot90(x, k=int(torch.randint(0, 4, (1,))), dims=[2, 3])
    gain = 1 + (torch.rand(x.shape[0], 1, 1, 1, device=x.device) - 0.5) * 0.2
    return (x * gain).clamp(0, 1)
