"""Genera el diagrama CRISP-DM personalizado (SVG) a partir de la estructura real del repositorio."""
import html

BOXES = [
    ("1. Comprensión del negocio", ["Calidad + tamaño de 6 frutas", "Costo asimétrico de errores", "Métrica: F1 macro, recall(mala)"], "#dbe8f5"),
    ("2. Comprensión de los datos", ["Kaggle + Fruit grade + propias", "'regular' solo en Fruit grade", "Fuente adivinable: 98,8 %"], "#e3f0dc"),
    ("3. Preparación de los datos", ["pHash + ráfagas → 6 012 grupos", "Otsu / GrabCut, fondo gris", "Split agrupado por fruit_id"], "#fdf0d5"),
    ("4. Modelado", ["Base: mayoritaria, logreg", "SVM, Random Forest (CV agrupada)", "CNN 3 conv. desde cero"], "#f6dede"),
    ("5. Evaluación", ["Test una sola vez + McNemar", "Por especie y por fuente", "Costo esperado, recall(mala)"], "#e6dcf0"),
    ("6. Despliegue", ["predict.py único", "Streamlit + selector de modelo", "Rechazo: no reconocida / revisión"], "#d8eceb"),
]
W, H, GAP, X0, Y0 = 250, 150, 40, 30, 90
parts = [f'<svg xmlns="http://www.w3.org/2000/svg" width="{X0*2+3*W+2*GAP}" height="{Y0+2*H+GAP+150}" font-family="Helvetica, Arial, sans-serif">',
         '<defs><marker id="a" markerWidth="10" markerHeight="8" refX="9" refY="4" orient="auto"><path d="M0,0 L10,4 L0,8 z" fill="#444"/></marker></defs>',
         f'<text x="{X0}" y="40" font-size="20" font-weight="bold">CRISP-DM adaptado: clasificador de calidad y tamaño de frutas</text>',
         f'<text x="{X0}" y="64" font-size="13" fill="#555">Iteraciones reales del proyecto (notas al pie); el flujo principal avanza de 1 a 6.</text>']
pos = []
for i, (t, lines, c) in enumerate(BOXES):
    r, col = divmod(i, 3)
    x, y = X0 + col * (W + GAP), Y0 + r * (H + GAP)
    pos.append((x, y))
    parts.append(f'<rect x="{x}" y="{y}" width="{W}" height="{H}" rx="10" fill="{c}" stroke="#444"/>')
    parts.append(f'<text x="{x+12}" y="{y+28}" font-size="15" font-weight="bold">{html.escape(t)}</text>')
    for k, l in enumerate(lines):
        parts.append(f'<text x="{x+14}" y="{y+58+k*26}" font-size="13">• {html.escape(l)}</text>')
def arrow(x1, y1, x2, y2, d=""):
    parts.append(f'<path d="M{x1},{y1} {d} L{x2},{y2}" fill="none" stroke="#444" stroke-width="1.8" marker-end="url(#a)"/>')
for i in (0, 1):
    arrow(pos[i][0] + W, pos[i][1] + H // 2, pos[i + 1][0], pos[i + 1][1] + H // 2)
arrow(pos[2][0] + W // 2, pos[2][1] + H, pos[3][0] + W // 2 + 0, pos[3][1], "")
for i in (3, 4):
    arrow(pos[i][0] + W, pos[i][1] + H // 2, pos[i + 1][0], pos[i + 1][1] + H // 2)
# iteraciones
yb = Y0 + 2 * H + GAP + 20
notes = [
    "Iteración 3→2: el EDA mostró ráfagas casi idénticas (12 909 imágenes en 6 012 grupos) → se limitó a 3 imágenes por grupo y se agrupó el split.",
    "Iteración 3→3: la segmentación fallaba en banano y lima por un control de forma demasiado estricto → umbrales por especie y re-segmentación.",
    "Iteración 4→3: la CNN subajustada (79 %) → más filtros, aumento más suave y tasa de aprendizaje por época.",
    "Regla de la fase 5: el test se evalúa una sola vez, una sola vez y con el split congelado; el test externo (otra fuente) mide la generalización.",
]
for k, n in enumerate(notes):
    parts.append(f'<text x="{X0}" y="{yb+k*24}" font-size="12.5" fill="#333">{html.escape(n)}</text>')
parts.append("</svg>")
open("docs/diagramas/crisp_dm_personalizado.svg", "w", encoding="utf-8").write("\n".join(parts))
