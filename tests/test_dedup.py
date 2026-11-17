from PIL import Image

from src.data import dedup
from src.data.dedup import compute_groups
from tests.helpers import noise_array, save_img


def test_identicas_y_casi_identicas_mismo_grupo(tmp_path):
    a = noise_array(1)
    c = a.copy()
    c[:2, :2] = 0  # cambio minimo
    arrs = [a, a.copy(), c, noise_array(2)]
    paths = [save_img(tmp_path / f"{n}.jpg", x) for n, x in zip("abcd", arrs)]
    g = compute_groups(paths)
    assert g[paths[0]] == g[paths[1]] == g[paths[2]] == 0
    assert g[paths[3]] == 3


def test_distintas_grupos_distintos(tmp_path):
    paths = [save_img(tmp_path / f"{i}.jpg", noise_array(i + 10)) for i in range(4)]
    assert sorted(compute_groups(paths).values()) == [0, 1, 2, 3]


def test_lista_vacia():
    assert compute_groups([]) == {}


def test_determinismo(tmp_path):
    arrs = [noise_array(1), noise_array(1), noise_array(2), noise_array(3)]
    paths = [save_img(tmp_path / f"{i}.jpg", x) for i, x in enumerate(arrs)]
    assert compute_groups(paths) == compute_groups(paths)
    assert [compute_groups(paths)[p] for p in paths][:2] == [0, 0]


class FakeHash:
    def __init__(self, v):
        self.v = v

    def __sub__(self, other):
        return abs(self.v - other.v)


def fake_images(tmp_path, monkeypatch, values):
    """Imagenes cuyo ancho codifica el 'hash' falso (v + 1)."""
    paths = []
    for name, v in values.items():
        p = tmp_path / f"{name}.png"
        Image.new("RGB", (v + 1, 1)).save(p)
        paths.append(p)
    monkeypatch.setattr(dedup.imagehash, "phash", lambda im: FakeHash(im.size[0] - 1))
    return paths


def test_transitividad(tmp_path, monkeypatch):
    # A~B (5), B~C (5), pero A-C = 10 > umbral 6: deben unirse igualmente
    a, b, c, d = fake_images(tmp_path, monkeypatch, {"a": 0, "b": 5, "c": 10, "d": 100})
    g = compute_groups([a, b, c, d], hamming_threshold=6)
    assert [g[p] for p in (a, b, c, d)] == [0, 0, 0, 3]


def test_transitividad_orden_inverso(tmp_path, monkeypatch):
    c, b, a = fake_images(tmp_path, monkeypatch, {"c": 10, "b": 5, "a": 0})
    g = compute_groups([c, b, a], hamming_threshold=6)
    assert [g[p] for p in (c, b, a)] == [0, 0, 0]


def test_umbral_limite(tmp_path, monkeypatch):
    a, b, c = fake_images(tmp_path, monkeypatch, {"a": 0, "b": 6, "c": 7})
    assert compute_groups([a, b], 6) == {a: 0, b: 0}   # distancia == umbral: une
    assert compute_groups([a, c], 6) == {a: 0, c: 1}   # distancia > umbral: separa
