"""Poses OpenPose prontas (esqueleto COCO-18) para guiar o ControlNet. Sem dependencias alem de numpy/cv2.

Coordenadas normalizadas (x, y em 0..1), vista frontal: o lado "direito" da pessoa fica na esquerda da imagem.
Ordem dos pontos: nariz, pescoco, ombro D, cotovelo D, pulso D, ombro E, cotovelo E, pulso E,
quadril D, joelho D, tornozelo D, quadril E, joelho E, tornozelo E, olho D, olho E, orelha D, orelha E.
"""
from __future__ import annotations

import cv2
import numpy as np

_LIMBS = [(1, 2), (1, 5), (2, 3), (3, 4), (5, 6), (6, 7), (1, 8), (8, 9), (9, 10), (1, 11), (11, 12), (12, 13),
          (1, 0), (0, 14), (14, 16), (0, 15), (15, 17)]
_COLORS = [(255, 0, 0), (255, 85, 0), (255, 170, 0), (255, 255, 0), (170, 255, 0), (85, 255, 0), (0, 255, 0),
           (0, 255, 85), (0, 255, 170), (0, 255, 255), (0, 170, 255), (0, 85, 255), (0, 0, 255), (85, 0, 255),
           (170, 0, 255), (255, 0, 255), (255, 0, 170), (255, 0, 85)]


def _body(**over):
    """Esqueleto base em pe, bracos levemente afastados; `over` troca pontos pelo nome."""
    p = {"nose": (.50, .20), "neck": (.50, .27), "rsho": (.40, .28), "relb": (.36, .42), "rwri": (.35, .55),
         "lsho": (.60, .28), "lelb": (.64, .42), "lwri": (.65, .55), "rhip": (.45, .55), "rkne": (.44, .73),
         "rank": (.44, .90), "lhip": (.55, .55), "lkne": (.56, .73), "lank": (.56, .90),
         "reye": (.48, .19), "leye": (.52, .19), "rear": (.45, .20), "lear": (.55, .20)}
    p.update(over)
    return [p[k] for k in ("nose", "neck", "rsho", "relb", "rwri", "lsho", "lelb", "lwri", "rhip", "rkne", "rank",
                           "lhip", "lkne", "lank", "reye", "leye", "rear", "lear")]


PRESETS: dict[str, list[tuple[float, float]]] = {
    "idle": _body(),
    "hands_on_hips": _body(relb=(.33, .42), rwri=(.42, .55), lelb=(.67, .42), lwri=(.58, .55)),
    "arms_up": _body(relb=(.34, .26), rwri=(.33, .13), lelb=(.66, .26), lwri=(.67, .13)),
    "walk": _body(relb=(.38, .41), rwri=(.40, .53), lelb=(.62, .41), lwri=(.58, .53),
                  rkne=(.42, .72), rank=(.37, .90), lkne=(.58, .70), lank=(.63, .85)),
}


def render_pose(points: list[tuple[float, float]], size: int = 1024) -> np.ndarray:
    """Desenha o esqueleto no formato que o ControlNet OpenPose espera (fundo preto, RGB uint8)."""
    canvas = np.zeros((size, size, 3), np.uint8)
    pts = [(x * size, y * size) for x, y in points]
    stick = max(2, size // 128)
    for (a, b), color in zip(_LIMBS, _COLORS):
        (x1, y1), (x2, y2) = pts[a], pts[b]
        length = float(np.hypot(x2 - x1, y2 - y1))
        angle = float(np.degrees(np.arctan2(y2 - y1, x2 - x1)))
        poly = cv2.ellipse2Poly((int((x1 + x2) / 2), int((y1 + y2) / 2)), (int(length / 2), stick), int(angle), 0, 360, 1)
        cv2.fillConvexPoly(canvas, poly, tuple(int(c * 0.6) for c in color))
    for (x, y), color in zip(pts, _COLORS):
        cv2.circle(canvas, (int(x), int(y)), stick, color, -1)
    return canvas


def preset_image(name: str, size: int = 1024) -> np.ndarray:
    if name not in PRESETS:
        raise ValueError(f"pose desconhecida: {name!r}; opcoes: {', '.join(PRESETS)}")
    return render_pose(PRESETS[name], size)


def run_cycle(frames: int = 8) -> list[list[tuple[float, float]]]:
    """Ciclo de corrida de perfil (olhando para a direita): `frames` esqueletos, um passo completo (2 passadas).

    Cinematica simples: coxas e bracos oscilam em oposicao, o joelho dobra na fase de recuperacao e o
    quadril sobe e desce duas vezes por ciclo.
    """
    from math import cos, pi, sin

    def pt(origin, length, ang):  # ang 0 = para baixo, positivo = para a frente (+x)
        return (origin[0] + length * sin(ang), origin[1] + length * cos(ang))

    out = []
    for i in range(frames):
        ph = 2 * pi * i / frames
        hip = (0.46, 0.53 + 0.02 * cos(2 * ph))
        lean = 0.22
        neck = (hip[0] + 0.27 * sin(lean), hip[1] - 0.27 * cos(lean))
        nose = (neck[0] + 0.06, neck[1] - 0.07)
        legs, arms = {}, {}
        for side, p in (("r", ph), ("l", ph + pi)):
            thigh = 0.8 * sin(p)
            flex = 0.15 + 1.2 * max(0.0, cos(p))
            knee = pt(hip, 0.18, thigh)
            legs[side] = (knee, pt(knee, 0.18, thigh - flex))
            sho = (neck[0] + (-0.01 if side == "r" else 0.01), neck[1] + 0.015)
            up = -0.95 * sin(p)
            elb = pt(sho, 0.12, up)
            arms[side] = (sho, elb, pt(elb, 0.11, up + 1.5))
        out.append([nose, neck, arms["r"][0], arms["r"][1], arms["r"][2], arms["l"][0], arms["l"][1], arms["l"][2],
                    (hip[0] - 0.01, hip[1]), legs["r"][0], legs["r"][1], (hip[0] + 0.01, hip[1]), legs["l"][0], legs["l"][1],
                    (nose[0] - 0.012, nose[1] - 0.012), (nose[0] + 0.012, nose[1] - 0.012),
                    (nose[0] - 0.05, nose[1] + 0.005), (nose[0] - 0.035, nose[1] + 0.005)])
    return out
