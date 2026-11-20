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


def render_pose(points: list[tuple[float, float]], size: int = 1024, thickness: float = 1.0) -> np.ndarray:
    """Desenha o esqueleto no formato que o ControlNet OpenPose espera (fundo preto, RGB uint8)."""
    canvas = np.zeros((size, size, 3), np.uint8)
    pts = [(x * size, y * size) for x, y in points]
    stick = max(2, int(size / 128 * thickness))
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


def run_cycle(frames: int = 8, half: bool = False, three_quarter: bool = True,
              shoulder_spread: float = 0.05, lean: float = 0.35, flight: float = 0.025) -> list[list[tuple[float, float]]]:
    """Ciclo de CORRIDA de perfil (olhando para a direita), um passo completo (2 passadas).

    Diferencas para uma caminhada: tronco inclinado, joelho da perna de recuperacao sobe alto (calcanhar
    perto do gluteo), bracos dobrados em ~100 graus e bombeando, fase de voo (os dois pes fora do chao) e
    quadril que sobe no voo e desce no apoio. Com ``half=True`` devolve so meio ciclo (``frames`` poses de
    0 a pi): a segunda metade e a primeira com direita/esquerda trocadas, entao o loop fecha sozinho.
    ``three_quarter`` gira o tronco ~20 graus para a camera: ombros e quadris ficam separados, e o braco
    de tras (que o antebraco abre mais, mao atras do quadril) deixa de ficar escondido atras do tronco.
    """
    from math import cos, pi, sin

    def pt(origin, length, ang):  # ang 0 = para baixo, positivo = para a frente (+x)
        return (origin[0] + length * sin(ang), origin[1] + length * cos(ang))

    steps = frames * 2 if half else frames
    out = []
    for i in range(frames):
        ph = 2 * pi * i / steps
        hip = (0.46, 0.55 - flight * (1 - cos(2 * ph)))  # sobe ate 5% da altura no voo, desce no apoio
        neck = (hip[0] + 0.27 * sin(lean), hip[1] - 0.27 * cos(lean))
        nose = (neck[0] + 0.06, neck[1] - 0.06)
        legs, arms = {}, {}
        hs = 0.02 if three_quarter else 0.01
        for side, p in (("r", ph), ("l", ph + pi)):
            thigh = 0.2 + 0.75 * sin(p)                       # coxa vai mais para a frente que para tras
            flex = 0.2 + 1.7 * max(0.0, cos(p))               # recuperacao: joelho dobra ate o calcanhar subir
            knee = pt(hip, 0.18, thigh)
            legs[side] = (knee, pt(knee, 0.18, thigh - flex))
            spread = shoulder_spread if three_quarter else 0.01
            sho = (neck[0] + (-spread if side == "r" else spread), neck[1] + 0.015)
            up = -1.0 * sin(p) - 0.1
            flex = 0.9 + 0.9 * max(0.0, -sin(p))             # braco de tras mais aberto, o da frente fechado
            elb = pt(sho, 0.11, up)
            arms[side] = (sho, elb, pt(elb, 0.10, up + flex))
        out.append([nose, neck, arms["r"][0], arms["r"][1], arms["r"][2], arms["l"][0], arms["l"][1], arms["l"][2],
                    (hip[0] - hs, hip[1]), legs["r"][0], legs["r"][1], (hip[0] + hs, hip[1]), legs["l"][0], legs["l"][1],
                    (nose[0] - 0.025, nose[1] - 0.012), (nose[0] + 0.005, nose[1] - 0.012),
                    (nose[0] - 0.05, nose[1] + 0.005), (nose[0] - 0.035, nose[1] + 0.005)])
    return out


# ---------------------------------------------------------------- movimentos por quadros-chave
# Um "movimento" e uma lista de quadros-chave com os parametros do esqueleto; skeleton() faz a cinematica e
# interpolate() gera N poses suaves (com volta ao inicio, para o loop fechar). Novos movimentos = novos
# quadros-chave, sem codigo novo de desenho.

POSE_DEFAULTS: dict[str, float] = {
    "hip_x": 0.46, "lean": 0.10, "chest": 0.0, "head_dx": 0.06, "head_dy": -0.06, "spread": 0.05, "hs": 0.02,
    "ground": 0.89, "hip_dy": 0.0,
    # bracos: angulo do braco (0 = para baixo, + para a frente) e dobra do cotovelo; r = perto da camera
    "r_arm": 0.10, "r_elbow": 0.50, "l_arm": -0.08, "l_elbow": 0.60,
    # pernas: angulo da coxa e dobra do joelho
    "r_thigh": 0.12, "r_knee": 0.10, "l_thigh": -0.10, "l_knee": 0.10,
}


def skeleton(**kw: float) -> list[tuple[float, float]]:
    """Esqueleto COCO-18 por cinematica direta. Parametros em POSE_DEFAULTS; o pe mais baixo toca o chao
    (`ground`), a menos que `hip_dy` desloque o corpo (agachar, pular)."""
    from math import cos, sin

    p = {**POSE_DEFAULTS, **kw}
    unknown = set(kw) - set(POSE_DEFAULTS)
    if unknown:
        raise ValueError(f"parametros de pose desconhecidos: {sorted(unknown)}")

    def pt(origin, length, ang):
        return (origin[0] + length * sin(ang), origin[1] + length * cos(ang))

    def drop(side):  # extensao vertical da perna (quadril -> tornozelo)
        t, f = p[f"{side}_thigh"], p[f"{side}_knee"]
        return 0.18 * cos(t) + 0.18 * cos(t - f)

    hip = (p["hip_x"], p["ground"] - max(drop("r"), drop("l")) + p["hip_dy"])
    neck = (hip[0] + 0.27 * sin(p["lean"]), hip[1] - 0.27 * cos(p["lean"]) - p["chest"])
    nose = (neck[0] + p["head_dx"], neck[1] + p["head_dy"])
    legs, arms = {}, {}
    for side in ("r", "l"):
        knee = pt(hip, 0.18, p[f"{side}_thigh"])
        legs[side] = (knee, pt(knee, 0.18, p[f"{side}_thigh"] - p[f"{side}_knee"]))
        sho = (neck[0] + (-p["spread"] if side == "r" else p["spread"]), neck[1] + 0.015)
        up = p[f"{side}_arm"]
        elb = pt(sho, 0.11, up)
        arms[side] = (sho, elb, pt(elb, 0.10, up + p[f"{side}_elbow"]))
    hs = p["hs"]
    return [nose, neck, arms["r"][0], arms["r"][1], arms["r"][2], arms["l"][0], arms["l"][1], arms["l"][2],
            (hip[0] - hs, hip[1]), legs["r"][0], legs["r"][1], (hip[0] + hs, hip[1]), legs["l"][0], legs["l"][1],
            (nose[0] - 0.025, nose[1] - 0.012), (nose[0] + 0.005, nose[1] - 0.012),
            (nose[0] - 0.05, nose[1] + 0.005), (nose[0] - 0.035, nose[1] + 0.005)]


def interpolate(keys: list[tuple[float, dict[str, float]]], frames: int = 8) -> list[list[tuple[float, float]]]:
    """Quadros-chave [(t, parametros)], t em [0, 1), ciclicos (depois do ultimo volta ao primeiro) -> `frames` poses.

    Parametros ausentes num quadro-chave herdam o valor padrao; a interpolacao e cosseno (comeca e termina devagar).
    """
    from math import cos, pi

    ks = sorted(keys, key=lambda k: k[0])
    if not ks or any(not 0 <= t < 1 for t, _ in ks):
        raise ValueError("quadros-chave precisam ter t em [0, 1)")
    ext = ks + [(ks[0][0] + 1.0, ks[0][1])]  # fecha o ciclo
    out = []
    for i in range(frames):
        t = i / frames
        if t < ks[0][0]:
            t += 1.0
        a = max(j for j in range(len(ext) - 1) if ext[j][0] <= t)
        (t0, p0), (t1, p1) = ext[a], ext[a + 1]
        w = 0.5 - 0.5 * cos(pi * (t - t0) / (t1 - t0))
        names = set(p0) | set(p1)
        out.append(skeleton(**{n: (1 - w) * p0.get(n, POSE_DEFAULTS[n]) + w * p1.get(n, POSE_DEFAULTS[n]) for n in names}))
    return out


def idle_cycle(frames: int = 8, breath: float = 1.0) -> list[list[tuple[float, float]]]:
    """Parado respirando: o peito sobe e desce, os joelhos cedem um pouco na expiracao, os bracos balancam
    quase nada. `breath` multiplica a amplitude (1 = padrao)."""
    b = breath
    return interpolate([
        (0.00, {"chest": 0.000, "r_knee": 0.12, "l_knee": 0.12, "lean": 0.10, "r_arm": 0.10, "l_arm": -0.08}),
        (0.25, {"chest": 0.012 * b, "r_knee": 0.06, "l_knee": 0.06, "lean": 0.10 + 0.02 * b, "r_arm": 0.10 + 0.05 * b, "l_arm": -0.08 - 0.04 * b}),
        (0.50, {"chest": 0.018 * b, "r_knee": 0.04, "l_knee": 0.04, "lean": 0.10 + 0.03 * b, "r_arm": 0.10 + 0.08 * b, "l_arm": -0.08 - 0.06 * b, "head_dy": -0.06 - 0.004 * b}),
        (0.75, {"chest": 0.006 * b, "r_knee": 0.09, "l_knee": 0.09, "lean": 0.10 + 0.01 * b, "r_arm": 0.10 + 0.03 * b, "l_arm": -0.08 - 0.02 * b}),
    ], frames)
