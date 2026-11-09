"""Registered simulation cells (C2): a region mirroring the measured grid next to a loudspeaker in a shoebox room.

Region: 9 x 9 x 8 grid nodes at 0.15 m (1.2 x 1.2 x 1.05 m), the geometry of the measured subregions. Microphones
are a random subset of M nodes; every other node not within 0.15 m of a driver is a target.
Room: 6.4 x 7.2 x 3.0 m, locally reacting walls with normal-incidence reflection coefficient 0.85 (Eyring T60 about
0.39 s), ISO 9613-1 air absorption, image sources to 40 reflections.
"""
from __future__ import annotations
import itertools
import numpy as np
import fields as F

ROOM = np.array([6.4, 7.2, 3.0])
SHAPE = (9, 9, 8)
EXTENT = F.SPACING * (np.array(SHAPE) - 1)
R0 = 0.85
NMAX = 40
EXCLUDE = 0.15
BETA_REALISTIC = {125.0: 0.05, 250.0: 0.10, 500.0: 0.20, 1000.0: 0.30}
BASE = dict(f=500.0, M=96, place='below', dist=0.3, height=0.4, beta='realistic', delta=0.0, second=0.0,
            jitter=0.0, speed=0.0, snr=30.0, floor_phase=0.0)


def cells():
    """Registered cells: base, one-factor-at-a-time axes and a 16-run two-level fractional factorial."""
    out = {'base': dict(BASE)}

    def add(name, **kw):
        c = dict(BASE); c.update(kw); out[name] = c
    for name, kw in (('in', dict(place='inside')), ('below0.1', dict(dist=0.1, height=0.6)),
                     ('below0.6', dict(dist=0.6, height=0.1)), ('beside0.3', dict(place='beside', dist=0.3)),
                     ('beside0.6', dict(place='beside', dist=0.6)), ('beside1.2', dict(place='beside', dist=1.2))):
        add('A1-' + name, **kw)
    for f, M in itertools.product((125.0, 250.0, 500.0, 1000.0), (24, 48, 96, 192)):
        if (f, M) != (500.0, 96):
            add(f'A2-f{int(f)}-M{M}', f=f, M=M)
    lam = F.C0 / 500.0
    for ph in (0.0, 90.0, 180.0):
        for hw in (0.05, 0.25, None):
            if (ph, hw) != (0.0, None):
                add(f'A3-phi{int(ph)}-h{hw if hw else "base"}', floor_phase=ph,
                    height=BASE['height'] if hw is None else round(hw * lam, 4))
    for b in (0.0, 0.45):
        add(f'A4-beta{b}', beta=b)
    for d in (0.1, 0.2, 0.4):
        add(f'A5-delta{d}', delta=d)
    for a in (0.3, 0.6, 1.0):
        add(f'A6-second{a}', second=a)
    for j in (0.002, 0.005, 0.01, 0.02):
        add(f'A7-jitter{j}', jitter=j)
    for e in (0.005, 0.01, 0.02):
        add(f'A8-speed{e}', speed=e)
    for snr in (10.0, 20.0, 40.0):
        add(f'A9-snr{int(snr)}', snr=snr)
    # 2^(8-4) resolution IV: factors A..H, generators E = BCD, F = ACD, G = ABC, H = ABD.
    levels = dict(A=(('dist', 0.1, 'height', 0.6), ('dist', 0.6, 'height', 0.1)), B=(('beta', 'realistic'), ('beta', 0.45)),
                  C=(('delta', 0.0), ('delta', 0.2)), D=(('second', 0.0), ('second', 0.6)), E=(('jitter', 0.0), ('jitter', 0.01)),
                  F=(('speed', 0.0), ('speed', 0.02)), G=(('snr', 40.0), ('snr', 20.0)), H=(('floor_phase', 0.0), ('floor_phase', 180.0)))
    for i, (a, b, c, d) in enumerate(itertools.product((-1, 1), repeat=4)):
        sign = dict(A=a, B=b, C=c, D=d, E=b * c * d, F=a * c * d, G=a * b * c, H=a * b * d)
        kw = {}
        for fac, s in sign.items():
            lv = levels[fac][0 if s < 0 else 1]
            kw.update(dict(zip(lv[0::2], lv[1::2])))
        add(f'FF-{i:02d}', **kw)
        out[f'FF-{i:02d}']['ff_signs'] = sign
    return out


def _unit_h(rng):
    a = rng.uniform(0, 2 * np.pi)
    return np.array([np.cos(a), np.sin(a), 0.0])


def build(cell, seed):
    """Deterministic field for (cell, seed). All random draws use SeedSequence([seed, salt])."""
    f = float(cell['f'])
    lam = F.C0 / f
    rg = np.random.default_rng(np.random.SeedSequence([seed, 5101]))
    # region horizontal origin: region at least 1.2 m from the side walls
    ox = rg.uniform(1.2, ROOM[0] - 1.2 - EXTENT[0])
    oy = rg.uniform(1.2, ROOM[1] - 1.2 - EXTENT[1])
    u = rg.random(3)
    side = int(rg.integers(4))
    if cell['place'] == 'below':
        h = float(cell['height'])
        z0 = h + float(cell['dist'])
        src = np.array([ox + 0.3 + 0.6 * u[0], oy + 0.3 + 0.6 * u[1], h])
    elif cell['place'] == 'beside':
        z0 = 0.70
        zs = z0 + 0.15 + 0.6 * u[2]
        t = 0.3 + 0.6 * u[0]
        d = float(cell['dist'])
        src = [np.array([ox - d, oy + t * EXTENT[1], zs]), np.array([ox + EXTENT[0] + d, oy + t * EXTENT[1], zs]),
               np.array([ox + t * EXTENT[0], oy - d, zs]), np.array([ox + t * EXTENT[0], oy + EXTENT[1] + d, zs])][side]
    elif cell['place'] == 'inside':
        z0 = 0.70
        src = np.array([ox, oy, z0]) + 0.25 + (EXTENT - 0.5) * u
    else:
        raise ValueError(cell['place'])
    origin = np.array([ox, oy, z0])
    nodes = F.region_grid(origin, SHAPE)
    beta = BETA_REALISTIC[f] if cell['beta'] == 'realistic' else float(cell['beta'])
    axis = _unit_h(np.random.default_rng(np.random.SeedSequence([seed, 5102])))
    rs = np.random.default_rng(np.random.SeedSequence([seed, 5103]))
    drivers = [(src, 1.0 + 0j)]
    if cell['delta']:
        drivers.append((src + np.array([0, 0, float(cell['delta'])]), np.exp(1j * rs.uniform(0, 2 * np.pi))))
    if cell['second']:
        ang = rs.uniform(0, 2 * np.pi); r = rs.uniform(0.8, 1.2)
        s2 = np.array([src[0] + r * np.cos(ang), src[1] + r * np.sin(ang), src[2]])
        drivers.append((s2, float(cell['second']) * np.exp(1j * rs.uniform(0, 2 * np.pi))))
    for s_, _ in drivers:
        if np.any(s_ < 0.02) or np.any(s_ > ROOM - 0.02):
            raise ValueError('driver outside the room')
    dmin = np.min([np.linalg.norm(nodes - s_, axis=1) for s_, _ in drivers], axis=0)
    usable = np.where(dmin > EXCLUDE)[0]
    rm = np.random.default_rng(np.random.SeedSequence([seed, 5104, int(cell['M'])]))
    mic_idx = np.sort(rm.choice(usable, int(cell['M']), replace=False))
    tgt_idx = np.setdiff1d(usable, mic_idx)
    x = nodes[mic_idx]
    rj = np.random.default_rng(np.random.SeedSequence([seed, 5105]))
    true_x = x + float(cell['jitter']) * rj.normal(size=x.shape)
    zeta = {w: F.zeta_from_r0(R0) for w in ('x0', 'x1', 'y0', 'y1', 'z1')}
    zeta['z0'] = F.zeta_from_r0(R0 * np.exp(1j * np.deg2rad(float(cell['floor_phase']))))
    pts = np.vstack((true_x, nodes))
    p = np.zeros(len(pts), complex); direct = np.zeros(len(pts), complex)
    for s_, a in drivers:
        p += a * F.image_pressure(pts, s_, f, ROOM, zeta, NMAX, beta=beta, axis=axis)
        direct += a * F.direct_pressure(pts, s_, f, beta=beta, axis=axis)
    scale = np.sqrt(np.mean(np.abs(p[len(x):][usable]) ** 2))
    p, direct = p / scale, direct / scale
    pm, pn = p[:len(x)], p[len(x):]
    rn = np.random.default_rng(np.random.SeedSequence([seed, 5106, int(cell['M']), int(f)]))
    s2n = 10 ** (-float(cell['snr']) / 10)
    noise = np.sqrt(s2n / 2) * (rn.normal(size=len(x)) + 1j * rn.normal(size=len(x)))
    return dict(seed=seed, f=f, M=int(cell['M']), x=x, true_x=true_x, y=pm + noise, clean=pm, q=nodes[tgt_idx],
                truth=pn[tgt_idx], sources=np.array([s_ for s_, _ in drivers]), amps=np.array([a for _, a in drivers]),
                primary=src, origin=origin, box=(origin, origin + EXTENT), c_assumed=F.C0 * (1 + float(cell['speed'])),
                remainder_mics=(pm - direct[:len(x)]), remainder_targets=(pn - direct[len(x):])[tgt_idx],
                beta=beta, axis=axis, room=ROOM)
