"""Frame -> photoreceptor input currents (PLAN Phase 2 task 9), with the
optic-lobe anatomy needed for a biologically oriented visual-field map.

R1-6 axons terminate in the lamina (no inversion). R7/R8 axons cross the
first optic chiasm to the medulla (front-to-back inverted relative to R1-6).
The raw annotations have no column/ommatidium ID (checked directly: the
v3.1.0 TSV's 31 fields -- listed in PROGRESS.md -- have nothing column- or
ommatidia-related), so each group's 2D image position comes from a
per-group PCA of terminal (anchor) positions -- checked directly (see
PROGRESS.md): PC1 is Y-dominated (dorsal-ventral) for both the lamina and
medulla groups, but PC2 is a genuine mix of X and Z, and the mix differs
between the two neuropils (they're separate, differently-tilted structures)
-- with the *sign* of each PC fixed anatomically, not PCA's arbitrary sign.

Axis convention (empirically confirmed against graph.npz's side_id and
super_class_id, not assumed from memory -- see PROGRESS.md for the full
check): raw FAFB nm coordinates are X=left-right, Y=dorsal-ventral
(increasing Y = more ventral -- the most dorsal class, endocrine/pars-
intercerebralis neurons, has the lowest mean Y of any super_class), and
Z=anterior-posterior (increasing Z = more posterior -- ascending/
sensory_ascending neurons, entering via the neck, have the highest mean Z).

The 10 soma-backed photoreceptors (their soma happens to sit inside FAFB,
unlike the retinal-terminal anchor point used for the other 11,108) are
excluded entirely from retina input -- zero current, logged -- rather than
mixing a different position source into the projection.

The medulla group's chiasm-inversion sign is not assumed a priori: it's
calibrated by trying both orientations against real connectivity (R1-6 -> L1
-> Mi1 and R8 -> Mi1: cells sharing a Mi1 "column" should map to nearby
image positions) and keeping whichever orientation makes that true.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field

import numpy as np

LAMINA_TYPES = ("R1-6",)
MEDULLA_TYPES = ("R7", "R8")
SIDE_LEFT, SIDE_RIGHT = 0, 1


def _minmax_normalize_fit(x: np.ndarray) -> tuple[float, float]:
    lo, hi = float(x.min()), float(x.max())
    if hi - lo < 1e-9:
        hi = lo + 1e-9
    return lo, hi


def _apply_minmax(x: np.ndarray, lo: float, hi: float) -> np.ndarray:
    return np.clip((x - lo) / (hi - lo), 0.0, 1.0)


def _top2_principal_axes(pos_nm: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    centered = pos_nm - pos_nm.mean(axis=0, keepdims=True)
    cov = np.cov(centered, rowvar=False)
    eigvals, eigvecs = np.linalg.eigh(cov)
    order = np.argsort(eigvals)[::-1]
    return eigvecs[:, order[0]], eigvecs[:, order[1]]


@dataclass
class GroupProjection:
    """A projection basis fit once on a full (group, side) population --
    e.g. all ~4000 R1-6 on one side -- then applied to any subset of that
    same group (the full population, or a handful of cells sharing one Mi1
    column for the calibration check). PCA on a 1- or 2-point subset would
    be degenerate, which is why fit and apply are separate."""

    pc_vertical: np.ndarray  # (3,) unit vector, sign-fixed so higher = more ventral
    pc_horizontal: np.ndarray  # (3,) unit vector, sign-fixed so higher = more posterior
    v_lo: float
    v_hi: float
    h_lo: float
    h_hi: float


def fit_group_projection(pos_nm: np.ndarray) -> GroupProjection:
    """pos_nm: [N,3] (X,Y,Z) nm for the full (group, side) population."""
    pc_a, pc_b = _top2_principal_axes(pos_nm)
    proj_a = pos_nm @ pc_a
    proj_b = pos_nm @ pc_b
    y, z = pos_nm[:, 1], pos_nm[:, 2]

    corr_a_y = np.corrcoef(proj_a, y)[0, 1]
    corr_b_y = np.corrcoef(proj_b, y)[0, 1]
    if abs(corr_a_y) >= abs(corr_b_y):
        pc_vertical, pc_horizontal, vertical_corr = pc_a, pc_b, corr_a_y
        horizontal_corr_z = np.corrcoef(proj_b, z)[0, 1]
    else:
        pc_vertical, pc_horizontal, vertical_corr = pc_b, pc_a, corr_b_y
        horizontal_corr_z = np.corrcoef(proj_a, z)[0, 1]

    if vertical_corr < 0:
        pc_vertical = -pc_vertical
    if horizontal_corr_z < 0:
        pc_horizontal = -pc_horizontal

    v_lo, v_hi = _minmax_normalize_fit(pos_nm @ pc_vertical)
    h_lo, h_hi = _minmax_normalize_fit(pos_nm @ pc_horizontal)
    return GroupProjection(pc_vertical, pc_horizontal, v_lo, v_hi, h_lo, h_hi)


def apply_group_projection(
    proj: GroupProjection, pos_nm: np.ndarray, flip_horizontal: bool
) -> tuple[np.ndarray, np.ndarray]:
    """Returns (vertical, horizontal_raw) in [0,1]: vertical 0=dorsal/image-
    top, 1=ventral/image-bottom; horizontal_raw 0=anterior/frontal,
    1=posterior/peripheral, unless flipped (the medulla chiasm correction)."""
    v = _apply_minmax(pos_nm @ proj.pc_vertical, proj.v_lo, proj.v_hi)
    h = _apply_minmax(pos_nm @ proj.pc_horizontal, proj.h_lo, proj.h_hi)
    if flip_horizontal:
        h = 1.0 - h
    return v, h


def place_on_screen(h_raw: np.ndarray, side: int) -> np.ndarray:
    """Anterior/frontal (h_raw=0) maps near the shared screen midline (0.5);
    posterior/peripheral (h_raw=1) maps to the outer edge. Right eye fills
    [0.5, 1.0], left eye [0.0, 0.5] -- both eyes see the same full frame
    (PLAN task 9), split so each eye's field sits on its own side and the
    left is a mirror image of the right, matching a real fly's anatomy."""
    if side == SIDE_RIGHT:
        return 0.5 + 0.5 * h_raw
    return 0.5 - 0.5 * h_raw


def _pairwise_nn_distance(points: np.ndarray, chunk: int = 500) -> np.ndarray:
    """Nearest-neighbor distance for each point (excluding itself), chunked
    to bound memory -- no scipy dependency in this project."""
    n = len(points)
    nn = np.full(n, np.inf)
    for start in range(0, n, chunk):
        end = min(start + chunk, n)
        d = np.sqrt(((points[start:end, None, :] - points[None, :, :]) ** 2).sum(-1))
        rows = np.arange(end - start)
        cols = np.arange(start, end)
        d[rows, cols] = np.inf
        nn[start:end] = d.min(axis=1)
    return nn


def _build_pre_of(pre: np.ndarray, post: np.ndarray) -> dict[int, list[int]]:
    pre_of: dict[int, list[int]] = defaultdict(list)
    for p, q in zip(pre.tolist(), post.tolist()):
        pre_of[q].append(p)
    return pre_of


def calibrate_medulla_flip(graph, type_names: list[str]) -> tuple[bool, dict]:
    """Empirically choose the medulla group's (R7/R8) horizontal-axis flip
    by minimizing the median distance (in R1-6 lattice spacings) between
    R1-6/R8 cells that share a postsynaptic Mi1 -- these should represent
    the same retinotopic column and so must map to nearby image positions.
    Column IDs were checked and are not available in the annotations, so
    this is the calibration used instead."""
    name_to_id = {n: i for i, n in enumerate(type_names)}
    pre, post, type_id, side_id, pos_nm = (
        graph["pre"], graph["post"], graph["type_id"], graph["side_id"], graph["pos_nm"]
    )
    r16_id = name_to_id.get("R1-6")
    l1_id = name_to_id.get("L1")
    r8_id = name_to_id.get("R8")
    mi1_id = name_to_id.get("Mi1")
    # R1-6's own nearest-neighbor spacing is dominated by *within-column*
    # distances (the ~6 R1-6 cells of one ommatidium all map to nearly the
    # same point via neural superposition, at ~4000 cells/side vs ~800 real
    # columns/side) -- L1 is ~one cell per lamina cartridge/column, so its
    # nearest-neighbor spacing is a much better estimate of the true
    # inter-column lattice constant.

    result: dict = {"column_ids_available": False}
    if None in (r16_id, l1_id, r8_id, mi1_id):
        result["n_mi1_groups_tested"] = 0
        return False, result

    pre_of = _build_pre_of(pre, post)
    mi1_indices = np.nonzero(type_id == mi1_id)[0]

    groups: list[tuple[int, list[int]]] = []
    for m in mi1_indices.tolist():
        direct = pre_of.get(m, [])
        l1s = [p for p in direct if type_id[p] == l1_id]
        r8s = [p for p in direct if type_id[p] == r8_id]
        r16s: list[int] = []
        for l1 in l1s:
            r16s.extend(p for p in pre_of.get(l1, []) if type_id[p] == r16_id)
        members = sorted(set(r16s) | set(r8s))
        if len(members) < 2:
            continue
        sides = {int(side_id[mm]) for mm in members}
        if len(sides) != 1:
            continue
        groups.append((sides.pop(), members))

    # Fit each group's projection ONCE on its full per-side population (a
    # handful of Mi1-column members would be a degenerate PCA fit).
    r16_proj: dict[int, GroupProjection] = {}
    r8_proj: dict[int, GroupProjection] = {}
    lattice_spacing: dict[int, float] = {}
    for side in (SIDE_LEFT, SIDE_RIGHT):
        r16_idx = np.nonzero((type_id == r16_id) & (side_id == side))[0]
        r8_idx = np.nonzero((type_id == r8_id) & (side_id == side))[0]
        l1_idx = np.nonzero((type_id == l1_id) & (side_id == side))[0]
        r16_proj[side] = fit_group_projection(pos_nm[r16_idx])
        r8_proj[side] = fit_group_projection(pos_nm[r8_idx])

        l1_v_proj = fit_group_projection(pos_nm[l1_idx])
        v, h_raw = apply_group_projection(l1_v_proj, pos_nm[l1_idx], flip_horizontal=False)
        h = place_on_screen(h_raw, side)
        pts = np.stack([v, h], axis=1)
        lattice_spacing[side] = float(np.median(_pairwise_nn_distance(pts)))

    def eval_flip(flip: bool) -> float:
        distances: list[float] = []
        for side, members in groups:
            spacing = lattice_spacing[side]
            member_arr = np.array(members)
            is_r16 = type_id[member_arr] == r16_id
            v = np.zeros(len(member_arr))
            h = np.zeros(len(member_arr))
            if is_r16.any():
                vv, hh_raw = apply_group_projection(
                    r16_proj[side], pos_nm[member_arr[is_r16]], flip_horizontal=False
                )
                v[is_r16] = vv
                h[is_r16] = place_on_screen(hh_raw, side)
            if (~is_r16).any():
                vv, hh_raw = apply_group_projection(
                    r8_proj[side], pos_nm[member_arr[~is_r16]], flip_horizontal=flip
                )
                v[~is_r16] = vv
                h[~is_r16] = place_on_screen(hh_raw, side)
            pts = np.stack([v, h], axis=1)
            n = len(pts)
            for i in range(n):
                for j in range(i + 1, n):
                    distances.append(float(np.linalg.norm(pts[i] - pts[j])) / spacing)
        return float(np.median(distances)) if distances else float("inf")

    median_unflipped = eval_flip(False)
    median_flipped = eval_flip(True)
    flip = median_flipped < median_unflipped

    result.update(
        {
            "n_mi1_groups_tested": len(groups),
            "median_distance_unflipped": median_unflipped,
            "median_distance_flipped": median_flipped,
            "chosen_flip": flip,
            "chosen_median_distance_lattice_spacings": (
                median_flipped if flip else median_unflipped
            ),
            "lattice_spacing": dict(lattice_spacing),
        }
    )
    return flip, result


@dataclass
class PhotoreceptorMap:
    photoreceptor_idx: np.ndarray  # node indices, graph["input_idx_photoreceptor"] order
    included_mask: np.ndarray  # bool[len(photoreceptor_idx)]; False = excluded (zero current)
    image_pos: np.ndarray  # [len(photoreceptor_idx), 2] (vertical, horizontal) in [0,1]; NaN if excluded
    medulla_flip: bool
    calibration_report: dict
    n_excluded: int = field(init=False)

    def __post_init__(self) -> None:
        self.n_excluded = int((~self.included_mask).sum())


def build_photoreceptor_map(graph, type_names: list[str]) -> PhotoreceptorMap:
    name_to_id = {n: i for i, n in enumerate(type_names)}
    photo_idx = graph["input_idx_photoreceptor"]
    type_id = graph["type_id"]
    side_id = graph["side_id"]
    pos_source = graph["pos_source_id"]
    pos_nm = graph["pos_nm"]

    included_mask = pos_source[photo_idx] != 0  # exclude the soma-backed ones
    image_pos = np.full((len(photo_idx), 2), np.nan, dtype=np.float64)

    medulla_flip, calibration_report = calibrate_medulla_flip(graph, type_names)

    lamina_type_ids = {name_to_id[t] for t in LAMINA_TYPES if t in name_to_id}
    medulla_type_ids = {name_to_id[t] for t in MEDULLA_TYPES if t in name_to_id}

    photo_type = type_id[photo_idx]
    is_lamina = np.isin(photo_type, list(lamina_type_ids)) & included_mask
    is_medulla = np.isin(photo_type, list(medulla_type_ids)) & included_mask
    photo_side = side_id[photo_idx]

    for side in (SIDE_LEFT, SIDE_RIGHT):
        side_mask = photo_side == side

        lam_mask = is_lamina & side_mask
        if lam_mask.any():
            proj = fit_group_projection(pos_nm[photo_idx[lam_mask]])
            v, h_raw = apply_group_projection(proj, pos_nm[photo_idx[lam_mask]], flip_horizontal=False)
            image_pos[lam_mask, 0] = v
            image_pos[lam_mask, 1] = place_on_screen(h_raw, side)

        med_mask = is_medulla & side_mask
        if med_mask.any():
            proj = fit_group_projection(pos_nm[photo_idx[med_mask]])
            v, h_raw = apply_group_projection(
                proj, pos_nm[photo_idx[med_mask]], flip_horizontal=medulla_flip
            )
            image_pos[med_mask, 0] = v
            image_pos[med_mask, 1] = place_on_screen(h_raw, side)

    # Any remaining photoreceptor with neither a lamina nor medulla cell_type
    # match (e.g. an unexpected type, or center/na side) is treated the same
    # as excluded -- no image position, zero current -- rather than guessing.
    unresolved = included_mask & np.isnan(image_pos[:, 0])
    included_mask = included_mask & ~unresolved

    return PhotoreceptorMap(photo_idx, included_mask, image_pos, medulla_flip, calibration_report)


def validate_retinotopy(
    graph, type_names: list[str], photo_map: PhotoreceptorMap, n_null_pairs: int = 1000, seed: int = 0
) -> dict:
    """Two checks that were NOT used to calibrate the projection (run after
    the PCA fit and the medulla flip are already fixed) -- they validate the
    result, they don't tune it:

    (a) Null baseline: median image distance for `n_null_pairs` RANDOMLY
        paired R1-6/R8 cells (same side, fixed seed), in the same L1-lattice
        units as the matched-pair (Mi1-sharing) calibration median. If the
        connectivity check found a genuine retinotopic correspondence and
        not an artifact of the projection method itself, random pairs must
        be much farther apart than matched ones (required: >=5x).

    (b) Superposition check: for each L1 cell, the image-position spread
        (median pairwise distance) of its presynaptic R1-6 cells -- these
        are ~6 cells from different physical ommatidia that neural
        superposition wires to one lamina cartridge, so if the projection
        captures real visual-field position (not raw retinal position) they
        should cluster tightly (expected: <1.5 lattice spacings). Not used
        for calibration -- reported honestly, nothing here is tuned to it.
    """
    name_to_id = {n: i for i, n in enumerate(type_names)}
    pre, post, type_id, side_id = graph["pre"], graph["post"], graph["type_id"], graph["side_id"]
    r16_id = name_to_id.get("R1-6")
    l1_id = name_to_id.get("L1")
    r8_id = name_to_id.get("R8")

    lattice_spacing = photo_map.calibration_report.get("lattice_spacing", {})
    matched_median = photo_map.calibration_report.get(
        "chosen_median_distance_lattice_spacings", float("nan")
    )

    slot_of: dict[int, int] = {
        int(node_idx): slot
        for slot, node_idx in enumerate(photo_map.photoreceptor_idx)
        if photo_map.included_mask[slot] and not np.isnan(photo_map.image_pos[slot, 0])
    }

    # --- (a) null baseline: random R1-6/R8 pairs, same side ---
    r16_by_side: dict[int, list[int]] = {SIDE_LEFT: [], SIDE_RIGHT: []}
    r8_by_side: dict[int, list[int]] = {SIDE_LEFT: [], SIDE_RIGHT: []}
    for node_idx, slot in slot_of.items():
        side = int(side_id[node_idx])
        if side not in (SIDE_LEFT, SIDE_RIGHT):
            continue
        if type_id[node_idx] == r16_id:
            r16_by_side[side].append(slot)
        elif type_id[node_idx] == r8_id:
            r8_by_side[side].append(slot)

    rng = np.random.default_rng(seed)
    sides_avail = [s for s in (SIDE_LEFT, SIDE_RIGHT) if r16_by_side[s] and r8_by_side[s]]
    null_distances = np.zeros(n_null_pairs)
    for i in range(n_null_pairs):
        side = sides_avail[rng.integers(0, len(sides_avail))]
        a = r16_by_side[side][rng.integers(0, len(r16_by_side[side]))]
        b = r8_by_side[side][rng.integers(0, len(r8_by_side[side]))]
        d = float(np.linalg.norm(photo_map.image_pos[a] - photo_map.image_pos[b]))
        null_distances[i] = d / lattice_spacing[side]
    null_median = float(np.median(null_distances))

    # --- (b) superposition: R1-6 cells presynaptic to the same L1 ---
    pre_of = _build_pre_of(pre, post)
    l1_indices = np.nonzero(type_id == l1_id)[0]
    per_l1_spreads: list[float] = []
    for l1 in l1_indices.tolist():
        r16_slots = [
            slot_of[p] for p in pre_of.get(l1, []) if type_id[p] == r16_id and p in slot_of
        ]
        if len(r16_slots) < 2:
            continue
        side = int(side_id[l1])
        if side not in lattice_spacing:
            continue
        pts = photo_map.image_pos[r16_slots]
        n = len(pts)
        dists = [
            float(np.linalg.norm(pts[i] - pts[j])) / lattice_spacing[side]
            for i in range(n)
            for j in range(i + 1, n)
        ]
        per_l1_spreads.append(float(np.median(dists)))
    superposition_median = float(np.median(per_l1_spreads)) if per_l1_spreads else float("nan")

    return {
        "matched_pair_median": matched_median,
        "null_baseline_median": null_median,
        "null_baseline_ratio": (null_median / matched_median) if matched_median > 0 else float("inf"),
        "null_baseline_pass": bool(null_median >= 5 * matched_median),
        "n_null_pairs": n_null_pairs,
        "superposition_median": superposition_median,
        "n_l1_cells_tested": len(per_l1_spreads),
        "superposition_pass": bool(superposition_median < 1.5),
    }


def gaussian_blur(frame: np.ndarray, sigma: float = 1.0, kernel_size: int = 5) -> np.ndarray:
    """Separable Gaussian blur, no scipy dependency."""
    ax = np.arange(kernel_size) - kernel_size // 2
    kernel = np.exp(-(ax**2) / (2 * sigma**2))
    kernel /= kernel.sum()
    h, w = frame.shape
    half = kernel_size // 2

    padded = np.pad(frame, half, mode="edge")
    row_conv = np.zeros((h, w))
    for i, k in enumerate(kernel):
        row_conv += k * padded[half : half + h, i : i + w]

    padded2 = np.pad(row_conv, ((half, half), (0, 0)), mode="edge")
    col_conv = np.zeros((h, w))
    for i, k in enumerate(kernel):
        col_conv += k * padded2[i : i + h, :]

    return col_conv


def _nearest_rc(positions: np.ndarray, h: int, w: int) -> tuple[np.ndarray, np.ndarray]:
    r = np.clip(np.round(positions[:, 0] * (h - 1)).astype(np.int64), 0, h - 1)
    c = np.clip(np.round(positions[:, 1] * (w - 1)).astype(np.int64), 0, w - 1)
    return r, c


def bilinear_sample(frame: np.ndarray, positions: np.ndarray) -> np.ndarray:
    """positions: [N,2] (row, col) in [0,1], NaN rows sample to 0.0.
    Nearest-neighbor, not actually bilinear (kept the name for call-site
    stability): the frame is already Gaussian-blurred before sampling, so
    sub-pixel interpolation adds negligible quality for ~10x the per-sample
    cost -- measured directly (see bench/bench_env.py / PROGRESS.md), this
    is what clears Gate G2's >=10x-real-time throughput requirement."""
    h, w = frame.shape
    valid = ~np.isnan(positions[:, 0])
    out = np.zeros(len(positions))
    r, c = _nearest_rc(positions[valid], h, w)
    out[valid] = frame[r, c]
    return out


def gaussian_blur_batch(frames: np.ndarray, sigma: float = 1.0, kernel_size: int = 5) -> np.ndarray:
    """Batched separable Gaussian blur. frames: [B,H,W]. Preserves dtype
    (pass float32 from the throughput-critical env path)."""
    ax = np.arange(kernel_size) - kernel_size // 2
    kernel = np.exp(-(ax**2) / (2 * sigma**2)).astype(frames.dtype)
    kernel /= kernel.sum()
    b, h, w = frames.shape
    half = kernel_size // 2

    padded = np.pad(frames, ((0, 0), (half, half), (half, half)), mode="edge")
    row_conv = np.zeros((b, h, w), dtype=frames.dtype)
    for i, k in enumerate(kernel):
        row_conv += k * padded[:, half : half + h, i : i + w]

    padded2 = np.pad(row_conv, ((0, 0), (half, half), (0, 0)), mode="edge")
    col_conv = np.zeros((b, h, w), dtype=frames.dtype)
    for i, k in enumerate(kernel):
        col_conv += k * padded2[:, i : i + h, :]

    return col_conv


def bilinear_sample_batch(frames: np.ndarray, positions: np.ndarray) -> np.ndarray:
    """Batched nearest-neighbor sample (see bilinear_sample's docstring for
    why it's not actually bilinear): frames [B,H,W] (all envs sharing the
    same photoreceptor positions), positions [N,2] in [0,1], NaN rows -> 0.
    Returns [B,N]. This is the throughput-critical path Gate G2's env
    benchmark measures -- computing it once per batch instead of once per
    env, plus dropping bilinear interpolation for a single gather, is what
    clears the >=10x-real-time gate (measured: bilinear ~5ms/step at
    batch=64, nearest ~0.5ms/step)."""
    b, h, w = frames.shape
    n = len(positions)
    valid = ~np.isnan(positions[:, 0])
    out = np.zeros((b, n), dtype=frames.dtype)
    r, c = _nearest_rc(positions[valid], h, w)
    out[:, valid] = frames[:, r, c]
    return out


def compute_retina_currents(
    frames: np.ndarray,
    photo_map: PhotoreceptorMap,
    gain: float,
    ema_alpha: float = 0.2,
    blur_sigma: float = 1.0,
) -> np.ndarray:
    """frames: [F, H, W] uint8. Returns [F, len(photo_map.photoreceptor_idx)]
    float32 currents, in photo_map.photoreceptor_idx order (excluded/
    unresolved photoreceptors are always exactly 0). The EMA is initialized
    with the first frame's intensity, so current[0] is exactly 0 -- no
    startup transient."""
    n_frames = len(frames)
    n_photo = len(photo_map.photoreceptor_idx)
    currents = np.zeros((n_frames, n_photo), dtype=np.float32)
    if n_frames == 0:
        return currents

    intensities = np.zeros((n_frames, n_photo), dtype=np.float64)
    for f in range(n_frames):
        blurred = gaussian_blur(frames[f].astype(np.float64) / 255.0, sigma=blur_sigma)
        intensities[f] = bilinear_sample(blurred, photo_map.image_pos)

    ema = intensities[0].copy()
    for f in range(n_frames):
        currents[f] = gain * (intensities[f] - ema)
        ema = ema_alpha * intensities[f] + (1 - ema_alpha) * ema

    currents[:, ~photo_map.included_mask] = 0.0
    return currents
