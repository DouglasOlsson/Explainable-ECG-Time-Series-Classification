import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.colors as mcolors
from matplotlib.collections import LineCollection
from matplotlib.ticker import MultipleLocator


# -----------------------------
# Labels / text
# -----------------------------
LABEL_MI = 0
LABEL_NORMAL = 1

def _get_label_text(label_idx: int) -> str:
    if label_idx == LABEL_MI:
        return "Myocardial Infarction"
    if label_idx == LABEL_NORMAL:
        return "Normal"
    return str(label_idx)


# -----------------------------
# Signal utilities
# -----------------------------
def baseline_center(signal: np.ndarray, method: str = "median") -> np.ndarray:
    s = np.asarray(signal, dtype=float)
    if method == "mean":
        return s - float(np.mean(s))
    return s - float(np.median(s))


def pad_signal_for_strip(signal, weights=None, *, pre_pad: int = 20, post_pad: int = 120):
    """
    Skapar 'slag + paus' för layout (syntetiskt).
    """
    s = np.asarray(signal, dtype=float)
    left = np.zeros(int(pre_pad), dtype=float)
    right = np.zeros(int(post_pad), dtype=float)
    s2 = np.concatenate([left, s, right])

    if weights is None:
        return s2, None

    w = np.asarray(weights, dtype=float)
    w2 = np.concatenate([np.zeros(int(pre_pad)), w, np.zeros(int(post_pad))])
    return s2, w2


# -----------------------------
# Plot styling
# -----------------------------
def add_ecg_like_grid(
    ax: plt.Axes,
    x_minor: float = 2,
    x_major: float = 10,
    y_minor: float = 0.2,
    y_major: float = 1.0,
):
    ax.xaxis.set_minor_locator(MultipleLocator(x_minor))
    ax.xaxis.set_major_locator(MultipleLocator(x_major))
    ax.yaxis.set_minor_locator(MultipleLocator(y_minor))
    ax.yaxis.set_major_locator(MultipleLocator(y_major))
    ax.grid(which="minor", linestyle="-", linewidth=0.4, alpha=0.25)
    ax.grid(which="major", linestyle="-", linewidth=0.8, alpha=0.45)


def set_symmetric_ylim(ax: plt.Axes, signal: np.ndarray, pad: float = 0.25):
    s = np.asarray(signal, dtype=float)
    m = float(np.max(np.abs(s))) if len(s) else 1.0
    ax.set_ylim(-(m + pad), (m + pad))
    ax.axhline(0, color="black", linewidth=0.9, alpha=0.6)


def _plot_colored_line(
    ax: plt.Axes,
    signal: np.ndarray,
    weights_dir: np.ndarray,
    title_prefix: str,
    prediction: int | None = None,
    ground_truth: int | None = None,
    *,
    ecg_grid: bool = True,
    grid_params=None,
    symmetric_ylim: bool = True,
):
    """
    weights_dir: DIREKTIONSVÄRDEN där:
      > 0  = driver mot MI (klass 0)  -> rött
      < 0  = driver mot Normal (klass 1) -> blått
    """
    s = np.asarray(signal, dtype=float)
    w = np.asarray(weights_dir, dtype=float)

    if len(w) != len(s):
        raise ValueError(f"weights length ({len(w)}) must match signal length ({len(s)})")

    t = np.arange(len(s))

    # Vi vill: negativ=blå, positiv=röd => använd "coolwarm" (INTE _r)
    cmap = cm.get_cmap("coolwarm")

    max_abs = float(np.max(np.abs(w))) if len(w) else 1.0
    limit = 1.0 if max_abs < 1e-9 else max_abs * 1.04
    norm = mcolors.TwoSlopeNorm(vmin=-limit, vcenter=0, vmax=limit)

    # Färgad linje
    points = np.array([t, s]).T.reshape(-1, 1, 2)
    segments = np.concatenate([points[:-1], points[1:]], axis=1)

    lc = LineCollection(segments, cmap=cmap, norm=norm)
    lc.set_array(w[:-1])
    lc.set_linewidth(2.5)
    ax.add_collection(lc)

    ax.scatter(t, s, c=w, cmap=cmap, norm=norm, s=10, zorder=10,
               edgecolor="none", alpha=0.75)

    ax.set_xlim(t.min(), t.max())
    if symmetric_ylim:
        set_symmetric_ylim(ax, s, pad=0.25)
    else:
        ax.set_ylim(float(np.min(s)) - 0.5, float(np.max(s)) + 0.5)

    if prediction is not None and ground_truth is not None:
        pred_str = _get_label_text(int(prediction))
        true_str = _get_label_text(int(ground_truth))
        status = "RÄTT" if int(prediction) == int(ground_truth) else "FEL"
        ax.set_title(
            f"{title_prefix}\nGissning: {pred_str} | Facit: {true_str}\nResultat: {status}",
            fontsize=12
        )
    else:
        ax.set_title(title_prefix)

    ax.set_xlabel("Relativ tid (index)")
    ax.set_ylabel("Relativ amplitud (normaliserad)")

    if ecg_grid:
        params = grid_params or {}
        add_ecg_like_grid(ax, **params)

    return norm


# -----------------------------
# SHAP helpers
# -----------------------------
def _extract_shap_class_vectors(shap_values, n_features: int, *, class_mi: int = 0, class_normal: int = 1):
    """
    Returnerar (w_mi, w_normal) som 1D-arrayer (n_features,)
    Försöker hantera vanliga SHAP-format:
      - list med 2 element
      - ndarray med shape (2, n) eller (1, n, 2) etc.
    """
    sv = shap_values

    # Fall 1: list -> anta index 0=MI, 1=Normal
    if isinstance(sv, list) and len(sv) >= 2:
        mi = np.asarray(sv[class_mi], dtype=float)
        no = np.asarray(sv[class_normal], dtype=float)
        mi = np.squeeze(mi)
        no = np.squeeze(no)
        # Om det finns batch-dimension
        if mi.ndim == 2 and mi.shape[0] == 1:
            mi = mi[0]
        if no.ndim == 2 and no.shape[0] == 1:
            no = no[0]
        return mi.reshape(-1)[:n_features], no.reshape(-1)[:n_features]

    # Fall 2: ndarray
    sv = np.asarray(sv, dtype=float)

    # (1, n, 2) eller (n, 2)
    if sv.ndim == 3 and sv.shape[-1] == 2:
        # batch first
        sv0 = sv[0]
        mi = sv0[:, class_mi]
        no = sv0[:, class_normal]
        return mi.reshape(-1)[:n_features], no.reshape(-1)[:n_features]

    if sv.ndim == 2 and sv.shape[-1] == 2:
        mi = sv[:, class_mi]
        no = sv[:, class_normal]
        return mi.reshape(-1)[:n_features], no.reshape(-1)[:n_features]

    # (2, n)
    if sv.ndim == 2 and sv.shape[0] == 2 and sv.shape[1] == n_features:
        mi = sv[class_mi]
        no = sv[class_normal]
        return mi.reshape(-1), no.reshape(-1)

    # Sista fallback: bara en vektor -> kan inte få båda
    v = np.squeeze(sv).reshape(-1)[:n_features]
    # Tolka som "Normal"-vektor och sätt MI=0 (du bör undvika detta genom att extrahera båda i SHAP)
    mi = np.zeros_like(v)
    no = v
    return mi, no


# -----------------------------
# LIME helpers
# -----------------------------
def _lime_pairs_to_weights(pairs, n_features: int) -> np.ndarray:
    w = np.zeros(int(n_features), dtype=float)
    for feature_idx, weight in pairs:
        try:
            idx = int(str(feature_idx).split("_")[-1])
            if 0 <= idx < n_features:
                w[idx] = float(weight)
        except (ValueError, TypeError):
            continue
    return w


def _extract_lime_weights_for_label(explanation, n_features: int, label: int) -> np.ndarray:
    """
    Försöker läsa as_list(label=label). Om label saknas kastar LIME exception.
    """
    pairs = explanation.as_list(label=int(label))
    return _lime_pairs_to_weights(pairs, n_features)


def _extract_lime_direction_weights(explanation, n_features: int) -> np.ndarray:
    """
    Direktion = w_MI - w_Normal.
    Kräver i praktiken att du har skapat explanation med labels=(0,1) eller top_labels=2.
    Har robust fallback men bästa är att säkerställa labels i explain_instance.
    """
    w_mi = None
    w_no = None

    # 1) Försök hämta båda
    try:
        w_mi = _extract_lime_weights_for_label(explanation, n_features, LABEL_MI)
    except Exception:
        w_mi = None

    try:
        w_no = _extract_lime_weights_for_label(explanation, n_features, LABEL_NORMAL)
    except Exception:
        w_no = None

    # 2) Om en saknas: försök hitta vilka labels som finns
    if (w_mi is None or w_no is None) and hasattr(explanation, "local_exp") and isinstance(explanation.local_exp, dict):
        available = sorted(list(explanation.local_exp.keys()))
        # Om bara en finns, använd den som "pred_label" approx:
        if len(available) == 1:
            only = int(available[0])
            try:
                w_only = _extract_lime_weights_for_label(explanation, n_features, only)
            except Exception:
                w_only = np.zeros(int(n_features), dtype=float)

            # Approx: om w_only är för Normal => direktion = -w_only, om för MI => +w_only
            return (w_only if only == LABEL_MI else -w_only)

    # 3) Om båda finns: korrekt direktion
    if w_mi is None:
        w_mi = np.zeros(int(n_features), dtype=float)
    if w_no is None:
        w_no = np.zeros(int(n_features), dtype=float)

    return w_mi - w_no


# -----------------------------
# Public plot functions
# -----------------------------
def plot_raw_signal(
    signal,
    prediction=None,
    ground_truth=None,
    *,
    baseline: str = "median",
    ecg_grid: bool = True,
    pause_view: bool = True,
    pre_pad: int = 30,
    post_pad: int = 80,
):
    s = np.asarray(signal, dtype=float)
    s0 = baseline_center(s, method=baseline) if baseline else s

    if pause_view:
        s_plot, _ = pad_signal_for_strip(s0, None, pre_pad=pre_pad, post_pad=post_pad)
        grid = {"x_minor": 10, "x_major": 50, "y_minor": 0.2, "y_major": 1.0}
    else:
        s_plot = s0
        grid = {"x_minor": 2, "x_major": 10, "y_minor": 0.2, "y_major": 1.0}

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(np.arange(len(s_plot)), s_plot, color="black", linewidth=2)

    if prediction is not None and ground_truth is not None:
        pred_str = _get_label_text(int(prediction))
        true_str = _get_label_text(int(ground_truth))
        status = "RÄTT" if int(prediction) == int(ground_truth) else "FEL"
        ax.set_title(
            f"Blackbox Vy (Ingen förklaring)\nGissning: {pred_str} | Facit: {true_str}\nResultat: {status}",
            fontsize=12,
        )
    else:
        ax.set_title("Raw ECG Signal")

    ax.set_xlabel("Relativ tid (index)")
    ax.set_ylabel("Relativ amplitud (normaliserad)")

    set_symmetric_ylim(ax, s_plot, pad=0.25)
    if ecg_grid:
        add_ecg_like_grid(ax, **grid)

    plt.tight_layout()
    plt.show()


def plot_shap_signal_importance(
    shap_values,
    signal,
    prediction=None,
    ground_truth=None,
    *,
    baseline: str = "median",
    pause_view: bool = True,
    pre_pad: int = 30,
    post_pad: int = 80,
):
    s = np.asarray(signal, dtype=float)
    s0 = baseline_center(s, method=baseline) if baseline else s

    w_mi, w_no = _extract_shap_class_vectors(shap_values, n_features=len(s0))
    w_dir = w_mi - w_no  # >0 = MI (rött), <0 = Normal (blått)

    if pause_view:
        s_plot, w_plot = pad_signal_for_strip(s0, w_dir, pre_pad=pre_pad, post_pad=post_pad)
        grid = {"x_minor": 10, "x_major": 50, "y_minor": 0.2, "y_major": 1.0}
    else:
        s_plot, w_plot = s0, w_dir
        grid = {"x_minor": 2, "x_major": 10, "y_minor": 0.2, "y_major": 1.0}

    fig, ax = plt.subplots(figsize=(6, 4))
    norm = _plot_colored_line(
        ax, s_plot, w_plot,
        title_prefix="SHAP Förklaring",
        prediction=prediction,
        ground_truth=ground_truth,
        ecg_grid=True,
        grid_params=grid,
        symmetric_ylim=True,
    )

    sm = cm.ScalarMappable(cmap=cm.get_cmap("coolwarm"), norm=norm)
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=ax)
    cbar.set_label("Riktning: MI (röd)  ← 0 →  Normal (blå)")
    plt.tight_layout()
    plt.show()


def plot_lime_signal_importance(
    explanation,
    signal,
    prediction=None,
    ground_truth=None,
    *,
    baseline: str = "median",
    pause_view: bool = True,
    pre_pad: int = 30,
    post_pad: int = 80,
):
    s = np.asarray(signal, dtype=float)
    s0 = baseline_center(s, method=baseline) if baseline else s

    # Direktion = w_MI - w_Normal (robust fallback om en label saknas)
    w_dir = _extract_lime_direction_weights(explanation, n_features=len(s0))

    if pause_view:
        s_plot, w_plot = pad_signal_for_strip(s0, w_dir, pre_pad=pre_pad, post_pad=post_pad)
        grid = {"x_minor": 10, "x_major": 50, "y_minor": 0.2, "y_major": 1.0}
    else:
        s_plot, w_plot = s0, w_dir
        grid = {"x_minor": 2, "x_major": 10, "y_minor": 0.2, "y_major": 1.0}

    fig, ax = plt.subplots(figsize=(6, 4))
    norm = _plot_colored_line(
        ax, s_plot, w_plot,
        title_prefix="LIME Förklaring",
        prediction=prediction,
        ground_truth=ground_truth,
        ecg_grid=True,
        grid_params=grid,
        symmetric_ylim=True,
    )

    sm = cm.ScalarMappable(cmap=cm.get_cmap("coolwarm"), norm=norm)
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=ax)
    cbar.set_label("Riktning: MI (röd)  ← 0 →  Normal (blå)")
    plt.tight_layout()
    plt.show()