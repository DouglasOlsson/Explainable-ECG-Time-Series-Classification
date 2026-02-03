# metrics.py

import numpy as np
import time
from scipy.spatial.distance import cosine
from sklearn.metrics import (
    accuracy_score, f1_score, confusion_matrix, precision_score, recall_score
)

K_VALUES = [5, 10, 20]


# -----------------------------
# Helper: predicted class
# -----------------------------

def predicted_class_from_model(model, sample_flat, threshold=0.5):
    """
    Model output: p = P(class=1=Normal).
    predicted_class = 1 om p > threshold annars 0 (MI).
    """
    sample_flat = np.asarray(sample_flat).flatten()
    p_class1 = float(model.predict(sample_flat.reshape(1, sample_flat.size, 1), verbose=0)[0][0])
    predicted_class = 1 if p_class1 > threshold else 0
    return predicted_class, p_class1


# -----------------------------
# Helpers: extract weights
# -----------------------------

def extract_lime_weights(explanation, num_features=96, class_idx=None):
    """
    Extraherar 1D-vektor (num_features,) från LIME explanation.

    - Om class_idx anges försöker vi hämta explanation.as_list(label=class_idx)
      för klass-specifik förklaring.
    - Om label saknas faller vi tillbaka på explanation.as_list() för att undvika crash.

    OBS: För att detta ska vara robust för label=0 måste LIME skapas med labels=(0,1)
    i explain_instance. (Annars kan local_exp sakna label 0.)
    """
    weights = np.zeros(int(num_features), dtype=float)

    if class_idx is not None:
        try:
            pairs = explanation.as_list(label=int(class_idx))
        except Exception:
            pairs = explanation.as_list()
    else:
        pairs = explanation.as_list()

    for feature_idx, weight in pairs:
        try:
            # feature_idx brukar vara "Time_42"
            idx = int(str(feature_idx).split("_")[-1])
            if 0 <= idx < num_features:
                weights[idx] = float(weight)
        except (ValueError, TypeError):
            continue

    return weights


def extract_shap_weights(shap_out, class_idx):
    """
    Returnerar 1D-vektor (96,) för SHAP-attributioner för vald klass.

    Hanterar vanligt förekommande format från KernelExplainer:
      - list: [vals_class0, vals_class1], där varje typiskt är (n,96)
      - ndarray: (n,96), (n,96,2), (96,), (96,2), (1,96), (1,96,2)
    """
    v = shap_out
    class_idx = int(class_idx)

    if isinstance(v, list):
        v = v[class_idx]  # typiskt (n,96)

    v = np.asarray(v)

    # Exempel: (n,96,2) eller (1,96,2)
    if v.ndim == 3:
        # Första instansen
        v0 = v[0]
        # Kan vara (96,2) eller (96,?) -> välj klasskolumn om det är 2
        if v0.ndim == 2 and v0.shape[1] >= 2:
            v = v0[:, class_idx]
        else:
            # fallback: platta ut första instansen
            v = v0.squeeze()

    # Exempel: (n,96) eller (1,96)
    elif v.ndim == 2:
        # Om (96,2) -> välj klasskolumn
        if v.shape[1] >= 2 and v.shape[0] == 96:
            v = v[:, class_idx]
        else:
            # anta (n,96) -> ta första raden
            v = v[0]

    elif v.ndim == 1:
        pass

    else:
        raise ValueError(f"Unsupported SHAP output shape: {v.shape}")

    v = np.squeeze(v).astype(float)

    return v


# -----------------------------
# Metrics: Fidelity & Stability
# -----------------------------

def calculate_fidelity(model, sample_flat, weights, k_values=K_VALUES, threshold=0.5):
    """
    Fidelity via Probability Drop (faithfulness).

    1) välj predikterad klass y_hat från modellen
    2) beräkna P(y_hat|x)
    3) maska top-k enligt |weights| med mean-of-selected
    4) beräkna P(y_hat|x_masked)
    5) fidelity_k = P(y_hat|x) - P(y_hat|x_masked)
    """
    x = np.asarray(sample_flat).flatten()
    w = np.asarray(weights).flatten()

    if x.size != w.size:
        raise ValueError(f"sample length ({x.size}) must match weights length ({w.size})")

    y_hat, p_class1 = predicted_class_from_model(model, x, threshold=threshold)
    p_y_x = p_class1 if y_hat == 1 else (1.0 - p_class1)

    # sortera på absolut vikt
    sorted_idx = np.argsort(np.abs(w))  # stigande
    out = {}

    for k in k_values:
        k_eff = int(min(max(1, k), x.size))
        top_idx = sorted_idx[-k_eff:]

        x_masked = x.copy()
        mean_val = float(np.mean(x[top_idx]))
        x_masked[top_idx] = mean_val

        p_masked_class1 = float(model.predict(x_masked.reshape(1, x.size, 1), verbose=0)[0][0])
        p_y_x_masked = p_masked_class1 if y_hat == 1 else (1.0 - p_masked_class1)

        out[k] = p_y_x - p_y_x_masked

    return out


def calculate_stability(model, sample_flat, original_weights, explainer_func, noise_level=0.005):
    """
    Stability via cosine similarity mellan original och perturberad förklaring.
    """
    x = np.asarray(sample_flat).flatten()
    w0 = np.asarray(original_weights).flatten()

    noise = np.random.normal(0, noise_level, x.shape)
    x_noisy = x + noise

    w1 = np.asarray(explainer_func(x_noisy)).flatten()

    if x.size != w0.size or x.size != w1.size:
        raise ValueError("sample/weights length mismatch in stability calculation")

    if np.all(w0 == 0) or np.all(w1 == 0):
        return 0.0

    return 1 - cosine(w0, w1)


# -----------------------------
# Model evaluation (classic metrics)
# -----------------------------

def evaluate_whole_dataset(model, X_test, y_test, batch_size=32, threshold=0.5):
    print("\n" + "=" * 60)
    print("   UTVÄRDERING AV MODELL (FOKUS: MI/ISCHEMIA = 0)   ")
    print("=" * 60)

    X_test = np.asarray(X_test)
    y_test = np.asarray(y_test).flatten()

    # matcha input-shape (n,96,1)
    X_test_3d = X_test.reshape(X_test.shape[0], X_test.shape[1], 1) if X_test.ndim == 2 else X_test

    # p för klass 1 (Normal)
    y_pred_probs = model.predict(X_test_3d, batch_size=batch_size, verbose=0).flatten()
    y_pred = (y_pred_probs > threshold).astype(int)

    acc = accuracy_score(y_test, y_pred)

    # pos_label=0 eftersom 0 = MI/Ischemia (sjukdom)
    precision = precision_score(y_test, y_pred, pos_label=0, zero_division=0)
    recall = recall_score(y_test, y_pred, pos_label=0, zero_division=0)
    f1 = f1_score(y_test, y_pred, pos_label=0, average="binary", zero_division=0)

    cm = confusion_matrix(y_test, y_pred, labels=[0, 1])
    tp_mi = cm[0, 0]
    fn_mi = cm[0, 1]
    fp_mi = cm[1, 0]
    tn_normal = cm[1, 1]

    print("-" * 60)
    print("Confusion Matrix (0 = MI/Ischemia):")
    print(f"  TP (MI):        {tp_mi}")
    print(f"  FN (MI):        {fn_mi}  <-- Missade infarkter")
    print(f"  FP (MI):        {fp_mi}")
    print(f"  TN (Normal):    {tn_normal}")
    print("-" * 60)
    print(f"ACCURACY:   {acc:.4f} ({acc*100:.2f}%)")
    print(f"PRECISION:  {precision:.4f} (Träffsäkerhet på MI)")
    print(f"RECALL:     {recall:.4f} (Känslighet för MI)")
    print(f"F1-SCORE:   {f1:.4f}")
    print("-" * 60)

    input("Tryck Enter för att återgå...")


# -----------------------------
# Full XAI metrics run (LIME & SHAP)
# -----------------------------

def run_metrics_on_all_data(model, X_test, lime_tool, shap_tool, threshold=0.5):
    print("\n" + "=" * 95)
    print("   KÖR FULLSTÄNDIG UTVÄRDERING (LIME & SHAP) MED k ∈ {5, 10, 20}")
    print("=" * 95)
    print("Detta kommer ta en stund. Var god vänta...\n")

    X_test = np.asarray(X_test)
    k_values = list(K_VALUES)
    n_samples = len(X_test)

    results = {
        "lime_fid": {k: [] for k in k_values},
        "shap_fid": {k: [] for k in k_values},
        "lime_stab": [],
        "shap_stab": [],
    }

    start_time = time.time()

    for i in range(n_samples):
        sample = np.asarray(X_test[i]).flatten()
        n_steps = sample.size

        # Predikterad klass per instans (0=MI, 1=Normal)
        predicted_class, _ = predicted_class_from_model(model, sample, threshold=threshold)

        # -----------------
        # 1) LIME
        # -----------------
        exp = lime_tool.explain(model, sample)  # förutsätt labels=(0,1) i lime_tool
        w_lime = extract_lime_weights(exp, num_features=n_steps, class_idx=predicted_class)

        fid_scores_lime = calculate_fidelity(model, sample, w_lime, k_values=k_values, threshold=threshold)
        for k in k_values:
            results["lime_fid"][k].append(fid_scores_lime[k])

        def lime_wrapper(s):
            e = lime_tool.explain(model, np.asarray(s).flatten())
            return extract_lime_weights(e, num_features=n_steps, class_idx=predicted_class)

        results["lime_stab"].append(
            calculate_stability(model, sample, w_lime, lime_wrapper)
        )

        # -----------------
        # 2) SHAP  (klass per instans = predikterad klass)
        # -----------------
        shap_out = shap_tool.explain_batch(sample.reshape(1, n_steps))
        w_shap = extract_shap_weights(shap_out, class_idx=predicted_class)

        fid_scores_shap = calculate_fidelity(model, sample, w_shap, k_values=k_values, threshold=threshold)
        for k in k_values:
            results["shap_fid"][k].append(fid_scores_shap[k])

        def shap_wrapper(s):
            s = np.asarray(s).flatten()
            shap_noisy_out = shap_tool.explain_batch(s.reshape(1, n_steps))
            # håll samma target class (predicted_class) när du jämför original vs noisy
            return extract_shap_weights(shap_noisy_out, class_idx=predicted_class)

        results["shap_stab"].append(
            calculate_stability(model, sample, w_shap, shap_wrapper)
        )

        if (i + 1) % 5 == 0:
            elapsed = time.time() - start_time
            print(f"Bearbetat {i + 1}/{n_samples} prover... ({elapsed:.1f}s)")

    # -----------------
    # Presentation
    # -----------------
    print("\n" + "=" * 110)
    print("                           SLUTRESULTAT FÖR RESULTATAVSNITT 4.2")
    print("=" * 110)

    header = (
        f"{'METOD':<8} | {'STABILITY':<18} | {'FIDELITY (k=5)':<18} | "
        f"{'FIDELITY (k=10)':<18} | {'FIDELITY (k=20)':<18}"
    )
    print(header)
    print("-" * 110)

    for method_name in ["LIME", "SHAP"]:
        m_key = method_name.lower()

        stab_m = float(np.mean(results[f"{m_key}_stab"]))
        stab_s = float(np.std(results[f"{m_key}_stab"]))

        row = f"{method_name:<8} | {stab_m:.4f} ± {stab_s:.4f}"

        for k in k_values:
            fid_m = float(np.mean(results[f"{m_key}_fid"][k]))
            fid_s = float(np.std(results[f"{m_key}_fid"][k]))
            row += f" | {fid_m:.4f} ± {fid_s:.4f}"

        print(row)

    print("-" * 110)
    print(f"Fidelity (Probability Drop) mätt på {n_samples} test-prover.")
    print("Stability mätt via Cosine Similarity med Gaussian noise.")
    print("=" * 110)
    input("Tryck Enter för att återgå till huvudmenyn...")