import sys
import numpy as np
import os

# Egna moduler
from data.data_loader import load_data
from data.model_loader import load_ecg_model
from explainers.lime_explainer import ECGLime
from explainers.shap_explainer import ECGShap
from utils.helpers import get_prediction_details, get_label_text
from evaluation.metrics import (
    evaluate_whole_dataset,
    run_metrics_on_all_data,
    calculate_fidelity,
    calculate_stability,
    extract_lime_weights,
    extract_shap_weights,
)
from visualization import plot_utils

class ECGExplainerApp:
    def __init__(self):
        # Data & Model
        self.model = None
        self.X_train, self.y_train = None, None
        self.X_test, self.y_test = None, None
        
        # XAI Tools
        self.lime_tool = None
        self.shap_tool = None
        
        # State
        self.current_idx = 0
        self.filters = {"normal": [], "mi": [], "wrong": []}

    def startup(self):
        """Laddar in allt nödvändigt vid start."""
        print("\n" + "="*55)
        print("   ECG EXPLAINER TOOL - REFACTORED   ")
        print("="*55)

        print("\n[1/3] Laddar data...")
        self.X_train, self.y_train = load_data("ECG200_TRAIN.tsv")
        self.X_test, self.y_test = load_data("ECG200_TEST.tsv")
        
        # Konvertera labels (-1, 1) -> (0, 1)
        self.y_train = np.where(self.y_train == -1, 0, 1)
        self.y_test = np.where(self.y_test == -1, 0, 1)
        
        print("[2/3] Laddar modell...")
        self.model = load_ecg_model("ecg_model_89acc.keras")

        print("[3/3] Initierar XAI-verktyg...")
        self.lime_tool = ECGLime(self.X_train) 
        self.shap_tool = ECGShap(self.model, self.X_train[:20])
        
        self._prepare_filters()
        print("\nKlar! Redo att köra.")

    def _prepare_filters(self):
        """Kategoriserar datasetet för att snabbt kunna hitta specifika fall."""
        print("Analyserar datasetet för filtrering...")
        X_test_3d = self.X_test.reshape(self.X_test.shape[0], 96, 1)
        all_probs = self.model.predict(X_test_3d, verbose=0)
        all_preds = (all_probs > 0.5).astype(int).flatten()
        
        self.filters["normal"] = [i for i in range(len(self.y_test)) if self.y_test[i] == 1 and all_preds[i] == 1]
        self.filters["mi"] = [i for i in range(len(self.y_test)) if self.y_test[i] == 0 and all_preds[i] == 0]
        self.filters["wrong"] = [i for i in range(len(self.y_test)) if self.y_test[i] != all_preds[i]]
        
        print(f"Indexering klar: {len(self.filters['normal'])} Normal, {len(self.filters['mi'])} MI, {len(self.filters['wrong'])} Fel.")

    def show_current_status(self):
        """Visar info om det aktuella samplet."""
        sample_flat = self.X_test[self.current_idx]
        true_label = int(self.y_test[self.current_idx])
        
        pred_label, pred_prob = get_prediction_details(self.model, sample_flat)
        confidence = pred_prob if pred_label == 1 else (1 - pred_prob)
        status = "RÄTT" if pred_label == true_label else "FEL"
        
        print("\n" + "-"*55)
        print(f"SAMPLE:       {self.current_idx + 1} / {len(self.X_test)}")
        print(f"FACIT:        {get_label_text(true_label)} ({true_label})")
        print(f"MODELL:       {get_label_text(pred_label)} (Säkerhet: {confidence * 100:.1f}%)")
        print(f"STATUS:       {status}")
        print("-"*55)

    def calculate_sample_metrics(self):
        """Beräknar fidelity och stabilitet för nuvarande sample."""
        print(f"\n--- Metrics för Sample {self.current_idx + 1} ---")
        sample_flat = self.X_test[self.current_idx]
        pred_label, _ = get_prediction_details(self.model, sample_flat)

        # LIME Metrics
        exp = self.lime_tool.explain(self.model, sample_flat)
        w_lime = extract_lime_weights(exp, num_features=96, class_idx=pred_label)
        f_lime = calculate_fidelity(self.model, sample_flat, w_lime)
        s_lime = calculate_stability(self.model, sample_flat, w_lime, 
                                     lambda s: extract_lime_weights(self.lime_tool.explain(self.model, s), 96, pred_label))

        # SHAP Metrics
        shap_out = self.shap_tool.explain_batch(sample_flat.reshape(1, 96))
        w_shap = extract_shap_weights(shap_out, class_idx=pred_label)
        f_shap = calculate_fidelity(self.model, sample_flat, w_shap)
        s_shap = calculate_stability(self.model, sample_flat, w_shap, 
                                     lambda s: extract_shap_weights(self.shap_tool.explain_batch(s.reshape(1, 96)), pred_label))

        self._print_metrics_table(s_lime, f_lime, s_shap, f_shap)

    def _print_metrics_table(self, s_l, f_l, s_s, f_s):
        """Hjälpmetod för att skriva ut den snygga tabellen."""
        print("-" * 85)
        print(f"{'METOD':<8} | {'STAB.':<8} | {'FID (k=5)':<12} | {'FID (k=10)':<12} | {'FID (k=20)':<12}")
        print("-" * 85)
        print(f"LIME     | {s_l:.4f} | {f_l[5]:.4f}     | {f_l[10]:.4f}      | {f_l[20]:.4f}")
        print(f"SHAP     | {s_s:.4f} | {f_s[5]:.4f}     | {f_s[10]:.4f}      | {f_s[20]:.4f}")
        print("-" * 85)
        input("Tryck Enter för att återgå...")

    def handle_filter_menu(self):
        """Undermeny för att filtrera samples."""
        print("\n--- FILTRERA INSTANSER ---")
        print(" [1] Normal (Rätt klassade)")
        print(" [2] MI (Rätt klassade)")
        print(" [3] Visa alla FEL")
        val = input("Val: ").strip()
        
        mapping = {'1': 'normal', '2': 'mi', '3': 'wrong'}
        key = mapping.get(val)
        
        if key and self.filters[key]:
            self.current_idx = np.random.choice(self.filters[key])
            print(f"Hämtade sample {self.current_idx + 1}.")
        else:
            print("Hittade inga matchande fall.")

    def run(self):
        """Huvudloop."""
        self.startup()
        
        while True:
            self.show_current_status()
            print(" [e] Dataset Acc | [a] Kör All Metrics | [f] Filtrera | [m] Sample Metrics")
            print(" [1] LIME Plot   | [2] SHAP Plot      | [3] RAW Plot")
            print(" [n] Nästa       | [r] Slumpa         | [i] Gå till nr | [q] Avsluta")
            
            choice = input("\n>> ").strip().lower()
            sample_flat = self.X_test[self.current_idx]
            pred_label, _ = get_prediction_details(self.model, sample_flat)
            true_label = int(self.y_test[self.current_idx])

            if choice == 'q':
                print("Avslutar...")
                break
            
            # --- DISPATCHER LOGIK ---
            if choice == 'e': evaluate_whole_dataset(self.model, self.X_test, self.y_test)
            elif choice == 'a': run_metrics_on_all_data(self.model, self.X_test, self.lime_tool, self.shap_tool)
            elif choice == 'f': self.handle_filter_menu()
            elif choice == 'm': self.calculate_sample_metrics()
            elif choice == '1':
                exp = self.lime_tool.explain(self.model, sample_flat)
                plot_utils.plot_lime_signal_importance(exp, sample_flat, pred_label, true_label, pause_view=True)
            elif choice == '2':
                v = self.shap_tool.explain_batch(sample_flat.reshape(1, 96))
                plot_utils.plot_shap_signal_importance(v, sample_flat, pred_label, true_label, pause_view=True)
            elif choice == '3':
                plot_utils.plot_raw_signal(sample_flat, pred_label, true_label, pause_view=True)
            elif choice == 'n':
                self.current_idx = (self.current_idx + 1) % len(self.X_test)
            elif choice == 'r':
                self.current_idx = np.random.randint(0, len(self.X_test))
            elif choice == 'i':
                try:
                    idx = int(input(f"Ange 1-{len(self.X_test)}: ")) - 1
                    if 0 <= idx < len(self.X_test): self.current_idx = idx
                except ValueError: pass

if __name__ == "__main__":
    try:
        app = ECGExplainerApp()
        app.run()
    except KeyboardInterrupt:
        sys.exit(0)