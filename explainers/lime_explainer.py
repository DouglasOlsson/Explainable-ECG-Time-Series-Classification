import numpy as np
from lime import lime_tabular

class ECGLime:
    """
    LIME för binär ECG-klassificering där modellens sigmoid p = P(Normal).
    Wrappern bygger [P(MI), P(Normal)] = [1-p, p].

    Label-index:
      0 = MI / Ischemia
      1 = Normal
    """

    def __init__(self, X_train_flat, class_names=None):
        self.feature_names = [f"Time_{i}" for i in range(X_train_flat.shape[1])]

        # Viktigt: måste matcha wrapperns outputordning [MI, Normal]
        self.class_names = class_names if class_names is not None else ["Ischemia", "Normal"]

        self.explainer = lime_tabular.LimeTabularExplainer(
            training_data=X_train_flat,
            feature_names=self.feature_names,
            class_names=self.class_names,
            mode="classification",
            discretize_continuous=False,
        )

    def explain(self, model, instance_flat, num_features=96):
        instance_flat = np.asarray(instance_flat).reshape(-1)

        def predict_wrapper(data_flat):
            data_flat = np.asarray(data_flat)
            data_3d = data_flat.reshape((data_flat.shape[0], 96, 1))
            p_normal = model.predict(data_3d, verbose=0)

            # Keras sigmoid kan vara shape (n,1) eller (n,)
            p_normal = p_normal.reshape(-1, 1)

            # Bygg [P(MI), P(Normal)]
            preds = np.hstack((1.0 - p_normal, p_normal))
            return preds

        explanation = self.explainer.explain_instance(
            data_row=instance_flat,
            predict_fn=predict_wrapper,
            num_features=num_features,
            labels=(0, 1),          # <-- NYCKELN: beräkna båda labels => ingen crash vid label=0
            # alternativ: top_labels=2
        )
        return explanation