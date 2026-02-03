import numpy as np
import shap

# SHAP Explainer for ECG Data
class ECGShap:
    # Initialize the SHAP explainer with model and background data
    def __init__(self, model, background_data):
        self.model = model

        # Wrapper function to adapt model prediction for SHAP
        def predict_wrapper(data_flat):
            # Reshape: (n_samples, 96) -> (n_samples, 96, 1)
            data_3d = data_flat.reshape((data_flat.shape[0], 96, 1))
            preds = self.model.predict(data_3d, verbose=0) # verbose=0 för att slippa Keras-loggar

            # Hantera binär klassificering (1 output neuron)
            if preds.shape[1] == 1:
                preds = np.hstack((1 - preds, preds))

            return preds
        # Initialize SHAP Kernel Explainer
        self.explainer = shap.KernelExplainer(predict_wrapper, background_data)

    # Explain a batch of ECG instances
    def explain_batch(self, data_batch, nsamples="auto"):
        # Generate SHAP values for the batch
        shap_values = self.explainer.shap_values(data_batch, nsamples=nsamples)
        return shap_values