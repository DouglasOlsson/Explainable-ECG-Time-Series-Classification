import numpy as np

# gets the model's prediction and probability for a given ECG sample
def get_prediction_details(model, sample_flat):
    """Hämtar modellens gissning (0 eller 1) och sannolikhet."""
    sample_3d = sample_flat.reshape(1, 96, 1)
    pred_prob = model.predict(sample_3d, verbose=0)[0][0]
    
    if pred_prob > 0.5:
        pred_label = 1 # Normal
    else:
        pred_label = 0 # Myocardial Infarction
    
    return pred_label, pred_prob

# gets human-readable label text from label index
def get_label_text(label_idx):
    """Konverterar 0/1 till läsbar text."""
    if label_idx == 0: return "Myocardial Infarction"
    if label_idx == 1: return "Normal"
    return f"Unknown ({label_idx})"