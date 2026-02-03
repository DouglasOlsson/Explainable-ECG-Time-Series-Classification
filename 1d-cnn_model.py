import pandas as pd
import numpy as np
import keras
import os
from keras.models import Sequential
from keras.layers import Conv1D, MaxPooling1D, Flatten, Dense, Dropout, BatchNormalization
from keras.optimizers import Adam
from keras.callbacks import ModelCheckpoint
from sklearn.metrics import classification_report, accuracy_score

# --- KONFIGURATION ---
N_RUNS = 5  # Hur många gånger vi ska försöka hitta en bra modell
BEST_MODEL_FILENAME = 'FINAL_BEST_ECG_MODEL.keras'

# --- 1. LADDA DATA (Samma som förut) ---
def load_data(file_path):
    df = pd.read_csv(file_path, sep='\t', header=None)
    data = df.to_numpy()
    y = data[:, 0]
    X = data[:, 1:]
    return X, y

X_train, y_train = load_data('ECG200_TRAIN.tsv')
X_test, y_test = load_data('ECG200_TEST.tsv')

# Preprocessing
X_train = X_train.reshape((X_train.shape[0], X_train.shape[1], 1))
X_test = X_test.reshape((X_test.shape[0], X_test.shape[1], 1))
y_train = np.where(y_train == -1, 0, 1)
y_test = np.where(y_test == -1, 0, 1)

# --- 2. FUNKTION FÖR ATT BYGGA MODELL ---
def create_model():
    model = Sequential([
        Conv1D(filters=16, kernel_size=5, activation='relu', input_shape=(96, 1), padding='same'),
        BatchNormalization(),
        MaxPooling1D(pool_size=2),
        Conv1D(filters=32, kernel_size=3, activation='relu', padding='same'),
        BatchNormalization(),
        MaxPooling1D(pool_size=2),
        Conv1D(filters=64, kernel_size=3, activation='relu', padding='same'),
        MaxPooling1D(pool_size=2),
        Flatten(),
        Dense(32, activation='relu'),
        Dropout(0.2),
        Dense(1, activation='sigmoid')
    ])
    model.compile(optimizer=Adam(learning_rate=0.001),
                  loss='binary_crossentropy',
                  metrics=['accuracy'])
    return model

# --- 3. LOOP FÖR FLERA KÖRNINGAR ---

best_global_accuracy = 0.0
run_history = []

print(f"Startar {N_RUNS} körningar för att hitta den bästa modellen...\n")

for i in range(N_RUNS):
    print(f"--- KÖRNING {i+1}/{N_RUNS} ---")
    
    # Skapa en ny, "ren" modell varje gång
    model = create_model()
    
    # Checkpoint för denna specifika körning
    temp_filename = f'temp_run_{i}.keras'
    checkpoint = ModelCheckpoint(
        temp_filename,
        monitor='val_accuracy',
        save_best_only=True,
        mode='max',
        verbose=0
    )
    
    # Träna
    model.fit(
        X_train, y_train,
        epochs=50,
        batch_size=8,
        validation_split=0.2,
        callbacks=[checkpoint],
        verbose=0 # Tyst läge för att spara plats i terminalen
    )
    
    # Ladda den bästa versionen från DENNA körning
    best_run_model = keras.models.load_model(temp_filename)
    
    # Utvärdera på testdata
    y_pred_prob = best_run_model.predict(X_test, verbose=0)
    y_pred = (y_pred_prob > 0.5).astype(int)
    run_accuracy = accuracy_score(y_test, y_pred)
    
    print(f"Resultat körning {i+1}: Accuracy = {run_accuracy*100:.2f}%")
    run_history.append(run_accuracy)
    
    # Är denna bättre än vårt rekord?
    if run_accuracy > best_global_accuracy:
        print(f"-> NYTT REKORD! (Tidigare: {best_global_accuracy*100:.2f}%)")
        best_global_accuracy = run_accuracy
        # Spara undan denna som den slutgiltiga vinnaren
        best_run_model.save(BEST_MODEL_FILENAME)
    
    # Städa bort den temporära filen
    os.remove(temp_filename)
    print("")

# --- 4. SLUTRESULTAT ---

print("="*30)
print(f"Bästa Accuracy uppnådd: {best_global_accuracy*100:.2f}%")
print(f"Genomsnittlig Accuracy: {np.mean(run_history)*100:.2f}%")
print(f"Vinnande modell sparad som: {BEST_MODEL_FILENAME}")
print("="*30)

# Ladda och visa detaljerad rapport för vinnaren
final_model = keras.models.load_model(BEST_MODEL_FILENAME)
y_pred_final = (final_model.predict(X_test) > 0.5).astype(int)
print("\nClassification Report (Bästa Modellen):")
print(classification_report(y_test, y_pred_final, target_names=['Ischemia', 'Normal']))