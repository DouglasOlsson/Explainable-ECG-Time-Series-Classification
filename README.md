# Explainable ECG Time Series Classification

Detta projekt fokuserar på att klassificera EKG-signaler (Normal vs. MI) med hjälp av en 1D-CNN och förklara modellens beslut med hjälp av XAI-metoder som **LIME** och **SHAP**.

## Projektstruktur
* `data/` - Moduler för att ladda EKG-data och modeller.
* `evaluation/` - Funktioner för att beräkna metrics som Fidelity och Stability.
* `explainers/` - Implementeringar av LIME och SHAP för tidsserier.
* `visualization/` - Verktyg för att plotta signaler och viktiga segment.
* `main.py` - Huvudprogrammet med ett interaktivt gränssnitt.

## Komma igång

### Förutsättningar
* Python 3.11.x
* En virtuell miljö (rekommenderas)

### Installation
1. Klona repot:
   ```bash
   git clone [https://github.com/DouglasOlsson/Explainable-ECG-Time-Series-Classification.git](https://github.com/DouglasOlsson/Explainable-ECG-Time-Series-Classification.git)
   cd ECG_XAI_TOOL

2. Skapa och aktivera en virtuell miljö:
    python3 -m venv venv
    source venv/bin/activate  # Mac/Linux
    # eller
    .\venv\Scripts\activate  # Windows

3. Installera beroenden:
    pip install -r requirements.txt


###   Användning
1. För att starta det interaktiva analysverktyget körs följande kommando:
    
    python main.py

2. Funktioner i verktyget

* Navigering genom testdata (ECG200_TEST.tsv).

* Generering av LIME-förklaringar för individuella signaler.

* Generering av SHAP-förklaringar för individuella signaler.

* Beräkning av Fidelity och Stability för att utvärdera förklaringarnas tillförlitlighet.

* Filtrering av instanser baserat på klassificeringsresultat (t.ex. visa endast felaktiga gissningar).

### Modeller och Data
1. Projektet inkluderar följande filer för att säkerställa reproducerbarhet:

* ecg_model_89acc.keras: Detta är den specifika modellinstans som användes för resultaten i uppsatsen. Den har omdöpts från det generiska namnet FINAL_BEST_ECG_MODEL.keras och sparats som en fryst version. Detta har gjorts för att förhindra att modellen skrivs över vid nya träningskörningar och för att garantera att de XAI-förklaringar som genereras i koden matchar de resultat som presenteras i studien. Det är denna fil som läses in som standard i programmet.

* ECG200_TRAIN.tsv / ECG200_TEST.tsv: Dataset för träning och utvärdering.