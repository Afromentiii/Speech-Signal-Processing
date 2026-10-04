# Speech Signal Processing: Klasyfikacja języka mowy (Grecki vs Serbski)

Projekt klasyfikacji języka mowy (Spoken Language Identification – LID) dla nagrań języka greckiego i serbskiego z bazy Mozilla Common Voice. Implementacja wykorzystuje metody cyfrowego przetwarzania sygnałów mowy (DSP), ekstrakcję cech akustycznych, rytmicznych i prozodycznych oraz modele uczenia maszynowego (XGBoost z akceleracją CUDA GPU oraz regularizowany SVM RBF).

---

## 1. Zbiory danych

Do eksperymentów wykorzystano publiczne zbiory nagrań Mozilla Common Voice:
- [Język serbski (Mozilla Common Voice)](https://mozilladatacollective.com/datasets/cmu5thgz30067nq07e6rsuwsx)
- [Język grecki (Mozilla Common Voice)](https://mozilladatacollective.com/datasets/cmu62fru800nko107kvyi4hrc)

Nagrania powinny znajdować się w katalogach `Greek/` oraz `Serbian/` w formacie `.mp3` w głównym folderze projektu (katalogi te są wykluczone z repozytorium w `.gitignore`).

---

## 2. Architektura i struktura projektu

Projekt zrealizowano w architekturze modułowej zgodnie z zasadą separacji logiki od prezentacji. Notatniki Jupyter pełnią wyłącznie funkcję potoku uruchomieniowego (pipeline), a cała logika biznesowa, przetwarzanie sygnałów i modelowanie znajdują się w dedykowanym pakiecie `src/`.

```text
Speech-Signal-Processing/
├── src/
│   ├── __init__.py              # Eksport publicznego API pakietu
│   ├── audio_processing.py      # Filtry Butterwortha (75 Hz), audyt ciszy (VAD), segmentacja
│   ├── feature_extraction.py    # Obwiednia Hilberta, statystyki, widmo modulacji rytmu (1–8 Hz)
│   ├── spectral_features.py     # Mel-spektrogramy, 13 MFCC + Δ + ΔΔ, pooling czasowy, prozodia (F0, BPM)
│   ├── data_balancing.py        # Balansowanie mówców i czasu mowy, rozłączny podział (speaker-stratified split)
│   ├── modeling.py              # Modele (XGBoost GPU, SVM), grupowa walidacja (StratifiedGroupKFold), multi-seed
│   └── visualization.py         # Wykresy przebiegów, spektrogramów, ważności cech, stabilności CV i ziaren
├── SpeechML.ipynb               # Potok 1: obwiednia sygnału + widmo modulacji rytmu + model XGBoost (GPU)
├── SpeechSpectrogramSVM.ipynb   # Potok 2: spektrogramy MFCC + cechy prozodyczne + regularizowany SVM (RBF)
├── FEATURES.md                  # Szczegółowa dokumentacja matematyczna, jednostki i uzasadnienie fonetyczne cech
├── README.md                    # Dokumentacja projektu, architektura i podsumowanie wyników
├── requirements.txt             # Lista zależności pakietów Python
└── .gitignore                   # Wykluczenie danych audio, środowiska wirtualnego i plików roboczych
```

### Role modułów w `src/`:
- **`audio_processing.py`**: Wczytywanie plików audio z próbkowaniem 32 kHz, filtr górnoprzepustowy Butterwortha 75 Hz (usunięcie składowej stałej DC i przydźwięku sieciowego), audyt ciszy i detekcja aktywności głosowej (VAD) z progiem RMS 0.002, cięcie na fragmenty 1.0 s.
- **`feature_extraction.py`**: Analityczna transformata Hilberta, statystyki obwiedni (percentyle p10..p90, RMS, skośność, kurtoza) oraz analiza modulacji pasm rytmu mowy (1–8 Hz) przez FFT z okienkiem Hanna.
- **`spectral_features.py`**: Bank 13 filtrów MFCC z pochodnymi dynamicznymi ($\Delta, \Delta\Delta$), pooling czasowy (średnia i odchylenie standardowe) oraz cechy prozodyczne (estymacja $F_0$ algorytmem YIN w paśmie 60–400 Hz, tempo artykulacji BPM, współczynnik pauz do fonacji).
- **`data_balancing.py`**: Wyrównanie liczby próbek na mówcę (maks. 3) oraz łącznego czasu mowy per klasa; rozłączny podział mówców (`stratified_speaker_train_test_split`) zapobiegający wyciekowi danych.
- **`modeling.py`**: Konfiguracja klasyfikatorów (XGBoost GPU z regularyzacją, SVM RBF ze skalowaniem i PCA), 5-krotna grupowa walidacja krzyżowa (`StratifiedGroupKFold`), analiza luki generalizacji oraz ewaluacja stabilności wieloziarnowej (`run_multiseed_experiment`).
- **`visualization.py`**: Wykresy porównawcze mówców, spektrogramów Mel i MFCC, macierzy pomyłek, krzywych ROC, ważności cech oraz stabilności metryk między foldami i ziarnami losowości.

---

## 3. Zastosowane technologie i akceleracja sprzętowa

### Biblioteki i środowisko:
- **Język:** Python 3.13
- **Przetwarzanie sygnałów (DSP):** `librosa`, `scipy.signal`, `soundfile`
- **Uczenie maszynowe:** `scikit-learn`, `xgboost`, `optuna`
- **Przetwarzanie danych i tabele:** `numpy`, `pandas`, `pyarrow`
- **Wizualizacja:** `matplotlib`, `seaborn`

### Podział obliczeń (GPU vs CPU):
| Komponent | Środowisko wykonawcze | Szczegóły implementacji |
| :--- | :---: | :--- |
| **Model XGBoost (`SpeechML.ipynb`)** | **GPU (CUDA)** | Wykorzystuje `device='cuda'` oraz `tree_method='hist'`. Budowa histogramów i drzew decyzyjnych zachodzi bezpośrednio na GPU NVIDIA. |
| **Model SVM (`SpeechSpectrogramSVM.ipynb`)** | **CPU** | Estymator `sklearn.svm.SVC` oparty jest na silniku LIBSVM w C++ i wykonuje się na CPU. Zrównoleglenie strojenia realizowane jest wieloprocesowo przez `joblib`. |
| **Ekstrakcja cech audio (`librosa`)** | **CPU (Wielowątkowość)** | Wyznaczanie FFT, MFCC oraz tonu krtaniowego $F_0$ (YIN) zrównoleglono na 8 wątkach (`ThreadPoolExecutor`). |

---

## 4. Schemat potoku przetwarzania (Pipeline)

```text
+-----------------------------------------------------------------------------------+
|                        POTOK PRZETWARZANIA SYGNAŁU AUDIO                          |
+-----------------------------------------------------------------------------------+
                                         |
                                         v
               [Nagrania wejściowe MP3] (Mozilla Common Voice: EL / SR)
                                         |
                                         v
       [Filtr górnoprzepustowy] (Butterworth 75 Hz, usunięcie składowej stałej DC)
                                         |
                                         v
        [Audyt ciszy / VAD] (Odrzucenie nagrań cichych i uszkodzonych, RMS < 0.002)
                                         |
                                         v
             [Segmentacja okienkowa] (Pocięcie na równe fragmenty 1.0 s)
                                         |
     +-----------------------------------+------------------------------------+
     |                                                                        |
     v (Potok A: SpeechML)                                                    v (Potok B: SpeechSpectrogramSVM)
[Ekstrakcja rytmu i obwiedni]                                [Ekstrakcja cech spektralnych i prozodii]
 - Obwiednia Hilberta                                         - Bank 13 filtrów MFCC + Δ + ΔΔ
 - Statystyki i percentyle (p10..p90)                         - Pooling czasowy (średnia i odch. std)
 - Widmo modulacji FFT w pasmach 1–8 Hz                       - Prozodia: F0 (pitch), proporcja pauz, BPM
     |                                                                        |
     +-----------------------------------+------------------------------------+
                                         |
                                         v
                 [Balansowanie mówców i łącznego czasu mowy]
                  - Ograniczenie próbek per pojedynczy mówca (maks. 3)
                  - Wyrównanie liczby fragmentów i sekund per klasa
                                         |
                                         v
               [Rozłączny podział mówców (Stratified Speaker Split)]
                - Zbiór treningowy (75%) vs Zbiór testowy (25%)
                - Gwarancja braku tego samego mówcy w obu zbiorach (brak wycieku)
                                         |
     +-----------------------------------+------------------------------------+
     |                                                                        |
     v (XGBoost GPU / Random Forest)                                          v (SVM RBF z regularyzacją)
[Klasyfikacja i analiza ważności]                            [Klasyfikacja i strojenie]
 - Strojenie wag i parametrów drzew                           - Grid Search z walidacją grupową (C, gamma)
 - Ważność cech i odrzucenie szumu                            - Skalowanie cech i redukcja wymiarowości PCA
     |                                                                        |
     +-----------------------------------+------------------------------------+
                                         |
                                         v
                    [Walidacja krzyżowa i ocena stabilności]
                     - 5-krotna grupowa walidacja (StratifiedGroupKFold)
                     - Analiza luki generalizacji (Train - Val Gap)
                     - Multi-seed evaluation (badanie wariancji dla ziaren 42, 100, 2024, 7, 999)
```

---

## 5. Metodologia i odporność na zniekształcenia akustyczne

### Eliminacja wycieku tożsamości mówcy (Data Leakage)
W nagraniach crowdsourcingowych modele uczenia maszynowego łatwo zapamiętują barwę głosu konkretnego człowieka zamiast cech języka. Aby temu zapobiec:
- Zbiory testowe i foldy walidacyjne tworzone są za pomocą `StratifiedGroupKFold` z parametrem `groups=speaker`.
- W zbiorze testowym znajduje się **972 całkowicie innych mówców** niż w treningu (2916 mówców).

### Wpływ mikrofonu i środowiska akustycznego (Channel Mismatch)
- **Cechy podatne na tor audio:** Statyczne poziomy energii (`rms`, `p_10`, `p_25`) mogą częściowo odzwierciedlać szum tła otoczenia nagraniowego.
- **Cechy odporne na mikrofon:**
  - **Pochodne dynamiczne ($\Delta$ i $\Delta\Delta$ MFCC):** Ponieważ stacjonarna odpowiedź częstotliwościowa mikrofonu jest stała w czasie ($C_{mic} \approx \text{const}$), operacja różniczkowania w czasie usuwa stałą składową kanału transmisyjnego ($\Delta C_{mic} = 0$).
  - **Prozodia i ton krtaniowy ($F_0$):** Estymacja autokorelacyjna YIN mierzy fizjologiczną częstotliwość drgań więzadeł głosowych, która nie zależy od charakterystyki pasmowej mikrofonu.
  - **Widmo modulacji obwiedni (1–8 Hz):** Mierzy tempo sylabiczne i zestroje akcentowe języka, niezależnie od barwy przetwornika.

---

## 6. Porównanie wyników eksperymentów

Ewaluację obu podejść przeprowadzono na **w 100% identycznej próbie danych** (dokładnie 2000 nagrań na język, po 4369 zbalansowanych próbek 1s na klasę, identyczny rozłączny podział mówców):

| Metryka / Parametr | `SpeechML.ipynb` (Obwiednia + Rytm / XGBoost GPU) | `SpeechSpectrogramSVM.ipynb` (MFCC + Prozodia / SVM RBF) |
| :--- | :---: | :---: |
| **Typ ekstrahowanych cech** | Obwiednia Hilberta, percentyle, pasma modulacji 1–8 Hz | 13 MFCC + $\Delta$ + $\Delta\Delta$ (pooling), statystyki $F_0$, tempo BPM, pauzy |
| **Wymiar wektora cech** | 27 cech | 90 cech (po PCA: 20 komponentów) |
| **Wielkość zbioru po balansie** | **4369 próbek (1s)** na język (8738 łącznie) | **4369 próbek (1s)** na język (8738 łącznie) |
| **Podział mówców (Train / Test)** | **2916 mówców / 972 mówców** | **2916 mówców / 972 mówców** |
| **Liczba próbek (Train / Test)** | **6539 próbek / 2199 próbek** | **6539 próbek / 2199 próbek** |
| **Optymalne hiperparametry** | Drzewa decyzyjne z regularyzacją L1/L2 | $C = 2.0$, $\gamma = \text{'scale'}$, Kernel = RBF, PCA = 20 |
| **Dokładność treningowa (Train Acc)** | 72.89% | 86.16% |
| **Dokładność testowa (Test Acc)** | **66.35%** | **77.26%** |
| **Luka generalizacji (Train - Test)** | 6.54% | 8.90% |
| **F1-Score na teście (Macro)** | 0.66 | **0.78** (0.7820) |
| **ROC-AUC na zbiorze testowym** | 0.743 | **0.850** (0.8502) |
| **5-Fold CV Val Acc (średnia $\pm \sigma$)** | 67.91% $\pm$ 1.03% | **77.51% $\pm$ 1.25%** |
| **5-Fold CV Val ROC-AUC** | 0.743 | **0.854** (0.8535) |
| **Współczynnik zmienności CV ($CV$)** | 0.0152 (1.52%) | 0.0161 (1.61%) |
| **Średnia Test Acc w Multi-seed (5 ziaren)** | 66.8% (ok.) | **77.83% $\pm$ 0.91%** |
| **Przedział ufności 95% (Multi-seed)** | [65.0%; 68.6%] | **[76.00%; 79.66%]** |
| **Współczynnik zmienności Multi-seed ($CV$)** | 0.015 | **0.0118 (1.18%)** |
| **Ocena stabilności modelu** | Wysoka stabilność | **Wysoka odporność na losowość podziału** |

### Kluczowe wnioski:
1. **Wpływ cech spektralno-prozodycznych na identycznej próbie:** Przy w 100% identycznej wielkości próby model MFCC + Prozodia + SVM RBF osiąga **77.26% dokładności testowej** oraz **0.850 ROC-AUC**, przewyższając model oparty na samej obwiedni i widmie modulacji (66.35% Acc, 0.743 ROC-AUC) o **+10.91 p.p.**
2. **Odporność na przeuczenie:** W obu podejściach luka generalizacji utrzymuje się poniżej 9 punktów procentowych, a współczynnik zmienności $CV$ w 5-krotnej walidacji grupowej i w badaniu 5 ziaren losowości nie przekracza $1.7\%$, co świadczy o wysokiej stabilności modeli na niezależnych mówcach.
3. **Ważność cech:** W modelu XGBoost cechami o największej stabilności między foldami były percentyle amplitudy (`p_25`, `p_10`), tempo występowania szczytów sylabicznych (`peak_rate_hz`, $CV=0.05$) oraz pasmo modulacji akcentowej (`rhythm_2hz`).
