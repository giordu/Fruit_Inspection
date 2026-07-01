# Project Name: Fruit Inspection
**Image Processing and Computer Vision Project**  

---

## Descrizione del Progetto
Questo progetto accademico ha come obiettivo lo sviluppo di un sistema automatizzato per l'ispezione della frutta, in grado di segmentare le immagini, rilevare imperfezioni e identificare specifici difetti superficiali. 

Le immagini utilizzate nel progetto sono state acquisite sia tramite una fotocamera a colori (RGB) sia tramite una fotocamera **NIR** (*Near Infra-Red*). 

Il progetto è suddiviso in tre task principali, ciascuno focalizzato su una specifica fase di analisi:

### 1. [first_task.ipynb](first_task.ipynb) - Segmentazione e Rilevamento dei Difetti
Focalizzato sulle immagini con difetti facilmente visibili.

### 2. [second_task.ipynb](second_task.ipynb) - Rilevamento delle Zone Color Ruggine
Focalizzato sull'identificazione di alterazioni cromatiche note come "zone ruggine" (aree marroni). L'obiettivo è isolare accuratamente queste zone riducendo al minimo i falsi positivi, al fine di permettere una corretta classificazione e selezione del frutto.

### 3. [final_task.ipynb](final_task.ipynb) - Kiwi Inspection
Applicazione dei modelli e delle tecniche di Computer Vision all'ispezione specifica dei kiwi. Il task affronta sfide di segmentazione avanzate, inclusa la gestione di elementi di disturbo sullo sfondo come la presenza di bollini/adesivi commerciali sul frutto.

---

## Prerequisiti e Tecnologie Utilizzate
Il sistema è sviluppato in **Python** e si basa principalmente sulle seguenti librerie di Computer Vision:
* **OpenCV (cv2):** Per l'elaborazione delle immagini, thresholding, segmentazione e operazioni morfologiche.
* **NumPy:** Per la manipolazione efficiente delle matrici e delle maschere binarie.
* **Matplotlib:** Per la visualizzazione dei risultati e il plotting dei grafici comparativi.

---

## Struttura della Repository
* `first_task.ipynb`: Notebook dedicato alla segmentazione base e rilevamento difetti.
* `second_task.ipynb`: Notebook focalizzato sulle lesioni da ruggine.
* `final_task.ipynb`: Notebook finale incentrato sulla rilevazione difetti nei kiwi e gestione sfondi complessi.
* `utils/`: Contiene i moduli Python di supporto per il plotting, la segmentazione e l'identificazione dei difetti (`plotting.py`, `segmentation.py`, `defects_identification.py`).
* `fruit-inspection-images/`: Cartella contenente i dataset di immagini RGB e NIR divisi per task.

## Come Eseguire il Progetto

1. **Clonare la repository:**
   ```bash
   git clone https://github.com/giordu/Fruit_Inspection.git
   cd Fruit_Inspection

2. **Installare le librerie necessarie**
    ```bash
   pip install opencv-python matplotlib numpy
3. **Lanciare il programma**
     - Per avviare il primo task (Segmentazione e difetti):
         ```bash
         python first_task.py
     - Per avviare il secondo task (Rilevamento zone ruggine):
         ```bash
           python second_task.py
     - Per avviare l'utimo task (Ispezione kiwi):
         ```bash
           python final_task.py
  
        
         
  
      
