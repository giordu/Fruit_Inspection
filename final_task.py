import cv2 as cv
import matplotlib.pyplot as plt
import numpy as np

from utils.defects_identification import *
from utils.plotting import *
from utils.segmentation import *

SEED = 14

cv.setRNGSeed(SEED)
np.random.seed(SEED)

def detect_defects_on_kiwis(images_nir, images_clr, images_names, THRESHOLD_SEGMENTATION, BILATERAL_KERNEL, SIGMA_COLOR, SIGMA_SPACE, THRESHOLD_H, THRESHOLD_L):
    circled_images = []

    # FILTRO MEDIANO
    median_images = [cv.medianBlur(img, 5) for img in images_nir]

    # BINARIZZAZIONE MANUALE CON SOGLIA GLOBALE
    first_masks, final_color_masks, final_masks = manual_binarization(images_clr, median_images, THRESHOLD_SEGMENTATION)

    # EROSIONE
    element = cv.getStructuringElement(cv.MORPH_CROSS, (5, 5))
    eroded_masks = [cv.erode(img, element) for img in first_masks]

    # ELIMINAZIONE ZONE SUPERFLUE
    kiwi_masks = []
    
    for img in eroded_masks:
     # Verifica se l'immagine ha più di un canale e convertila in scala di grigi
        if len(img.shape) > 2:
            img = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
        
        # Trova le componenti connesse e le loro statistiche
        num_labels, labels, stats, _ = cv.connectedComponentsWithStats(img, connectivity=8)
        
        # Inizializza variabili per tracciare l'area massima
        max_area = 0
        largest_label = 0
        
        # Itera su tutte le etichette, partendo da 1 (saltando lo sfondo, label 0)
        for label in range(1, num_labels):
            area = stats[label, cv.CC_STAT_AREA]
            if area > max_area:
                max_area = area
                largest_label = label
        
        # Crea una nuova immagine vuota
        output_image = np.zeros_like(img, dtype=np.uint8)
        
        # Mantieni solo l'area maggiore
        output_image[labels == largest_label] = 255
        
        kiwi_masks.append(output_image)

    # EROSIONE
    element = cv.getStructuringElement(cv.MORPH_ELLIPSE, (5, 5))
    eroded_kiwi_masks = [cv.erode(img, element) for img in kiwi_masks]

    # CHIUSURA MORFOLOGICA
    element = cv.getStructuringElement(cv.MORPH_ELLIPSE, (30, 30))
    final_masks = [cv.morphologyEx(img, cv.MORPH_CLOSE, element) for img in eroded_kiwi_masks]
    segmentation_images = overlap_masks(final_masks, images_nir)

    # BILATERAL E CANNY
    blur_images, edge_masks = apply_bilateral_and_canny(segmentation_images, BILATERAL_KERNEL, SIGMA_COLOR, SIGMA_SPACE, THRESHOLD_H, THRESHOLD_L)
    highlighted_edges_images = highlight_edges(images_nir, edge_masks)

    # DILATAZIONE
    defect_edges = []

    for i, img in enumerate(edge_masks):
        #get the bg
        background = 255 - final_masks[i]

        #dilate bg
        background_dilated = cv.dilate(background,cv.getStructuringElement(cv.MORPH_ELLIPSE,(20,20)),iterations = 2)
        
        #subtract the dilated bg from the edges detected by Canny
        defect_edges.append(cv.subtract(edge_masks[i],background_dilated))

    # CHIUSURA MORFOLOGICA
    element = cv.getStructuringElement(cv.MORPH_ELLIPSE, (50, 50))
    defect_masks = [cv.morphologyEx(img, cv.MORPH_CLOSE, element) for img in defect_edges]
    highlighted_defects = highlight_edges(images_clr, defect_masks)

    # FILTRO MEDIANO
    defect_masks = [cv.medianBlur(m, 7) for m in defect_masks]
    highlighted_defects = highlight_edges(images_clr, defect_masks)

    # DISEGNO I CERCHI ATTORNO AI DIFETTI
    for img_color, defect_mask in zip(images_clr, defect_masks):
        # Trova i contorni del difetto
        contours, _ = cv.findContours(defect_mask, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
        
        # Crea una copia dell'immagine originale
        circled_img = img_color.copy()
        
        # Disegna i cerchi attorno ai difetti
        for contour in contours:
            # Ottieni il rettangolo circoscritto minimo (per calcolare il centro e il raggio)
            (x, y), radius = cv.minEnclosingCircle(contour)
            center = (int(x), int(y))
            radius = int(radius) + 5
            
            # Disegna il cerchio (verde per evidenziare il difetto)
            cv.circle(circled_img, center, radius, (0, 0, 255), 2)
        
        circled_images.append(circled_img)
    
    # MOSTRA LE IMMAGINI
    plot_image_grid(circled_images, images_names, 'Circled Defects')  

def main():
    THRESHOLD_SEGMENTATION = 35
    SIGMA_COLOR = 30
    SIGMA_SPACE = SIGMA_COLOR
    THRESHOLD_L = 35
    THRESHOLD_H = 80
    BILATERAL_KERNEL = 7

    # salvo le foto NIR in un array
    images_nir = []
    for i in range(1, 6):
        if i == 5:
            filename = f"fruit-inspection-images/final_challenge/C0_0000{i+5}.png" 
        else:
            filename = f"fruit-inspection-images/final_challenge/C0_00000{i+5}.png" 
        img = cv.imread(filename)
        if img is not None:
            images_nir.append(img)
        else:
            print(f"Immagine {filename} non trovata!")

    #salvo le foto colorate in un array
    images_clr = []
    for i in range(1, 6):
        if i == 5:
            filename = f"fruit-inspection-images/final_challenge/C1_0000{i+5}.png"
        else:
            filename = f"fruit-inspection-images/final_challenge/C1_00000{i+5}.png"
        img = cv.imread(filename)
        if img is not None:
            images_clr.append(img)
        else:
            print(f"Immagine {filename} non trovata!")

    #salvo array contenenti i nomi delle foto NIR e a colori
    nir_names = ['C0_000006.png', 'C0_000007.png', 'C0_000008.png', 'C0_000009png', 'C0_000010.png']
    color_names = ['C1_000006.png', 'C1_000007.png', 'C1_000008.png', 'C1_000009.png', 'C1_000010.png']

    #verifica che le immagini siano state caricate correttamente
    if any(img is None for img in images_nir + images_clr):
        print("Errore: alcune immagini non sono state caricate correttamente.")
        return

    #esegui la funzione per rilevare ed evidenziare i difetti
    highlighted_images, circled_images = detect_defects_on_kiwis(
        images_nir, 
        images_clr,
        color_names,
        THRESHOLD_SEGMENTATION,
        BILATERAL_KERNEL,
        SIGMA_COLOR,
        SIGMA_SPACE, 
        THRESHOLD_H, 
        THRESHOLD_L,
        color_names
    )

if __name__ == "__main__":
    main()

