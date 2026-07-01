import cv2 as cv
import matplotlib.pyplot as plt
import numpy as np

from utils.defects_identification import *
from utils.plotting import *
from utils.segmentation import *

SEED = 14

cv.setRNGSeed(SEED)
np.random.seed(SEED)

def detect_defects_on_fruits(images_nir, images_clr, threshold_segmentation, sigma, th, tl, images_names):
    circled_images = []

    # SEGMENTAZIONE (MANUAL BINARIZATION CON SOGLIA GLOBALE)
    first_masks, final_color_masks, final_masks = manual_binarization(images_clr, images_nir, threshold_segmentation)
    segmentation_images_nir = overlap_masks(final_masks, images_nir)

    # APERTURA MORFOLOGICA
    masks = refine_masks(final_masks, 5)
    segmentation_images_nir = overlap_masks(masks, images_nir)  

    # CANNY'S EDGE DETECTOR
    blur_images, edge_masks = apply_gaussian_and_canny(segmentation_images_nir, sigma, th, tl)

    # EROSIONE
    element = cv.getStructuringElement(cv.MORPH_RECT, (15,15))
    eroded_masks = [cv.erode(img, element) for img in masks]
    eroded_edge_masks = apply_mask_to_image(edge_masks, eroded_masks)

    # RIEMPIMENTO DEI DIFETTI
    element = cv.getStructuringElement(cv.MORPH_ELLIPSE, (50, 50))
    defect_masks = [cv.morphologyEx(img, cv.MORPH_CLOSE, element) for img in eroded_edge_masks]

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
            cv.circle(circled_img, center, radius, (0, 0, 255), 3)
        
        circled_images.append(circled_img)
    
    # MOSTRA LE IMMAGINI
    plot_image_grid(circled_images, images_names, 'Circled Defects')  


def main():
    THRESHOLD_SEGMENTATION = 30
    SIGMA = 1
    THRESHOLD_H = 100
    THRESHOLD_L = 45
    
    #salvo le foto NIR in un array
    images_NIR = []
    for x in range(1, 4):
        filename = f"images/first_task/C0_00000{x}.png"
        img = cv.imread(filename)
        if img is not None:
            images_NIR.append(img)
        else:
            print(f"Immagine {filename} non trovata!")
    
    #salvo le foto colorate in un array
    images_COLOR = []
    for x in range(1, 4):
        filename = f"images/first_task/C1_00000{x}.png"
        img = cv.imread(filename)
        if img is not None:
            images_COLOR.append(img)
        else:
            print(f"Immagine {filename} non trovata!")
    
    #salvo array contenenti i nomi delle foto NIR e colorate
    nir_names = ['C0_000001.png', 'C0_000002.png', 'C0_000003.png']
    color_names = ['C1_000001.png', 'C1_000002.png', 'C1_000003.png']

    #verifica che le immagini siano state caricate correttamente
    if any(img is None for img in images_COLOR + images_NIR):
        print("Errore: alcune immagini non sono state caricate correttamente.")
        return

    #esegui la funzione per rilevare ed evidenziare i difetti
    highlighted_images, circled_images = detect_defects_on_fruits(
        images_NIR, 
        images_COLOR,
        THRESHOLD_SEGMENTATION,
        SIGMA, 
        THRESHOLD_H, 
        THRESHOLD_L,
        color_names
    )

if __name__ == "__main__":
    main()


