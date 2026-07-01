import cv2 as cv
import math
import numpy as np
import time

from utils.plotting import *

def manual_binarization(images_color, images_median, threshold):
    first_masks = []
    flood_filled_masks = []
    final_masks = []
    final_color_masks = []
    
    #applico il threshold alle immagini NIR pre-elaborate
    for i, img in enumerate(images_median):
        _, binary_img = cv.threshold(img, threshold, 255, cv.THRESH_BINARY)
        first_masks.append(binary_img)
    
    #FLOOD-FILL
    for i, img in enumerate(first_masks):    
        #assicuriamoci che la maschera sia in scala di grigi
        if len(img.shape) == 3: 
            img = cv.cvtColor(img, cv.COLOR_BGR2GRAY) #conversione in scala di grigi
        
        #ottengo altezza e larghezza
        h, w = img.shape
        mask_padding = np.zeros((h+2, w+2), dtype=np.uint8)
        
        #faccio una copia dell'immagine originale
        flood_filled_img = img.copy()
        
        #inonda lo sfondo nero di bianco partendo dall'angolo (0,0) (che è esterno)
        cv.floodFill(flood_filled_img, mask_padding, (0, 0), 255)
        
        #invertiamo i colori: ora i buchi interni (che erano rimasti neri) diventano bianchi
        flood_filled_masks.append(~flood_filled_img)
    
    #FUSIONE (Mela con i buchi + Tappi dei buchi)
    #sovrappongo le immagini di `first_masks` con quelle di `flood_filled_masks`
    for first_img, flood_img in zip(first_masks, flood_filled_masks):
        if len(first_img.shape) == 3:  # Se è a colori
            first_img = cv.cvtColor(first_img, cv.COLOR_BGR2GRAY)  # Converti in scala di grigi
        if len(flood_img.shape) == 3:  # Se è a colori
            flood_img = cv.cvtColor(flood_img, cv.COLOR_BGR2GRAY)  # Converti in scala di grigi
            
        final_img = np.maximum(first_img, flood_img)  # Sovrappone le immagini (riempie i buchi)
        final_masks.append(final_img)
    
    #sovrapponiamo la maschera finale a un'immagine a colori in 'images_COLOR'
    for final_mask, color_img in zip(final_masks, images_color):
        #converte l'immagine in formato RGB (se non lo è già)
        color_img_rgb = cv.cvtColor(color_img, cv.COLOR_BGR2RGB)
    
        #crea la maschera gialla con trasparenza (senza canale alfa)
        yellow_mask = np.zeros_like(color_img_rgb)  # Crea un'immagine nera con 3 canali (RGB)
        
        #rende gialli i pixel bianchi nella maschera finale
        yellow_mask[final_mask == 255] = [255, 255, 0]  # Giallo
    
        #opacità della maschera gialla (0 = completamente trasparente, 1 = completamente opaco)
        alpha = 0.25  # Impostiamo un valore di trasparenza del 25%
    
        #sovrappongo la maschera gialla semi-trasparente sull'immagine originale
        for c in range(3):  # Per i canali R, G, B
            # Combinazione lineare tra l'immagine originale e la maschera gialla con trasparenza
            color_img_rgb[..., c] = (1 - alpha) * color_img_rgb[..., c] + alpha * yellow_mask[..., c]
    
        #aggiungo l'immagine finale con la maschera gialla semi-trasparente
        final_color_masks.append(color_img_rgb)
    
    #restituiamo sia le maschere binarie perfette, sia le foto a colori scontornate
    return first_masks, final_color_masks, final_masks


def otsu_binarization(images_color, images_median):
    first_masks = []
    flood_filled_masks = []
    final_masks = []
    final_color_masks = []

    #applico la binarizzazione tramite Otsu alle mie immagini
    for i, img in enumerate(images_median):
        #converto l'immagine in scala di grigi nel caso non lo fosse
        if len(img.shape) == 3:  #se l'immagine è a colori
            img = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

        #utilizzo Otsu per calcolare il threshold
        _, binary_img = cv.threshold(img, 0, 255, cv.THRESH_BINARY + cv.THRESH_OTSU)
        first_masks.append(binary_img)

    #FLOOD-FILL
    #itero sulle immagini binarie
    for i, img in enumerate(first_masks):
        #ottengo altezza e larghezza
        h, w = img.shape
        mask_padding = np.zeros((h+2, w+2), dtype=np.uint8)
        
        #faccio una copia dell'immagine originale
        flood_filled_img = img.copy()
        
        #inondo lo sfondo nero di bianco partendo dall'angolo (0,0) (che è esterno)
        cv.floodFill(flood_filled_img, mask_padding, (0, 0), 255)
        
        #invertiamo i colori: ora i buchi interni (che erano rimasti neri) diventano bianchi
        flood_filled_masks.append(~flood_filled_img)

     #FUSIONE (Mela con i buchi + Tappi dei buchi)
    #sovrappongo le immagini di `first_masks` con quelle di `flood_filled_masks`
    for first_img, flood_img in zip(first_masks, flood_filled_masks):
        if len(first_img.shape) == 3:  # Se è a colori
            first_img = cv.cvtColor(first_img, cv.COLOR_BGR2GRAY)  # Converti in scala di grigi
        if len(flood_img.shape) == 3:  # Se è a colori
            flood_img = cv.cvtColor(flood_img, cv.COLOR_BGR2GRAY)  # Converti in scala di grigi
            
        final_img = np.maximum(first_img, flood_img)  # Sovrappone le immagini (riempie i buchi)
        final_masks.append(final_img)
    
    #sovrapponiamo la maschera finale a un'immagine a colori in 'images_COLOR'
    for final_mask, color_img in zip(final_masks, images_color):
        #converte l'immagine in formato RGB (se non lo è già)
        color_img_rgb = cv.cvtColor(color_img, cv.COLOR_BGR2RGB)
    
        #crea la maschera gialla con trasparenza (senza canale alfa)
        yellow_mask = np.zeros_like(color_img_rgb)  # Crea un'immagine nera con 3 canali (RGB)
        
        #rende gialli i pixel bianchi nella maschera finale
        yellow_mask[final_mask == 255] = [255, 255, 0]  # Giallo
    
        #opacità della maschera gialla (0 = completamente trasparente, 1 = completamente opaco)
        alpha = 0.25  # Impostiamo un valore di trasparenza del 25%
    
        #sovrappongo la maschera gialla semi-trasparente sull'immagine originale
        for c in range(3):  # Per i canali R, G, B
            # Combinazione lineare tra l'immagine originale e la maschera gialla con trasparenza
            color_img_rgb[..., c] = (1 - alpha) * color_img_rgb[..., c] + alpha * yellow_mask[..., c]
    
        #aggiungo l'immagine finale con la maschera gialla semi-trasparente
        final_color_masks.append(color_img_rgb)
    
    #restituiamo sia le maschere binarie perfette, sia le foto a colori scontornate
    return first_masks, final_color_masks, final_masks

def adaptive_threshold(images_color, images_median, block_size, C):
    first_masks = []
    flood_filled_masks = []
    final_masks = []
    final_color_masks = []

    #applico la binarizzazione tramite Otsu alle mie immagini
    for i, img in enumerate(images_median):
        #converto l'immagine in scala di grigi nel caso non lo fosse
        if len(img.shape) == 3:  #se l'immagine è a colori
            img = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

        #utilizzo Adaptive Threshold per calcolare il threshold
        binary_img = cv.adaptiveThreshold(
            img, 255, cv.ADAPTIVE_THRESH_MEAN_C, cv.THRESH_BINARY, block_size, C
        )
        binary_img = np.pad(binary_img, 1, mode='constant', constant_values=255)
        cv.floodFill(binary_img, None, (0, 0), 0)
        binary_img = binary_img[1:-1, 1:-1]

        first_masks.append(binary_img)

    #FLOOD-FILL
    #itero sulle immagini binarie
    for i, img in enumerate(first_masks):
        #ottengo altezza e larghezza
        h, w = img.shape
        mask_padding = np.zeros((h+2, w+2), dtype=np.uint8)
        
        #faccio una copia dell'immagine originale
        flood_filled_img = img.copy()
        
        #inondo lo sfondo nero di bianco partendo dall'angolo (0,0) (che è esterno)
        cv.floodFill(flood_filled_img, mask_padding, (0, 0), 255)
        
        #invertiamo i colori: ora i buchi interni (che erano rimasti neri) diventano bianchi
        flood_filled_masks.append(~flood_filled_img)

    #FUSIONE (Mela con i buchi + Tappi dei buchi)
    # Sovrappongo le immagini di `first_masks` con quelle di `flood_filled_masks`
    for first_img, flood_img in zip(first_masks, flood_filled_masks):
        if len(first_img.shape) == 3:  # Se è a colori
            first_img = cv.cvtColor(first_img, cv.COLOR_BGR2GRAY)  # Converti in scala di grigi
        if len(flood_img.shape) == 3:  # Se è a colori
            flood_img = cv.cvtColor(flood_img, cv.COLOR_BGR2GRAY)  # Converti in scala di grigi
            
        final_img = np.maximum(first_img, flood_img)  # Sovrappone le immagini (riempie i buchi)
        final_masks.append(final_img)
    
    #sovrapponiamo la maschera finale a un'immagine a colori in 'images_COLOR'
    for final_mask, color_img in zip(final_masks, images_color):
        #converte l'immagine in formato RGB (se non lo è già)
        color_img_rgb = cv.cvtColor(color_img, cv.COLOR_BGR2RGB)
    
        #crea la maschera gialla con trasparenza (senza canale alfa)
        yellow_mask = np.zeros_like(color_img_rgb)  # Crea un'immagine nera con 3 canali (RGB)
        
        #rende gialli i pixel bianchi nella maschera finale
        yellow_mask[final_mask == 255] = [255, 255, 0]  # Giallo
    
        #opacità della maschera gialla (0 = completamente trasparente, 1 = completamente opaco)
        alpha = 0.25  # Impostiamo un valore di trasparenza del 25%
    
        #sovrappongo la maschera gialla semi-trasparente sull'immagine originale
        for c in range(3):  # Per i canali R, G, B
            # Combinazione lineare tra l'immagine originale e la maschera gialla con trasparenza
            color_img_rgb[..., c] = (1 - alpha) * color_img_rgb[..., c] + alpha * yellow_mask[..., c]
    
        #aggiungo l'immagine finale con la maschera gialla semi-trasparente
        final_color_masks.append(color_img_rgb)
    
    #restituiamo sia le maschere binarie perfette, sia le foto a colori scontornate
    return first_masks, final_color_masks, final_masks

def segmentation_timing(images, threshold, block_size, C, iterations):
    #converto tutte le immagini in scala di grigi (se necessario)
    images_gray = []
    for img in images:
        if len(img.shape) == 3:  # Se l'immagine è a colori
            images_gray.append(cv.cvtColor(img, cv.COLOR_BGR2GRAY))
        else:
            images_gray.append(img)

    #misurazione del tempo per soglia globale
    start_time = time.time()
    for _ in range(iterations):
        for img in images_gray:
            _, _ = cv.threshold(img, threshold, 255, cv.THRESH_BINARY)
    global_time = time.time() - start_time

    #misurazione del tempo per Otsu
    start_time = time.time()
    for _ in range(iterations):
        for img in images_gray:
            _, _ = cv.threshold(img, 0, 255, cv.THRESH_BINARY + cv.THRESH_OTSU)
    otsu_time = time.time() - start_time

    #misurazione del tempo per soglia locale
    start_time = time.time()
    for _ in range(iterations):
        for img in images_gray:
            _ = cv.adaptiveThreshold(
                img, 255, cv.ADAPTIVE_THRESH_GAUSSIAN_C, cv.THRESH_BINARY, block_size, C
            )
    adaptive_time = time.time() - start_time

    #stampa dei risultati dei tempi
    print(f"Tempo totale di segmentazione (Manuale con Soglia globale) per {iterations} iterazioni: {global_time:.4f} secondi")
    print(f"Tempo totale di segmentazione (Otsu) per {iterations} iterazioni: {otsu_time:.4f} secondi")
    print(f"Tempo totale di segmentazione (Soglia locale) per {iterations} iterazioni: {adaptive_time:.4f} secondi")

def overlap_masks(masks, images):
    final_masks = []  # Lista per salvare le immagini finali con maschera applicata

    for mask, img in zip(masks, images):
        #crea una copia dell'immagine NIR per applicare la maschera
        masked_img = img.copy().astype(np.float32)

        #imposta l'opacità della maschera
        alpha = 0.15

        #applica la trasparenza sovrapponendo la maschera bianca all'immagine 
        masked_img[mask == 255] = (1 - alpha) * masked_img[mask == 255] + alpha * 255

        #imposta a nero lo sfondo (pixel fuori dalla maschera)
        masked_img[mask != 255] = 0

        #salva l'immagine risultante dopo averla riportata a valori validi (8-bit)
        final_masks.append(masked_img.astype(np.uint8))

    return final_masks

def refine_masks(masks, kernel_size):
   # Crea un elemento strutturante rettangolare della dimensione scelta
    kernel = cv.getStructuringElement(cv.MORPH_RECT, (kernel_size, kernel_size))
    
    # Lista per salvare le maschere raffinate
    refined_masks = []

    for mask in masks:
        # Applica l'APERTURA morfologica (Erosione + Dilatazione)
        # Questo eliminerà i piccoli punti bianchi (rumore) sullo sfondo nero
        refined_mask = cv.morphologyEx(mask, cv.MORPH_OPEN, kernel)
        refined_masks.append(refined_mask)

    return refined_masks

def overlap_masks_color(masks, images):
    final_masks = [] 
    for mask, img in zip(masks, images):
        # Crea una copia (non serve float32 se non fai alpha blending)
        masked_img = img.copy()

        # 2TAGLIO NETTO: dove la maschera è NERA, metti a nero l'immagine
        # Se img è a colori, mask deve essere convertita o usata come mask
        res = cv.bitwise_and(img, img, mask=mask)

        final_masks.append(res)
    return final_masks