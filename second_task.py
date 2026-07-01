import matplotlib.pyplot as plt
import numpy as np

from utils.defects_identification import *
from utils.plotting import *
from utils.segmentation import *

SEED = 14

cv.setRNGSeed(SEED)
np.random.seed(SEED)

def detect_russets_on_fruits(images_gray, images_clr, THRESHOLD, images_names):
    circled_images = []

    # FILTRO MEDIANO PER LE FOTO IN BIANCO E NERO
    images_gray_filtered = [cv.medianBlur(img, 5) for img in images_gray]

    # FILTRO BILATERALE PER LE FOTO A COLORI
    images_color_filtered = [cv.bilateralFilter(img, d=5, sigmaColor=75, sigmaSpace=75) for img in images_clr]

    # SEGMENTAZIONE TRAMITE BINARIZZAZIONE A SOGLIA GLOBALE
    first_masks, final_color_masks, final_masks = manual_binarization(images_clr, images_gray_filtered, THRESHOLD)
    masked_images = overlap_masks_color(final_masks, images_color_filtered)


    # CONVERSIONE IN LUV DEI SAMPLES
    samples_names = []

    for x in range(1, 30):
        samples_names.append(f"sample_{x:02d}.png")

    samples = []
    for x in range(1, 30):
        filename = f"fruit-inspection-images/second_task/samples/sample_{x:02d}.png" 
        img = cv.imread(filename)
        if img is not None:
            samples.append(img)
        else:
            print(f"Immagine {filename} non trovata!")

    luv_samples = []
    for i in range(len(samples)):
        luv_samples.append(cv.cvtColor(samples[i],cv.COLOR_RGB2Luv))
    luv_img = []
    for i in range(len(images_clr)):
        luv_img.append(cv.cvtColor(images_clr[i],cv.COLOR_RGB2Luv))

    # STIMA DEL VALORE ATTESO E DELLA MATRICE DI COVARIANZA
    w_min = min(img.shape[1] for img in luv_samples)

    sample_matrix = np.zeros((1, w_min, 3), dtype=np.uint8)

    for img in luv_samples:
        width = img.shape[1]  
        n_subsamples = width // w_min  
        
        for j in range(0, n_subsamples * w_min, w_min):
            sub_sample = img[:, j:j+w_min, :]  
            sample_matrix = np.concatenate((sample_matrix, sub_sample), axis=0)

    sample_matrix = sample_matrix[1:, :, :]

    uv_data = sample_matrix[:, :, 1:3].reshape(-1, 2)

    covar_matrix, mean_color = cv.calcCovarMatrix(
        uv_data, 
        None, 
        flags=cv.COVAR_ROWS | cv.COVAR_NORMAL | cv.COVAR_SCALE
    )

    # DISTANZA DI MALAHANOBIS
    inv_cov = cv.invert(covar_matrix)[1]

    russets = []

    for img_idx, luv_image in enumerate(luv_img):
        russet_mask = np.zeros((luv_image.shape[0], luv_image.shape[1]), dtype=np.uint8)
        
        for i in range(luv_image.shape[0]):
            for j in range(luv_image.shape[1]):
                if final_masks[img_idx][i, j] == 255:
                    #distanza di Mahalanobis
                    pixel_uv = np.float64(luv_image[i, j, 1:3])  # Canali U e V del pixel
                    mahal_dist = cv.Mahalanobis(mean_color.flatten(), pixel_uv, inv_cov)

                    #se la distanza è inferiore alla soglia, segna il pixel come marrone
                    if mahal_dist < 1.25:
                        russet_mask[i, j] = 255

        russets.append(russet_mask)

    # DILATAZIONE 
    russets_cleaned = []

    for i, luv in enumerate(luv_img):
        background = 255 - final_masks[i]

        kernel = cv.getStructuringElement(cv.MORPH_ELLIPSE, (11, 11))
        dilated_bg = cv.dilate(background, kernel, iterations=3)

        russet_cleaned = cv.subtract(russets[i], dilated_bg)
        russets_cleaned.append(russet_cleaned)

    # CHIUSURA MORFOLOGICA
    final_russets = []
    russets_rgb = []

    structuring_element = cv.getStructuringElement(cv.MORPH_ELLIPSE, (30, 30))

    for russet_mask, color_image in zip(russets_cleaned, images_clr):
        closed_russet = cv.morphologyEx(russet_mask, cv.MORPH_CLOSE, structuring_element)
        final_russets.append(closed_russet)

    russets_rgb = highlight_edges(images_clr, final_russets)

    # DISEGNO I CERCHI INTORNO AI DIFETTI
    for img_color, russet_mask in zip(images_clr, final_russets):
        # Trova i contorni del difetto
        contours, _ = cv.findContours(russet_mask, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
        
        # Crea una copia dell'immagine originale
        circled_img = img_color.copy()
        
        # Disegna i cerchi attorno ai difetti
        for contour in contours:
            # Ottieni il rettangolo circoscritto minimo (per calcolare il centro e il raggio)
            (x, y), radius = cv.minEnclosingCircle(contour)
            center = (int(x), int(y))
            radius = int(radius)
            
            # Disegna il cerchio (verde per evidenziare il difetto)
            cv.circle(circled_img, center, radius, (0, 0, 255), 2)
    
        circled_images.append(circled_img)
    
    # MOSTRA LE IMMAGINI
    plot_image_grid(circled_images, images_names, 'Circled Defects')  

def main():
    THRESHOLD_SEGMENTATION = 25
    
    # Salvo una versione grayscale delle immagini a colori
    images_gray = []
    for i in range(1, 3):
        filename = f"fruit-inspection-images/second_task/C1_00000{i+3}.png"
        img = cv.imread(filename)
        if img is not None:
            images_gray.append(cv.cvtColor(img, cv.COLOR_BGR2GRAY))
        else:
            print(f"Immagine {filename} non trovata!")
    
    # Salvo le foto a colori in un array
    images_clr = []
    for i in range(1, 3):
        filename = f"fruit-inspection-images/second_task/C1_00000{i+3}.png"
        img = cv.imread(filename)
        if img is not None:
            images_clr.append(img)
        else:
            print(f"Immagine {filename} non trovata!")
        
    # Salvo array contenenti i nomi delle foto a colori
    color_names = ['C1_000004.png', 'C1_000005.png']

    # Verifica che le immagini siano state caricate correttamente
    if any(img is None for img in images_clr + images_gray):
        print("Errore: alcune immagini non sono state caricate correttamente.")
        return

    # Esegui la funzione per rilevare ed evidenziare i difetti
    highlighted_images, circled_images = detect_russets_on_fruits(
        images_gray, 
        images_clr,
        THRESHOLD_SEGMENTATION,
        color_names
    )

if __name__ == "__main__":
    main()
