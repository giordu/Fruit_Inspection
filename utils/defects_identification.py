import cv2 as cv
import math
import numpy as np
import time

from utils.plotting import *
from utils.segmentation import *

def apply_gaussian_and_canny(images, sigma, th, tl):
    #converto tutte le immagini in scala di grigi (se necessario)
    images_gray = []
    for img in images:
        if len(img.shape) == 3:
            images_gray.append(cv.cvtColor(img, cv.COLOR_BGR2GRAY))
        else:
            images_gray.append(img)

    #costruisco il kernel
    k = math.ceil(3 * sigma)
    kernel_size = (2 * k + 1, 2 * k + 1)

    #filtro e applico l'edge detector a tutte le immagini
    blur_images = []
    edge_masks = []
    for img in images_gray:
        blur_image = cv.GaussianBlur(img, kernel_size, sigma) #prima di tutto applicazione del filtro gaussiano
        blur_images.append(blur_image) #insieme di immagini post blur Gaussiano

        edge_masks.append(cv.Canny(blur_image, th, tl)) #maschere finali con difetti evidenziati da bordi bianchi

    # #nel caso ci siano dei pixel di rumore impulsivo sullo sfondo
    # for mask in edge_masks:
    #     cv.floodFill(mask, None, (0, 0), 0)

    return blur_images, edge_masks

#funzione per evidenziare i bordi (in rosso)
def highlight_edges(images, edge_masks, edge_color=(0, 0, 255), alpha=0.5):
    highlighted_edges_images = []

    for img, edge_mask in zip(images, edge_masks):
        #controllo che l'immagine sia in scala di grigi altrimenti converto
        if len(img.shape) == 2 or img.shape[2] == 1:
            color_img = cv.cvtColor(img, cv.COLOR_GRAY2BGR)
        else:
            color_img = img.copy()

        #creo un'immagine vuota del colore desiderato
        colored_edges = np.zeros_like(color_img)
        colored_edges[edge_mask == 255] = edge_color

        #applico la sovrapposizione semi-trasparente
        final_img = color_img.copy()
        mask_indices = edge_mask == 255  #trovo i pixel dove applicare l'overlay
        for c in range(3):  #applico per ciascun canale (R, G, B)
            final_img[..., c][mask_indices] = (
                (1 - alpha) * color_img[..., c][mask_indices] + alpha * colored_edges[..., c][mask_indices]
            ).astype(np.uint8)

        highlighted_edges_images.append(final_img)

    return highlighted_edges_images

def apply_mask_to_image(image_1, image_2):
     # Verifica che i due array abbiano la stessa lunghezza
    if len(image_1) != len(image_2):
        raise ValueError("Entrambi gli array di immagini devono avere la stessa lunghezza.")

    masked_images = []

    # Itera attraverso le immagini nei due array
    for img1, img2 in zip(image_1, image_2):
        # Se l'immagine è a colori (3 canali), converte la maschera in formato RGB
        if len(img1.shape) == 3:
            img2 = cv.cvtColor(img2, cv.COLOR_GRAY2BGR)

        # Applica l'operazione bitwise AND tra le immagini
        masked_img = img1 & img2  # Bitwise AND tra le immagini
        masked_images.append(masked_img)

    return masked_images

def apply_bilateral_and_canny(images, kernel_size, sigma_c, sigma_s, th, tl):
    #converto tutte le immagini in scala di grigi (se necessario)
    images_gray = []
    for img in images:
        if len(img.shape) == 3:
            images_gray.append(cv.cvtColor(img, cv.COLOR_BGR2GRAY))
        else:
            images_gray.append(img)

    #filtro e applico l'edge detector a tutte le immagini
    blur_images = []
    edge_masks = []
    for img in images_gray:
        blur_image = cv.bilateralFilter(img, kernel_size, sigma_c, sigma_s) #prima di tutto applicazione del filtro gaussiano
        blur_images.append(blur_image) #insieme di immagini post blur bilaterale
        edge_masks.append(cv.Canny(blur_image, th, tl)) #maschere finali

    #nel caso ci siano dei pixel di rumore impulsivo sullo sfondo
    for mask in edge_masks:
        cv.floodFill(mask, None, (0, 0), 0)

    return blur_images, edge_masks
