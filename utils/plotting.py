import cv2 as cv
import matplotlib.pyplot as plt

###TASK 1###
#funzione per il plotting delle immagini iniziali
def plot_images():
    fig, axes = plt.subplots(2, 3, figsize=(15,10))

    #Titoli per i due gruppi di immagini
    group_titles = ['NIR Images', 'Color Images']

    #Primo ciclo per i gruppi (Gruppo 1 (i=0), Gruppo 2 (i=1))
    for i in range(2): 
        #Secondo ciclo per le immagini (3 per group)
        for j in range(1,4):
            path = f"fruit-inspection-images/first_task/C{i}_00000{j}.png"

            img = cv.imread(path)

            if img is None:
                print(f"Immagine {path} non trovata...")
            else:
                ax = axes[i, j-1]
                ax.imshow(cv.cvtColor(img, cv.COLOR_BGR2RGB))
                ax.axis('off') #nascosti gli assi

                #title
                if j == 1:
                    if i == 0:
                        fig.text(0.5, 0.91, group_titles[i], ha='center', fontsize=20)
                    else:
                        fig.text(0.5, 0.49, group_titles[i], ha='center', fontsize=20)
    plt.show()

#funzione per il plotting degli istogrammi dei livelli di grigio delle immagini NIR
def plot_NIR_hist():
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    #titolo unico per tutti e tre gli istogrammi
    plt.suptitle('Gray-level scale Histograms (NIR Images)', fontsize=16)

    #ciclo per le immagini
    for i in range(1,4):
        path = f"fruit-inspection-images/first_task/C{0}_00000{i}.png"

        img = cv.imread(path)

        if img is None:
            print(f"Immagine {path} non trovata...")
        else:
            hist = cv.calcHist([img], [0], None, [256], [0, 256])

            #mostra l'hist nel subplot
            axes[i-1].plot(hist)
            axes[i-1].set_xlim(0, 255) #imposta il limite dell'asse x a 255
            axes[i-1].set_title(f"C0_00000{i}.png")
            axes[i-1].set_xlabel('Gray-level Scale')
            axes[i-1].set_ylabel('Frequency')
    
    #aggiusto distanza tra i subplot
    plt.subplots_adjust(wspace=0.4)

    plt.show()


#funzione per il plotting delle NIR filters  
def plot_hist_filters(images_1, images_2, image_names):
    # Usiamo una lista interna per gestire i due set di titoli e immagini
    sets = [
        (images_2, 'Filtered Images using Median Filter Hist'),
        (images_1, 'NON Filtered Images Hist')
    ]
    
    for current_images, main_title in sets:
        fig, axes = plt.subplots(1, len(current_images), figsize=(18, 6))
        fig.suptitle(main_title, fontsize=16)
        
        # Se c'è solo un'immagine, axes non è una lista, lo rendiamo tale per il ciclo
        if len(current_images) == 1:
            axes = [axes]
            
        for i in range(len(current_images)):
            img = current_images[i]
            if img is not None:
                hist = cv.calcHist([img], [0], None, [256], [0, 256])
                
                axes[i].plot(hist)
                axes[i].set_xlim(0, 255)
                axes[i].set_title(image_names[i])  
                axes[i].set_xlabel('Livelli di grigio')  
                axes[i].set_ylabel('Frequenza') 
            else:
                print(f"Immagine all'indice {i} non trovata.")
        
        plt.subplots_adjust(wspace=0.3)
        plt.show()

#funzione per il plotting delle immagini
def plot_image_grid(images, images_names, title):
    assert images_names is None or len(images_names) == len(images), \
        '`images_names` must not be provided or it must have the same size as `images`.'

    fig = plt.figure(figsize=(15, 5), constrained_layout=True)
    if title is not None:
        fig.suptitle(title, fontsize=16)
    else:
        fig.suptitle('', fontsize=16)
    for idx, img in enumerate(images):
        #aggiungo un asse al plot
        plt.subplot(1, len(images), idx + 1)
        #rimuovo gli assi numerici
        plt.axis('off')
        #se l'immagine ha 3 dimensioni per il plot allora è rgb, altrimenti è in scala di grigi
        if len(img.shape) == 3:
            plt.imshow(cv.cvtColor(img, cv.COLOR_BGR2RGB))
        else:
            plt.imshow(img, cmap='gray', vmin=0, vmax=255)
        if images_names is not None:
            plt.title(images_names[idx])
    plt.show()

def plot_segmentation(first_masks, final_masks, final_color_masks, images_names):
    fig, axs = plt.subplots(3, len(first_masks), figsize=(15, 15))
    
    # Titoli principali sopra le immagini
    plt.suptitle("Post Threshold, post Flood-Fill e post confronto tra final mask e immagine colorata", fontsize=16)
    
    # Visualizza le immagini da `first_masks` (dopo threshold)
    for i, img in enumerate(first_masks):
        axs[0, i].imshow(img, cmap='gray')  # Mostra l'immagine in scala di grigi
        axs[0, i].set_title(images_names[i])  # Titolo sopra l'immagine
        axs[0, i].axis('off')  # Disabilita gli assi per una visualizzazione più pulita
    
    # Visualizza le immagini da `final_masks` (dopo flood fill)
    for i, img in enumerate(final_masks):
        axs[1, i].imshow(img, cmap='gray')  # Mostra l'immagine in scala di grigi
        axs[1, i].set_title(images_names[i])  # Titolo sopra l'immagine
        axs[1, i].axis('off')  # Disabilita gli assi per una visualizzazione più pulita
    
    # Visualizza le immagini finali da `final_color_masks`
    for i, img in enumerate(final_color_masks):
        axs[2, i].imshow(img, cmap='gray')  # Mostra l'immagine in scala di grigi
        axs[2, i].set_title(images_names[i])  # Titolo sopra l'immagine
        axs[2, i].axis('off')  # Disabilita gli assi per una visualizzazione più pulita
    
    plt.tight_layout()  # Ottimizza la disposizione degli elementi
    plt.subplots_adjust(top=0.93)  # Regola la posizione del titolo principale
    plt.show()

### TASK 2###
#uso sia funzioni del task 1 sia scrivo nuove funzioni per il task 2
def plot_images_2():
    fig, axes = plt.subplots(2, 2, figsize=(10,10))

    #Titoli per i due gruppi di immagini
    group_titles = ['NIR Images', 'Color Images']

    #Primo ciclo per i gruppi (Gruppo 1 (i=0), Gruppo 2 (i=1))
    for i in range(2): 
        #Secondo ciclo per le immagini (2 per group)
        for j in range(1,3):
            path = f"fruit-inspection-images/second_task/C{i}_00000{j+3}.png"

            img = cv.imread(path)

            if img is None:
                print(f"Immagine {path} non trovata...")
            else:
                ax = axes[i, j-1]
                ax.imshow(cv.cvtColor(img, cv.COLOR_BGR2RGB))
                ax.axis('off') #nascosti gli assi

                #title
                if j == 1:
                    if i == 0:
                        fig.text(0.5, 0.91, group_titles[i], ha='center', fontsize=20)
                    else:
                        fig.text(0.5, 0.49, group_titles[i], ha='center', fontsize=20)
    plt.show()

### TASK 3###
#uso sia funzioni del task 1 e 2 sia scrivo nuove funzioni per il task 3
def plot_images_3():
    fig, axes = plt.subplots(2, 5, figsize=(10,10))

    #Titoli per i due gruppi di immagini
    group_titles = ['NIR Images', 'Color Images']

    #Primo ciclo per i gruppi (Gruppo 1 (i=0), Gruppo 2 (i=1))
    for i in range(2): 
        #Secondo ciclo per le immagini (5 per group)
        for j in range(1,6):
            if j==5:
                path = f"fruit-inspection-images/final_challenge/C{i}_0000{j+5}.png"
            else:
                path = f"fruit-inspection-images/final_challenge/C{i}_00000{j+5}.png"

            img = cv.imread(path)

            if img is None:
                print(f"Immagine {path} non trovata...")
            else:
                ax = axes[i, j-1]
                ax.imshow(cv.cvtColor(img, cv.COLOR_BGR2RGB))
                ax.axis('off') #nascosti gli assi

                #title
                if j == 1:
                    if i == 0:
                        fig.text(0.5, 0.91, group_titles[i], ha='center', fontsize=20)
                    else:
                        fig.text(0.5, 0.49, group_titles[i], ha='center', fontsize=20)
    plt.show()