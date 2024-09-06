def find_peak_indexes(Coordinates, find_max_index=True, find_min_index=False, find_max_peak_index=False, find_min_peak_index=False):
    """
    Fonction qui trouve les indices de la liste où la coordonnée a atteint des pics (maximum et minimum)

    Coordinates : Liste des coordonnées dont on doit trouver les pics

    find_max_index : bool
        : True : trouve la position du maximum

    find_min_index : bool
        : True : trouve la position du minimum

    find_max_peak_index : bool
        : True : trouve la position du plus grand pic vers le haut

    find_min_index : bool
        : True : trouve la position du plus grand pic vers le bas

    Par défaut : Max activé et les autres désactivés


    pics à trouver : find_max, find_min : minimum et maximum absolu
    find_max_peak, find_min_peak : trouve tous les pics (les pics minimas et maximas)

    Si plusieurs activés, fusionne les index

    ----------------------------
    return

    Peak_Indexes : Liste des indices de la liste Coordinates où les pics sélectionnés ont été atteints


    """
    min_index = []
    max_index = []
    min_peak_indexes = []
    max_peak_indexes = []

    from scipy.signal import find_peaks
    import numpy as np

    if find_max_index:
        max_index = [np.argmax(Coordinates)]

    if find_min_index:
        min_index = [np.argmin(Coordinates)]

    if find_max_peak_index:
        # All peak indexes
        max_peak_indexes = find_peaks(Coordinates)[0]

        if len(max_peak_indexes) > 1:
            # Finds the highest peak
            highest_peak_index = np.argmax(Coordinates[max_peak_indexes])

            # Selects only this highest peak
            max_peak_indexes = np.array([max_peak_indexes[highest_peak_index]])

    # Multiplie par -1 pour trouver le pic minimal
    if find_min_peak_index:
        min_peak_indexes = find_peaks(-1 * Coordinates)[0]
        if len(min_peak_indexes) > 1:
            # Finds the lowest peak
            lowest_peak_index = np.argmin(Coordinates[min_peak_indexes])

            # Selects only this lowest peak
            min_peak_indexes = np.array([min_peak_indexes[lowest_peak_index]])

    peak_indexes = [*min_index, *max_index, *min_peak_indexes, *max_peak_indexes]

    return peak_indexes


def read_picked_points(dataPath):
    """
    Reads a .pp file and converts it to a numpy array of points
    """
    # Module that can read .pp files can be found here : https://pypi.org/project/meshlab-pickedpoints/
    import meshlab_pickedpoints
    import numpy as np

    data = meshlab_pickedpoints.load(dataPath)

    mat = np.zeros([len(data), 3])

    for i in range(len(data)):

        for j in range(3):
            mat[i, j] = data[i]['point'][j]

    return mat


def get_text_position(texts):
    """
    Obtiens les coordonnées des textes et annotations placés dans le graphique qui est en train d'être tracé

    ------------
    returns

    x, y : numpy.array contenant les coordonnées des textes dans le graphique en cours
    """
    import matplotlib.pyplot as plt
    import numpy as np

    # the axis is the current axis
    ax = plt.gca()

    x = np.zeros(len(texts))
    y = np.zeros(len(texts))

    for i in range(len(texts)):

        x_i, y_i = texts[i].get_position()

        # Converts the coordinates to the units of the plot
        ax.xaxis.convert_units(x_i)
        ax.xaxis.convert_units(y_i)

        x[i] = x_i
        y[i] = y_i

    return x, y


def unsuperpose_plot_annotations(annotation_offset=None, annotation_reference_offset=None, update_xlim=False, update_ylim=False, annotation_reference_mode="max_y", **kwargs):
    """
    Utilisé par COPGraph pour déplacer les annotations d'angle de pic de COP pour qu'elles ne se superposent pas.
    Le script prend comme référence d'offset la taille du pic le plus haut (le pic de référence)

    Déplace l'annotation la plus haute de annotation_reference_offset = [x, y] * Taille_du_texte_de_référence
    et range les suivantes écartéees de annotation_offset = [x, y] * Taille_du_texte_de_référence

    annotation_reference_offset = Facteur multiplicateur du Déplacement [x, y] du texte de référence par rapport à sa position d'origine
                        [0, 0] = Pas de déplacement

    annotation_offset = Facteur multiplicateur de l'écart [x, y] entre les annotations

    Ces deux offsets sont exprimés dans les unités du graphique

    update_xlim : bool : contrôle le fait que l'on ajuste ou non les limites de l'axe x dans le cas où des annotations sortiraient du graphique
                         : False par defaut pour gagner en performance

    update_ylim : bool : contrôle le fait que l'on ajuste ou non les limites de l'axe y dans le cas où des annotations sortiraient du graphique
                         : False par defaut pour gagner en performance

    annotation_reference_mode : str : "min_y", "max_y", "min_x" or "max_x" : sets the reference annotation creation mode

        : selects which annotation will be the reference annotation to move the others and in which order they will be created
        Default = "max_y"
        and the annotation will be created along the coordinate of the annotation

        ex : "max_y" : reference = l'annotation avec le plus grand y, les annotations sont bougées du plus grand y au plus petit y
        ex : "min_x" : reference = l'annotation avec le plus petit x, les annotations sont bougées du plus petit x au plus grand x

    """

    import matplotlib.pyplot as plt
    import numpy as np

    # Default offset values if these arguments were not specified
    if annotation_offset is None:
        annotation_offset = [0.8, -2.1]

    if annotation_reference_offset is None:
        annotation_reference_offset = [0, 3]

    # obtiens le subplot en train d'etre tracé
    ax = plt.gca()

    texts = ax.texts

    # exit the unsuperpose function if no annotations were found
    if len(texts) == 0:
        return

    # Obtiens les positions des annotations
    x, y = get_text_position(texts)

    # Trie les indexes des annotations en les classant selon le mode choisi
    if annotation_reference_mode == "min_y":
        sort_y_indexes = np.argsort(y)
    elif annotation_reference_mode == "max_y":
        sort_y_indexes = np.argsort(y)[::-1]

    elif annotation_reference_mode == "min_x":
        sort_y_indexes = np.argsort(x)
    elif annotation_reference_mode == "max_x":
        sort_y_indexes = np.argsort(x)[::-1]

    # Position l'annotation la plus haute en y
    reference_annotation_index = sort_y_indexes[0]

    # Number of texts that were moved
    number_text_moved = 0

    # Axis limits must be called so that the limits aren't reset, and so it doesn't break the annotation movement
    xl, yl = ax.get_xlim(), ax.get_ylim()

    # Gets the bbox properties in the data coordinate system
    bbox = ax.transData.inverted().transform_bbox(texts[reference_annotation_index].get_window_extent())

    text_height = bbox.height

    text_width = bbox.width

    offset_between_texts = [text_width * annotation_offset[0], text_height * annotation_offset[1]]

    reference_annotation_offset = [text_width * annotation_reference_offset[0], text_height * annotation_reference_offset[1]]

    # Nouvelle coordonnée du pic le plus haut en y qui sert de base pour placer les autres
    new_max_peak_coordinates = [x[reference_annotation_index] + reference_annotation_offset[0], y[reference_annotation_index] + reference_annotation_offset[1]]

    # Parcours les index de la liste des textes rangés du plus haut au plus bas
    for index, peak_index in enumerate(sort_y_indexes):

        # Pour le maximum qui est en premier dans la liste, le place en fonction de reference_annotation_offset
        if index == 0:
            texts[peak_index].set_position((new_max_peak_coordinates[0], new_max_peak_coordinates[1]))
            # texts[peak_index].set_position((6, 15))

        # Pour les autres textes, les places avec un écart de offset_between_texts entre eux
        # On multiplie cet offset par le nombre de text qui ont été bougés
        else:
            texts[peak_index].set_position((new_max_peak_coordinates[0] + number_text_moved * offset_between_texts[0], new_max_peak_coordinates[1] + number_text_moved * offset_between_texts[1]))

        number_text_moved += 1

    # Updates the limits of the graph so no annotation are out of the subplot box
    if update_xlim or update_ylim:
        # Draws to be able to take the texts into account in the update of the limits
        plt.draw()

        # gets the new axis and texts informations
        ax_2 = plt.gca()
        texts = ax_2.texts

        annotation_corners = np.array([[], []]).T

        # Goes through all the texts and adds their corner dimensions to the Corner array
        for text in texts:

            bbox = text.get_window_extent()

            # Transforms the bbox object into the data coordinate system
            bbox_data = bbox.transformed(ax_2.transData.inverted())

            # Gets the text corners coordinates
            annotation_corners = np.append(annotation_corners, bbox_data.corners(), axis=0)

        # Uses the annotation_corners array to update the axis limits
        ax_2.update_datalim(annotation_corners, updatex=update_xlim, updatey=update_ylim)

        ax_2.autoscale_view()


def save_all_active_figures(save_folder_path, folder_name, file_name, save_format="png"):
    """Function that saves all the active figuresand saves them in a subfolder"""

    import os
    import matplotlib.pyplot as plt

    subfolder_path = f"{save_folder_path}/{folder_name}"
    # Creates the category folder
    os.mkdir(subfolder_path)

    # Get all active figures and save them
    for i in plt.get_fignums():
        plt.figure(i)
        plt.savefig(f"{subfolder_path}/{file_name}_{i}.{save_format}", bbox_inches='tight')
    plt.close("all")

    print(f"Figure saved in the folder : {save_folder_path}/{folder_name}")
