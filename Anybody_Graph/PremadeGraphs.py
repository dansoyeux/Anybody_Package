# -*- coding: utf-8 -*-
"""
Created on Mon Sep 18 09:24:20 2023

@author: user
"""

from Anybody_Package.Anybody_Graph.GraphFunctions import graph
from Anybody_Package.Anybody_Graph.GraphFunctions import COP_graph
from Anybody_Package.Anybody_Graph.GraphFunctions import muscle_graph
from Anybody_Package.Anybody_Graph.GraphFunctions import check_result_dictionary_data_structure

from Anybody_Package.Anybody_Graph.Tools import save_all_active_figures

import numpy as np


def graph_all_muscle_fibers(data, muscle_list, variable_x, variable_y, composante_y_muscle_part=["Total"], composante_y_muscle_combined=["Total"], cases_on=False, compare=False, **kwargs):
    """
    Trace tous les muscles parties par parties en les rassemblant par muscles
    adapte la taille du subplot en fonction nombre de parties (subplot [2, n])
    Le titre du graphique est le nom du muscle
    """

    # Get the muscle data depending on compare and caseson
    if compare is True:
        SimulationsList = list(data.keys())

        if cases_on is False:
            muscle_data = data[SimulationsList[0]]["Muscles"]
        else:
            cases_list = list(data[SimulationsList[0]].keys())
            muscle_data = data[SimulationsList[0]][cases_list[0]]["Muscles"]

    else:
        if cases_on is False:
            muscle_data = data["Muscles"]

        else:
            cases_list = list(data.keys())
            muscle_data = data[cases_list[0]]["Muscles"]

    for muscle_name in muscle_list:

        # -1 to not count the total for muscles with parts (muscle_name, muscle_name 1, muscle_name 2) = 3-1 = 2 parts
        # Gets the number of muscle parts (0 means there is only the total so 0 parts)
        number_of_parts = len(muscle_data[muscle_name]) - 1

        # Combined muscle graph
        muscle_graph(data, muscle_name, variable_x, variable_y, composante_y=composante_y_muscle_combined, figure_title=muscle_name, cases_on=cases_on, compare=compare, **kwargs)

        # Trace toutes les parties de muscles sur un subplot [2, n] en calculant le nombre n adéquat pour avoir tout sur un graphique
        if number_of_parts > 0:

            # Trouve la dimension en y la plus proche pour avoir un subplot 2xsubplot_Dimension_y (si number_of_parts est impair, le dernier graph sera vide)
            subplot_Dimension_y = int(np.ceil(number_of_parts / 2))

            subplotNumber = 1
            last_subplot = False

            # Parcours les numéros de partie de 1 à NombreDeParties
            for current_part_pumber in range(1, number_of_parts + 1, 1):

                # Specifies that it's the last subplot once we reach the last muscle part (even if the graph has one subplot empty)
                if current_part_pumber == number_of_parts:
                    last_subplot = True

                # Graph du muscle part
                muscle_graph(data, muscle_name, variable_x, variable_y, composante_y=composante_y_muscle_part, figure_title=muscle_name, cases_on=cases_on, compare=compare, subplot={"dimension": [2, subplot_Dimension_y], "number": subplotNumber, "last_subplot": last_subplot}, subplot_title=f"{muscle_name} {current_part_pumber}", muscle_part_on=[current_part_pumber], **kwargs)
                subplotNumber += 1


def muscle_graph_by_case_categories(data, case_categories, muscle_list, variable_x, variable_y, muscle_part_on=False, composante_y_muscle_part=["Total"], composante_y_muscle_combined=["Total"], **kwargs):
    """
    Trace les muscles contenus dans une liste et les sépare variable par variable (down, up, long, short...)
    Trace les parties de muscle individuellement si les muscles en ont

    Ce marche que avec des cas de simulation !!!

    muscle_part_on : Active ou non les parties de muscles par variables

    case_categories : dictionnaire
                   : détaille les variables à décomposer en valeurs
                   : 1 variable par ligne, chaque colonne correspond à une catégorie de valeur de cette variable

                   {"Nom_Variable_1": {"Titre_Catégorie_1": [Liste_Cases_Catégorie_1],
                                     "Titre_Catégorie_2": [Liste_Cases_Catégorie_2],},
                    "Nom_Variable_2": {"Titre_Catégorie_1": [Liste_Cases_Catégorie_1],
                                      "Titre_Catégorie_2": [Liste_Cases_Catégorie_2],}
                    }

    """

    # get the cases_list
    cases_list = list(data.keys())

    # Nombre de lignes dans le subplot (nombre de variables)
    n_subplot_lines = len(case_categories)

    # Nombre de catégories par variable
    n_categories = [len(case_categories[Category]) for Category in case_categories]

    # Nombre de colonnes dans le subplot (correspond au nombre maximal de catégories)
    n_subplot_columns = np.max(n_categories)

    # Get the muscle data
    muscle_data = data[cases_list[0]]["Muscles"]

    variables_list = list(case_categories.keys())

    last_subplot = False

    # Checks if the last category has the same number of columns than the subplot to always show the legend even if the last subplots are empty
    if len(case_categories[variables_list[-1]]) < n_subplot_columns:
        last_subplot_Number = (n_subplot_lines - 1) * n_subplot_columns + len(case_categories[variables_list[-1]])
    else:
        last_subplot_Number = n_subplot_lines * n_subplot_columns

    # Parcours la liste des muscles
    for muscle_name in muscle_list:

        subplot_Number = 1

        # -1 to not count the total for muscles with parts (muscle_name, muscle_name 1, muscle_name 2) = 3-1 = 2 parts
        # Gets the number of muscle parts (0 means there is only the total so 0 parts)
        number_of_parts = len(muscle_data[muscle_name]) - 1

        for Category_index, Category in enumerate(case_categories):

            # Parcours les catégories de la variable (1 catégorie par colomne)
            for Categorie_Name, Categories_Cases in case_categories[Category].items():
                if subplot_Number == last_subplot_Number:
                    last_subplot = True

                # Graph of the combined muscle
                muscle_graph(data, muscle_name, variable_x, variable_y, composante_y=composante_y_muscle_combined, figure_title=f"{muscle_name} : {variable_y} {composante_y_muscle_combined[0]}", cases_on=Categories_Cases, subplot={"dimension": [n_subplot_lines, n_subplot_columns], "number": subplot_Number, "last_subplot": last_subplot}, subplot_title=Categorie_Name, **kwargs)

                # Incrémente le numéro de subplot
                subplot_Number += 1

            # Met le numéro de subplot à la première case de la bonne colomne (utile si une case est vide car il n'y a pas le même nombre de catégories entre les variables)
            subplot_Number = (Category_index + 1) * n_subplot_columns + 1

        # Graphiques des parties individuelles des muscles si activé
        if muscle_part_on is True:

            # Trace les parties de muscles individuelles seulement si le muscle en a
            if number_of_parts > 0:

                # Graphique par variables par partie de muscle
                for current_part_pumber in range(1, number_of_parts + 1, 1):

                    subplot_Number = 1
                    for Category_index, Category in enumerate(case_categories):

                        if subplot_Number == last_subplot_Number:
                            last_subplot = True

                        # Parcours les catégories de la variable (1 catégorie par colomne)
                        for Categorie_Name, Categories_Cases in case_categories[Category].items():

                            muscle_graph(data, muscle_name, variable_x, variable_y, composante_y=composante_y_muscle_part, figure_title=f"{muscle_name} {current_part_pumber} : {variable_y} {composante_y_muscle_part[0]}", cases_on=Categories_Cases, subplot={"dimension": [n_subplot_lines, n_subplot_columns], "number": subplot_Number, "last_subplot": last_subplot}, subplot_title=Categorie_Name, muscle_part_on=[current_part_pumber], **kwargs)

                            # Incrémente le numéro de subplot
                            subplot_Number += 1

                        # Met le numéro de subplot à la première case de la bonne colomne (utile si une case est vide car il n'y a pas le même nombre de catégories entre les variables)
                        subplot_Number = (Category_index + 1) * n_subplot_columns + 1


def muscle_graph_from_list(data, muscle_list, subplot_dimension, variable_x, variable_y, figure_title, cases_on=False, compare=False, composante_y=["Total"], **kwargs):
    """
    Graph of a muscle list
    Must specify the dimension of the subplot

    Can compare, all cases from a list of cases (cases_on)
    """

    n_subplot = subplot_dimension[0] * subplot_dimension[1]

    # Cheks that the subplot is big enough to draw every muscles
    if n_subplot < len(muscle_list):
        raise ValueError(f"The subplot is too small to plot every muscles in the list.\n{subplot_dimension} = {n_subplot} subplots < {len(muscle_list)} muscles \nIncrease the dimensions of the subplot (subplot_dimension)")
        return

    last_subplot = False
    # Parcours tous les muscles de la liste
    for index, muscle_name in enumerate(muscle_list):
        # Activates the legend once the end of the list is reached even if the subplot is too big for the list
        if muscle_name == muscle_list[-1]:
            last_subplot = True

        muscle_graph(data, muscle_name, variable_x, variable_y, figure_title, composante_y=composante_y, compare=compare, cases_on=cases_on, subplot_title=muscle_name, subplot={"dimension": subplot_dimension, "number": index + 1, "last_subplot": last_subplot}, **kwargs)


def COP_graph_by_case_categories(data, case_categories, COP_contour=None, variable="COP", figure_title="", composantes=["x", "y"], DrawCOPPointsOn=True, **kwargs):
    """
    crée un subplot où dans chaque case on ne trace qu'une liste de cas


    case_categories : dictionnaire
                   : détaille catégories de cas de simulation
                   : On crée une entrée de dictionnaire qui correspond à une ligne, et ensuite on donne un nom à une liste de noms de cas de simulations dans cette catégorie

                   {"Ligne_1": {"Titre_Catégorie_1": [Liste_Cases_Catégorie_1],
                                     "Titre_Catégorie_2": [Liste_Cases_Catégorie_2],},
                    "Ligne_2": {"Titre_Catégorie_1": [Liste_Cases_Catégorie_1],
                                      "Titre_Catégorie_2": [Liste_Cases_Catégorie_2],}
                    }
    """

    # Nombre de lignes dans le subplot (nombre de variables)
    n_subplot_lines = len(case_categories)

    # Nombre de catégories par variable
    n_categories = [len(case_categories[Category]) for Category in case_categories]

    # Nombre de colonnes dans le subplot (correspond au nombre maximal de catégories)
    n_subplot_columns = np.max(n_categories)

    subplot_Number = 1
    last_subplot = False

    variables_list = list(case_categories.keys())

    # Checks if the last category has the same number of columns than the subplot to always show the legend even if the last subplots are empty
    if len(case_categories[variables_list[-1]]) < n_subplot_columns:
        last_subplot_Number = (n_subplot_lines - 1) * n_subplot_columns + len(case_categories[variables_list[-1]])
    else:
        last_subplot_Number = n_subplot_lines * n_subplot_columns

    # Parcours les variables, toutes les catégories de variable sera placée sur une même ligne
    for Category_index, Category in enumerate(case_categories):

        # Parcours les catégories de la variable (1 catégorie par colomne)
        for Categorie_Name, Categories_Cases in case_categories[Category].items():

            if subplot_Number == last_subplot_Number:
                last_subplot = True

            COP_graph(data, COP_contour, variable=variable, figure_title=figure_title, composantes=composantes, cases_on=Categories_Cases, subplot={"dimension": [n_subplot_lines, n_subplot_columns], "number": subplot_Number, "last_subplot": last_subplot},
                      DrawCOPPointsOn=DrawCOPPointsOn, subplot_title=Categorie_Name, **kwargs)

            # Incrémente le numéro de subplot
            subplot_Number += 1

        # Met le numéro de subplot à la première case de la bonne colomne (utile si une case est vide car il n'y a pas le même nombre de catégories entre les variables)
        subplot_Number = (Category_index + 1) * n_subplot_columns + 1


def graph_by_case_categories(data, case_categories, variable_x, variable_y, figure_title, composante_y=["Total"], **kwargs):
    """
    Graphique normal par catégories de cas de simulation
    sans comparaison
    """

    # Nombre de lignes dans le subplot (nombre de variables)
    n_subplot_lines = len(case_categories)

    # Nombre de catégories par variable
    n_categories = [len(case_categories[Category]) for Category in case_categories]

    # Nombre de colonnes dans le subplot (correspond au nombre maximal de catégories)
    n_subplot_columns = np.max(n_categories)

    subplot_Number = 1
    last_subplot = False

    variables_list = list(case_categories.keys())

    # Checks if the last category has the same number of columns than the subplot to always show the legend even if the last subplots are empty
    if len(case_categories[variables_list[-1]]) < n_subplot_columns:
        last_subplot_Number = (n_subplot_lines - 1) * n_subplot_columns + len(case_categories[variables_list[-1]])
    else:
        last_subplot_Number = n_subplot_lines * n_subplot_columns

    for Category_index, Category in enumerate(case_categories):

        # Parcours les catégories de la variable (1 catégorie par colomne)
        for Categorie_Name, Categories_Cases in case_categories[Category].items():
            if subplot_Number == last_subplot_Number:
                last_subplot = True

            graph(data, variable_x, variable_y, figure_title, composante_y=composante_y, cases_on=Categories_Cases, subplot={"dimension": [n_subplot_lines, n_subplot_columns], "number": subplot_Number, "last_subplot": last_subplot}, subplot_title=Categorie_Name, **kwargs)

            # Incrémente le numéro de subplot
            subplot_Number += 1

        # Met le numéro de subplot à la première case de la bonne colomne (utile si une case est vide car il n'y a pas le même nombre de catégories entre les variables)
        subplot_Number = (Category_index + 1) * n_subplot_columns + 1
