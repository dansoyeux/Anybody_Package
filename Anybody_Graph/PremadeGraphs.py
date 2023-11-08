# -*- coding: utf-8 -*-
"""
Created on Mon Sep 18 09:24:20 2023

@author: user
"""

import Anybody_Package.Anybody_Graph.GraphFunctions as Graph
import numpy as np


def graph_all_muscle_fibers(data, MuscleList, variable_x, variable_y, composante_y_muscle_part=["Total"], composante_y_muscle_combined=["Total"], cases_on=False, compare=False, **kwargs):
    """
    Trace tous les muscles parties par parties en les rassemblant par muscles
    adapte la taille du subplot en fonction nombre de parties (subplot [2, n])
    Le titre du graphique est le nom du muscle
    """

    # Get the muscle data depending on compare and caseson
    if compare is True:
        SimulationsList = list(data.keys())

        if cases_on is False:
            MuscleData = data[SimulationsList[0]]["Muscles"]
        else:
            CasesList = list(data[SimulationsList[0]].keys())
            MuscleData = data[SimulationsList[0]][CasesList[0]]["Muscles"]

    else:
        if cases_on is False:
            MuscleData = data["Muscles"]

        else:
            CasesList = list(data.keys())
            MuscleData = data[CasesList[0]]["Muscles"]

    for MuscleName in MuscleList:

        # -1 to not count the total for muscles with parts (MuscleName, MuscleName 1, MuscleName 2) = 3-1 = 2 parts
        # Gets the number of muscle parts (0 means there is only the total so 0 parts)
        Number_Of_Parts = len(MuscleData[MuscleName]) - 1

        # Combined muscle graph
        Graph.muscle_graph(data, MuscleName, variable_x, variable_y, composante_y=composante_y_muscle_combined, figure_title=MuscleName, cases_on=cases_on, compare=compare, **kwargs)

        # Trace toutes les parties de muscles sur un subplot [2, n] en calculant le nombre n adéquat pour avoir tout sur un graphique
        if Number_Of_Parts > 0:

            # Trouve la dimension en y la plus proche pour avoir un subplot 2xsubplot_Dimension_y (si Number_Of_Parts est impair, le dernier graph sera vide)
            subplot_Dimension_y = int(np.ceil(Number_Of_Parts / 2))

            subplotNumber = 1
            last_subplot = False

            # Parcours les numéros de partie de 1 à NombreDeParties
            for PartNumber in range(1, Number_Of_Parts + 1, 1):

                # Specifies that it's the last subplot once we reach the last muscle part (even if the graph has one subplot empty)
                if PartNumber == Number_Of_Parts:
                    last_subplot = True

                # Graph du muscle part
                Graph.muscle_graph(data, MuscleName, variable_x, variable_y, composante_y=composante_y_muscle_part, figure_title=MuscleName, cases_on=cases_on, compare=compare, subplot={"dimension": [2, subplot_Dimension_y], "number": subplotNumber, "last_subplot": last_subplot}, subplot_title=f"{MuscleName} {PartNumber}", MusclePartOn=[PartNumber], **kwargs)
                subplotNumber += 1


def muscle_graph_by_variable(data, CasesCategories, MuscleList, variable_x, variable_y, MusclePartOn=False, composante_y_muscle_part=["Total"], composante_y_muscle_combined=["Total"], **kwargs):
    """
    Trace les muscles contenus dans une liste et les sépare variable par variable (down, up, long, short...)
    Trace les parties de muscle individuellement si les muscles en ont

    Ce marche que avec des cas de simulation !!!

    MusclePartOn : Active ou non les parties de muscles par variables

    CasesCategories : dictionnaire
                   : détaille les variables à décomposer en valeurs
                   : 1 variable par ligne, chaque colonne correspond à une catégorie de valeur de cette variable

                   {"Nom_Variable_1": {"Titre_Catégorie_1": [Liste_Cases_Catégorie_1],
                                     "Titre_Catégorie_2": [Liste_Cases_Catégorie_2],},
                    "Nom_Variable_2": {"Titre_Catégorie_1": [Liste_Cases_Catégorie_1],
                                      "Titre_Catégorie_2": [Liste_Cases_Catégorie_2],}
                    }

    """

    # get the CasesList
    CasesList = list(data.keys())

    # Nombre de lignes dans le subplot (nombre de variables)
    n_Lines = len(CasesCategories)

    # Nombre de catégories par variable
    n_Categories = [len(CasesCategories[Category]) for Category in CasesCategories]

    # Nombre de colonnes dans le subplot (correspond au nombre maximal de catégories)
    n_Columns = np.max(n_Categories)

    # Get the muscle data
    MuscleData = data[CasesList[0]]["Muscles"]

    List_Variables = list(CasesCategories.keys())

    last_subplot = False

    # Checks if the last category has the same number of columns than the subplot to always show the legend even if the last subplots are empty
    if len(CasesCategories[List_Variables[-1]]) < n_Columns:
        last_subplot_Number = (n_Lines - 1) * n_Columns + len(CasesCategories[List_Variables[-1]])
    else:
        last_subplot_Number = n_Lines * n_Columns

    # Parcours la liste des muscles
    for MuscleName in MuscleList:

        subplot_Number = 1

        # -1 to not count the total for muscles with parts (MuscleName, MuscleName 1, MuscleName 2) = 3-1 = 2 parts
        # Gets the number of muscle parts (0 means there is only the total so 0 parts)
        Number_Of_Parts = len(MuscleData[MuscleName]) - 1

        for Category_index, Category in enumerate(CasesCategories):

            # Parcours les catégories de la variable (1 catégorie par colomne)
            for Categorie_Name, Categories_Cases in CasesCategories[Category].items():
                if subplot_Number == last_subplot_Number:
                    last_subplot = True

                # Graph of the combined muscle
                Graph.muscle_graph(data, MuscleName, variable_x, variable_y, composante_y=composante_y_muscle_combined, figure_title=f"{MuscleName}", cases_on=Categories_Cases, subplot={"dimension": [n_Lines, n_Columns], "number": subplot_Number, "last_subplot": last_subplot}, subplot_title=Categorie_Name, **kwargs)

                # Incrémente le numéro de subplot
                subplot_Number += 1

            # Met le numéro de subplot à la première case de la bonne colomne (utile si une case est vide car il n'y a pas le même nombre de catégories entre les variables)
            subplot_Number = (Category_index + 1) * n_Columns + 1

        # Graphiques des parties individuelles des muscles si activé
        if MusclePartOn is True:

            # Trace les parties de muscles individuelles seulement si le muscle en a
            if Number_Of_Parts > 0:

                # Graphique par variables par partie de muscle
                for PartNumber in range(1, Number_Of_Parts + 1, 1):

                    subplot_Number = 1
                    for Category_index, Category in enumerate(CasesCategories):

                        if subplot_Number == last_subplot_Number:
                            last_subplot = True

                        # Parcours les catégories de la variable (1 catégorie par colomne)
                        for Categorie_Name, Categories_Cases in CasesCategories[Category].items():

                            Graph.muscle_graph(data, MuscleName, variable_x, variable_y, composante_y=composante_y_muscle_part, figure_title=f"{MuscleName} {PartNumber}", cases_on=Categories_Cases, subplot={"dimension": [n_Lines, n_Columns], "number": subplot_Number, "last_subplot": last_subplot}, subplot_title=Categorie_Name, MusclePartOn=[PartNumber], **kwargs)

                            # Incrémente le numéro de subplot
                            subplot_Number += 1

                        # Met le numéro de subplot à la première case de la bonne colomne (utile si une case est vide car il n'y a pas le même nombre de catégories entre les variables)
                        subplot_Number = (Category_index + 1) * n_Columns + 1


def muscle_graph_from_list(data, MuscleList, subplot_dimension, variable_x, variable_y, figure_title, cases_on=False, compare=False, composante_y=["Total"], **kwargs):
    """
    Graph of a muscle list
    Must specify the dimension of the subplot

    Can compare, all cases from a list of cases (cases_on)
    """

    n_subplot = subplot_dimension[0] * subplot_dimension[1]

    # Cheks that the subplot is big enough to draw every muscles
    if n_subplot < len(MuscleList):
        raise ValueError(f"The subplot is too small to plot every muscles in the list.\n{subplot_dimension} = {n_subplot} subplots < {len(MuscleList)} muscles \nIncrease the dimensions of the subplot (subplot_dimension)")
        return

    last_subplot = False
    # Parcours tous les muscles de la liste
    for index, MuscleName in enumerate(MuscleList):
        # Activates the legend once the end of the list is reached even if the subplot is too big for the list
        if MuscleName == MuscleList[-1]:
            last_subplot = True

        Graph.muscle_graph(data, MuscleName, variable_x, variable_y, figure_title, composante_y=composante_y, compare=compare, cases_on=cases_on, subplot_title=MuscleName, subplot={"dimension": subplot_dimension, "number": index + 1, "last_subplot": last_subplot}, **kwargs)


def COP_graph_by_variable(data, CasesCategories, COP_contour=None, Variable="COP", figure_title="", Composantes=["x", "y"], DrawCOPPointsOn=True, **kwargs):
    """
    Trace le COP et rassemble par variables
    subplots [3, 3] (tilt, acromion, CSA)

    CasesCategories : dictionnaire
                   : détaille les variables à décomposer en valeurs
                   : 1 variable par ligne, chaque colonne correspond à une catégorie de valeur de cette variable

                   {"Nom_Variable_1": {"Titre_Catégorie_1": [Liste_Cases_Catégorie_1],
                                     "Titre_Catégorie_2": [Liste_Cases_Catégorie_2],},
                    "Nom_Variable_2": {"Titre_Catégorie_1": [Liste_Cases_Catégorie_1],
                                      "Titre_Catégorie_2": [Liste_Cases_Catégorie_2],}
                    }
    """

    # Nombre de lignes dans le subplot (nombre de variables)
    n_Lines = len(CasesCategories)

    # Nombre de catégories par variable
    n_Categories = [len(CasesCategories[Category]) for Category in CasesCategories]

    # Nombre de colonnes dans le subplot (correspond au nombre maximal de catégories)
    n_Columns = np.max(n_Categories)

    subplot_Number = 1
    last_subplot = False

    List_Variables = list(CasesCategories.keys())

    # Checks if the last category has the same number of columns than the subplot to always show the legend even if the last subplots are empty
    if len(CasesCategories[List_Variables[-1]]) < n_Columns:
        last_subplot_Number = (n_Lines - 1) * n_Columns + len(CasesCategories[List_Variables[-1]])
    else:
        last_subplot_Number = n_Lines * n_Columns

    # Parcours les variables, toutes les catégories de variable sera placée sur une même ligne
    for Category_index, Category in enumerate(CasesCategories):

        # Parcours les catégories de la variable (1 catégorie par colomne)
        for Categorie_Name, Categories_Cases in CasesCategories[Category].items():

            if subplot_Number == last_subplot_Number:
                last_subplot = True

            Graph.COP_graph(data, COP_contour, Variable=Variable, figure_title=figure_title, Composantes=Composantes, cases_on=Categories_Cases, subplot={"dimension": [n_Lines, n_Columns], "number": subplot_Number, "last_subplot": last_subplot},
                            DrawCOPPointsOn=DrawCOPPointsOn, subplot_title=Categorie_Name, **kwargs)

            # Incrémente le numéro de subplot
            subplot_Number += 1

        # Met le numéro de subplot à la première case de la bonne colomne (utile si une case est vide car il n'y a pas le même nombre de catégories entre les variables)
        subplot_Number = (Category_index + 1) * n_Columns + 1


def graph_by_variable(data, CasesCategories, variable_x, variable_y, figure_title, composante_y=["Total"], **kwargs):
    """
    Graphique [3, 3] par catégories de variables
    sans comparaison
    """

    # Nombre de lignes dans le subplot (nombre de variables)
    n_Lines = len(CasesCategories)

    # Nombre de catégories par variable
    n_Categories = [len(CasesCategories[Category]) for Category in CasesCategories]

    # Nombre de colonnes dans le subplot (correspond au nombre maximal de catégories)
    n_Columns = np.max(n_Categories)

    subplot_Number = 1
    last_subplot = False

    List_Variables = list(CasesCategories.keys())

    # Checks if the last category has the same number of columns than the subplot to always show the legend even if the last subplots are empty
    if len(CasesCategories[List_Variables[-1]]) < n_Columns:
        last_subplot_Number = (n_Lines - 1) * n_Columns + len(CasesCategories[List_Variables[-1]])
    else:
        last_subplot_Number = n_Lines * n_Columns

    for Category_index, Category in enumerate(CasesCategories):

        # Parcours les catégories de la variable (1 catégorie par colomne)
        for Categorie_Name, Categories_Cases in CasesCategories[Category].items():
            if subplot_Number == last_subplot_Number:
                last_subplot = True

            Graph.graph(data, variable_x, variable_y, figure_title, composante_y=composante_y, cases_on=Categories_Cases, subplot={"dimension": [n_Lines, n_Columns], "number": subplot_Number, "last_subplot": last_subplot}, subplot_title=Categorie_Name, **kwargs)

            # Incrémente le numéro de subplot
            subplot_Number += 1

        # Met le numéro de subplot à la première case de la bonne colomne (utile si une case est vide car il n'y a pas le même nombre de catégories entre les variables)
        subplot_Number = (Category_index + 1) * n_Columns + 1
