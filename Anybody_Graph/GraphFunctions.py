# -*- coding: utf-8 -*-
"""
Created on Thu Jul 20 09:53:59 2023
Functions to make graph from EpauleFDK .h5 and AnyFileOut
Makes variable graps, muscle graphs and COP_graph
@author: Dan
"""

import math
import numpy as np
import sys

import matplotlib.pyplot as plt

from Anybody_Package.Anybody_Graph.Tools import find_peak_indexes
from Anybody_Package.Anybody_Graph.Tools import read_picked_points
from Anybody_Package.Anybody_Graph.Tools import unsuperpose_plot_annotations

# %% Plot Variables setup


def define_simulations_line_style(SimulationsLineStyleDictionary):
    """
    Fonction qui crée la variable globale SimulationsLineStyle qui sera utilisée par la fonction get_simulation_line_style
    Fonction à appeler avant d'utiliser la fonction plot

    SimulationsLineStyleDictionary = {"Case 1": {"color": "COLORNAME", "marker": "MARKERNAME", "markersize": NUMBER, "linestyle": "LINESTYLE", "linewidth": NUMBER},
                            "Case 2": {"color": "COLORNAME", "marker": "MARKERNAME", "markersize": NUMBER, "linestyle": "LINESTYLE", "linewidth": NUMBER}
                            }
    If one of the settings isn't declared, it will be set to the default value (None)
    """
    global SimulationsLineStyle
    SimulationsLineStyle = SimulationsLineStyleDictionary.copy()


def define_simulation_description(SimulationDescriptionDictionary):
    """
    Function that sets the SimulationDescriptions to be used by the DefineSumulationLabel and get_simulation_description functions

    SimulationDescriptionDictionary = {"Simulation1 Name": "Simulation1 description",
                                       "Simulation2 Name": "Simulation2 description"...}

    """

    global simulation_description
    simulation_description = SimulationDescriptionDictionary.copy()


def flatten_compared_simulations(compared_simulations_data, case_on):
    """
    Function for graph that flattens the compared simulation result dictionary into a simulation cases dictionary to then be grap

    case_on = name of the case to select

    returns the dictionary flatten and the new list of cases
    """

    flatten_data = {}

    for simulation_name, simulation_data in compared_simulations_data.items():

        try:
            simulation_case = simulation_data[case_on]
            flatten_data[simulation_name] = simulation_case
        except:
            print(
                f"The simulation case : {case_on} doesn't exist in the simulation : {simulation_name}")

    cases_on = list(compared_simulations_data.keys())

    return flatten_data, cases_on


# %% Plot Setup Functions


def plot_graph_functions(data, x, y, graph_type, label=None, custom_label=None, **kwargs):
    """
    Function ploting the datas

    draw_GH_reactions_nodes= ARGUMENT PERSONNEL POUR TRACER EN PLUS DES POINT SUR LE CONTOUR, PEUT ÊTRE SUPPRIMÉ


    graph_annotation_on : Contrôle l'affichage ou non des angles de pic de COP (Vrai par défaut)
                       : Si True appelle la fonction draw_graph_annotation


    draw_COP_points_on : bool : active ou non le traçage des points sur le COP
                    : Si True appelle la fonction draw_COP_points

    # The style depends on the graph_type : "graph", "muscle_graph" "COP_graph"

    """

    variable_x = kwargs["variable_x"]
    variable_y = kwargs["variable_y"]

    graph_annotation_on = kwargs.get("graph_annotation_on", False)

    # if a custom label has been declared, it will overwrite the label given by the graph function
    if custom_label is not None:
        graph_label = custom_label
    else:
        graph_label = label

    # defines the color and the style of the line depending on the label name, from the global dictionary SimulationsLineStyleDictionary
    simulation_style_dictionary = get_simulation_line_style(graph_label)

    plt.plot(x, y, label=graph_label, **simulation_style_dictionary)

    # Draws peak angles
    if graph_annotation_on:

        # Annotation angle settings
        if graph_type == "COP_graph":
            # Variable
            annotation_variable = kwargs.get("annotation_variable", "Abduction")
        else:
            # Variable to draw in the peak annotation is the x variable
            annotation_variable = kwargs.get("annotation_variable", variable_x)

        # Composante to chose
        annotation_composante = kwargs.get("annotation_composante", "Total")

        peak_annotation_values = data[annotation_variable][annotation_composante]

        draw_graph_annotation(peak_annotation_values, x, y, **kwargs)

    if graph_type == "COP_graph":

        draw_COP_points_on = kwargs.get("draw_COP_points_on", False)

        # Defines the COP points function settings
        COP_points_variable = kwargs.get("COP_points_variable", "Abduction")
        COP_points_composante = kwargs.get("COP_points_composante", "Total")

        # Step de déplacement angulaire où les points seront tracés par la fonction draw_COP_points
        COP_points_step = kwargs.get("COP_points_step", 15)

        # Liste des angles de déplacement
        COP_points_coordinates = data[COP_points_variable][COP_points_composante]

        # Dessine le COP de début et d'autres points intermédiaires tous les x° d'angles de déplacement (COP_points_step)
        # Seulement si activé
        if draw_COP_points_on:
            draw_COP_points(COP_points_coordinates, x, y, COP_points_step)

        """
        CODE PERSONNEL POUR TRACER LES POINTS ORIGINE GHREACTIONS
        """
        draw_GH_reactions_nodes = kwargs.get("draw_GH_reactions_nodes", False)
        # If reactions nodes drawing is activated and if this in this case, there are GHReactions nodes in the current data
        if draw_GH_reactions_nodes and ("GHReactions" in list(data.keys())):
            CavityEdgeNodes = list(
                data["GHReactions"]["Cavity Nodes Position"].keys())
            NodePosx = np.array([])
            NodePosy = np.array([])

            # Gets the position of the CavityEdgenodes
            for Node in CavityEdgeNodes:
                NodePosx = np.append(
                    NodePosx, data["GHReactions"]["Cavity Nodes Position"][Node][0])
                NodePosy = np.append(
                    NodePosy, data["GHReactions"]["Cavity Nodes Position"][Node][1])

            # Draws the Cavity nodes on the COP graph
            plt.scatter(NodePosx, NodePosy, color=plt.gca(
            ).lines[-1].get_color(), marker='o', s=40)


def subplot_setup(subplot, figsize=None, add_graph=False):
    """

    Setup a subplot of dimension :
    subplot = {"Dimension: [nrows, ncolumns]", :"number": Number_of_the_subplot_selected, "figsize": [horizontal_size_inches, vertical_size_inches], "last_subplot": True}


        subplot["dimension"] = [nrows, ncolumns]
    And defines the active axis as the subplot["number"]=number of the plot


    subplot["figsize"] : Optional argument to set the size of the figure
                         subplot["figsize"] = [horizontal_size_inches, vertical_size_inches]
                         : default : [14, 10] inches for 2D ; [7, 5] for [1,1] subplot

    subplot["dimension"] and figsize : are only to be set for subplot["number"] = 1
    They are not taken in account otherwise

    subplot["LastPart"] : bool = Optional argument : Controls if the legend and figure title are drawn
                        : It's automatically set to True if we reach the maximum subplot Number
                        : But it can be overwritten so that the legend is drawn even if one of the subplot is empty


    Example : Dimension = [2,2]
              the grah numbers are 1 2
                                   3 4

              Number = 3 corresponds to subplot [1,0]

            : To plot on a graph with 2 line and 3 columns on the graph in the center
            subplot = {"dimension":[3,3],"number":5}
    """

    # default figure size
    if figsize is None:
        # default figsize for simple plot
        if subplot is None:
            figsize = [10, 10]
        else:
            figsize = [14, 10]

    # Sets the dimensions and default values depending on if the subplot is 1D or 2D
    if subplot is None and not add_graph:

        fig, ax = plt.subplots(figsize=figsize)

    elif subplot is None:
        # Gets the current figure and axes if the plot wasn't created on this iteration
        fig = plt.gcf()

    # Si l'argument subplot a été déclaré le subplot est en 2D
    elif subplot is not None:
        # Number of the subplot to graph
        subplot_number = subplot["number"]

        # Dimensions du subplot et tailles de figures
        if "dimension" in subplot:
            dimension_x = subplot["dimension"][0]
            dimension_y = subplot["dimension"][1]

        # If it's the first subplot then it initializes the figure
        if subplot_number == 1 and not add_graph:

            # Creates the subplot with the selected dimensions
            # Sets the figure size
            fig, ax = plt.subplots(dimension_x, dimension_y, figsize=figsize)
            axes = fig.axes
        else:
            # Gets the current figure and axes if the plot wasn't created on this iteration
            fig = plt.gcf()

        # Selects the current subplot
        plt.subplot(dimension_x, dimension_y, subplot_number)

    # Clears the current subplot if add_graph is False
    if add_graph is False:
        plt.cla()

    return fig


def define_simulation_label(labels):
    """
    Defines the labels in a legend
    Put the name of the Simulation case or the simulation is in simulation_description list if you want a more detailed legend than only their name

    simulation_description must be a global variable declared at the begining of a script.

    SimulationDescription = ["Case or Simulation Name","Legend text","Case or Simulation Name","Legend text"]

    Uses the function : "get_simulation_description" to change the case name to it's description'
    """

    case_labels = []
    # Parcours les label du graphique
    for label in labels:
        # Si ce label est dans la liste Simulation description, remplace ce label par sa description
        if label in simulation_description:
            case_labels.append(get_simulation_description(label))
        # Si ce label n'a pas de description, garde ce label
        else:
            case_labels.append(label)

    return case_labels


def clear_legend_duplicates(lines, labels):
    """
    Searches the list of all labels in a figure and deletes duplicates in the label list and in the lines list
    """

    labels_no_duplicates = []
    lines_no_duplicates = []

    # Goes through the labels list from the end to the beginning
    for label_index, label in enumerate(labels):

        # if the label is not already added in the list without duplicates adds it
        if label not in labels_no_duplicates:
            labels_no_duplicates.append(label)
            lines_no_duplicates.append(lines[label_index])

    return lines_no_duplicates, labels_no_duplicates


def legend_setup(fig, graph_type, legend_position='lower center', graph_annotation_on=False, **kwargs):
    """
    Setups the legend of a figure, depending on if a subplot is activated or not
    If a subplot is activated, puts only one legend with every labels on it
    Sets the color of the lines depending on the name of the case

    Sets the legend with multiple columns depending on the number of labels

    legend_position = str, controls where the legend is drawn outside the figure

                   location string of matplotlib 'upper right', 'center left'...
                   default = 'lower center'

                   Default value : lower center (below the figure)
                   WARNING : LOCATIONS IMPLEMENTED : lower center and center left, add more locations by adding an elif statement
    """
    # list des localisation implémentées
    location_list = ["lower center", "center left"]

    # get the axes
    ax = fig.axes

    # locations implemented
    if legend_position == 'lower center':
        # Location of the origin point of the legend box
        Anchor_loc = 'upper center'

        # x coordinate of the legend in the figure (Loc_x = 0 means on the left, Loc_x = 1 means on the right, Loc_x = 0.5 means in the middle)
        Loc_x = 0.5

        # y coordinate of the legend in the figure (Loc_y = 0 means on the bottom, Loc_y = 1 means on the top, Loc_x = 0.5 means in the middle)
        Loc_y = 0

        # Maximum number of labels per column in the legend
        LabelsPerColumn = 5

    elif legend_position == 'center left':
        # Location of the origin point of the legend box
        Anchor_loc = 'center right'

        # x coordinate of the legend in the figure (Loc_x = 0 means on the left, Loc_x = 1 means on the right, Loc_x = 0.5 means in the middle)
        Loc_x = 0

        # y coordinate of the legend in the figure (Loc_y = 0 means on the bottom, Loc_y = 1 means on the top, Loc_x = 0.5 means in the middle)
        Loc_y = 0.5

        # Maximum number of labels per column in the legend
        LabelsPerColumn = 100
    else:
        raise ValueError(
            f"La localisation legend_position={legend_position} n'est pas implémentée dans la fonction graph.legend_setup. \nLes localisations implémentées sont :\n{location_list}")
        return

    # If there is a graph annotation, place a texbox under the legend that says what the annotation correspond to
    if graph_annotation_on:

        import matplotlib.text

        # Informations for the legend for the annotation

        # Annotation angle settings
        if graph_type == "COP_graph":
            # Variable
            annotation_variable = kwargs.get(
                "annotation_variable", "Abduction")
        else:
            # Variable to draw in the peak annotation is the x variable
            variable_x = kwargs.get("variable_x")
            annotation_variable = kwargs.get("annotation_variable", variable_x)

        annotation_composante = kwargs.get("annotation_composante", "Total")
        annotation_mode = kwargs.get("annotation_mode", "max")

        legend_string = f" : {annotation_mode.capitalize()} {annotation_variable}"

        if not annotation_composante == "Total":
            legend_string += f" in the {annotation_composante} direction"

        # Annotation box
        props = dict(boxstyle='round', facecolor='grey', alpha=0.6)

    # if a legend already exists in the figure, deletes it to update it
    if fig.legends:
        fig.legends.clear()

    # maximumNumber of labels per columns in the legend

    # if the subplot dimensions are 1x1 (the ax list containing only 1 axis)
    if len(ax) == 1:

        # gets the axis object
        axe = ax[0]

        # Gets all labels of the figure
        lines, labels = axe.get_legend_handles_labels()

    else:

        # Collects all the labels and lines of the figure
        lines_labels = [axe.get_legend_handles_labels() for axe in ax]
        # reshaping so that the lists are two 1D lists
        lines, labels = [sum(i, []) for i in zip(*lines_labels)]

        # removes duplicates labels
        lines, labels = clear_legend_duplicates(lines, labels)

    # Only draws the legend if there are multiple labels in the figure
    if len(labels) > 1:

        # Number of columns in the legend to not exceed the max number of labels per column
        ncol = int(np.ceil((len(labels)) / LabelsPerColumn))

        # Changes the names of the case to their description
        Simulationlabels = define_simulation_label(labels)

        # Places one legend for the whole subplot
        leg = fig.legend(lines, Simulationlabels, bbox_to_anchor=(
            Loc_x, Loc_y), loc=Anchor_loc, ncol=ncol)

        if graph_annotation_on:
            # Creates the legend of the annotation boxes under the legend of the graoh
            # Create offset from lower right corner of legend box,
            # (0.0,0.0) is the coordinates of the offset point in the legend coordinate system
            # Then places a box under the legend
            offset_legend = matplotlib.text.OffsetFrom(leg, (0.0, 0))

            # Create annotation. Top right corner located -20 pixels below the offset point
            # (lower right corner of legend).
            graph_annotation_legend_box = plt.annotate("XX", xy=(0, 0),
                                                       xycoords='figure fraction', xytext=(5, -20), textcoords=offset_legend,
                                                       horizontalalignment='left', verticalalignment='bottom',
                                                       bbox=props)

            offset_legend_box = matplotlib.text.OffsetFrom(
                graph_annotation_legend_box, (1.5, 0))

            plt.annotate(legend_string, xy=(0, 0),
                         xycoords='figure fraction', xytext=(0, 0), textcoords=offset_legend_box,
                         horizontalalignment='left', verticalalignment='bottom')

    # If no legend was created and there are graph annotations on the graph
    elif len(labels) == 1 and graph_annotation_on:

        graph_annotation_legend_box = plt.annotate("XX", xy=(0.5, 0),
                                                   xycoords='figure fraction',
                                                   horizontalalignment='left', verticalalignment='bottom',
                                                   bbox=props)

        offset_legend_box = matplotlib.text.OffsetFrom(
            graph_annotation_legend_box, (1.5, 0))

        plt.annotate(legend_string, xy=(0, 0),
                     xycoords='figure fraction', xytext=(0, 0), textcoords=offset_legend_box,
                     horizontalalignment='left', verticalalignment='bottom')


def graph_grid_setup(fig, last_subplot=False, xlim=None, ylim=None, grid_x_step=None, grid_y_step=None, same_lim=False, **kwargs):
    """
    can overwrite the graph limits (x and y)

    subplot = the subplot dictionary

    last_subplot : bool True if this the last subplot drawn

    xlim : list : sets the min and max limit of the x axis [xmin, xmax]
    ylim : list : sets the min and max limit of the y axis [ymin, ymax]

    grid_x_step : list : sets the distance between each x axis ticks and grid line
    grid_y_step : list : sets the distance between each y axis ticks and grid line

    same_lim : bool, if True
                        all subplots have the same limit --> lim entered apply to every graph
                        or if not entered, collects the max and minimum of each limits and draws them
                        the function will be updated only if last_subplot is True (all subplots have been plot)

    """

    # sub-modules used to set axes ticks
    def set_xticks(axe, graph_xlim, grid_x_step):
        """
        function to set the xticks according to the grid_x_step argument
        and rounds the ticks to the nearest number dividable by the entered grid step
        """
        # rounds the lim to the nearest number dividable by the entered grid step
        min_lim = math.ceil(graph_xlim[0] / grid_x_step) * grid_x_step
        max_lim = math.ceil(graph_xlim[1] / grid_x_step) * grid_x_step

        axe.set_xticks(np.arange(min_lim, max_lim + grid_x_step, grid_x_step))

    def set_yticks(axe, graph_ylim, grid_y_step):
        """
        function to set the yticks according to the grid_x_step argument
        and rounds the ticks to the nearest number dividable by the entered grid step
        """
        # rounds the lim to the nearest number dividable by the entered grid step
        min_lim = math.ceil(graph_ylim[0] / grid_y_step) * grid_y_step
        max_lim = math.ceil(graph_ylim[1] / grid_y_step) * grid_y_step

        axe.set_yticks(np.arange(min_lim, max_lim + grid_y_step, grid_y_step))

    # only executes this function if at least one of these arguments are entered
    if not any([xlim, ylim, grid_x_step, grid_y_step, same_lim]):
        plt.grid(visible=True)
        return

    # get the axis of the subplot
    ax = fig.axes

    # if the subplot is in 2d (more than one axis)
    if len(ax) > 1:

        # if the limits are set to be the same for all subplots
        if same_lim:
            # Only does this mode when all the graphs are done
            if last_subplot:

                # Collects all the axis limits and calculates the extremes
                # if one of the limits need to be set automatically
                if xlim is None or ylim is None:

                    min_xlim = []
                    max_xlim = []
                    min_ylim = []
                    max_ylim = []

                    for axe in ax:

                        graph_xlim = axe.get_xlim()
                        graph_ylim = axe.get_ylim()

                        min_xlim.append(graph_xlim[0])
                        max_xlim.append(graph_xlim[1])
                        min_ylim.append(graph_ylim[0])
                        max_ylim.append(graph_ylim[1])

                # sets the x axis new limits
                if xlim:
                    graph_xlim = xlim
                # sets the graph_xlim as the most extremes limits across all the subplots
                else:
                    graph_xlim = [min(min_xlim), max(max_xlim)]

                # Sets the y axis new limits
                if ylim:
                    graph_ylim = ylim
                # sets the graph_ylim as the most extremes limits across all the subplots
                else:
                    graph_ylim = [min(min_ylim), max(max_ylim)]

                # collects all the graph limits
                for axe in ax:

                    axe.grid(visible=True)
                    axe.set_xlim(graph_xlim[0], graph_xlim[1])
                    axe.set_ylim(graph_ylim[0], graph_ylim[1])

                    # Sets the grid steps if needed
                    if grid_x_step:
                        set_xticks(axe, graph_xlim, grid_x_step)

                    if grid_y_step:
                        set_yticks(axe, graph_ylim, grid_y_step)

        # if the limits and step grid are set individually (same_lim == False)
        else:
            # if the limit is set only for the active axis
            axe = plt.gca()
            plt.grid(visible=True)

            # sets the x axis new limits
            if xlim:
                axe.set_xlim(xlim[0], xlim[1])

            # Sets the y axis new limits
            if ylim:
                axe.set_ylim(ylim[0], ylim[1])

            if grid_x_step:
                graph_xlim = axe.get_xlim()

                set_xticks(axe, graph_xlim, grid_x_step)

            if grid_y_step:
                graph_ylim = axe.get_ylim()

                set_yticks(axe, graph_ylim, grid_y_step)

    # if there is no subplot, the axis is one dimensional
    else:
        # gets the axis object
        axe = ax[0]

        plt.grid(visible=True)

        # sets the x axis new limits
        if xlim:
            axe.set_xlim(xlim[0], xlim[1])

        # Sets the y axis new limits
        if ylim:
            axe.set_ylim(ylim[0], ylim[1])

        if grid_x_step:
            graph_xlim = axe.get_xlim()
            set_xticks(axe, graph_xlim, grid_x_step)

        if grid_y_step:
            graph_ylim = axe.get_ylim()

            set_yticks(axe, graph_ylim, grid_y_step)


def get_simulation_description(label):
    """
    Transforms the case name into it's description (From the simulation_description global list)
    SimulationDescriptionDictionary must be a global list declared at the beginning of the code :
        global SimulationDescriptionDictionary
        SimulationDescriptionDictionary = {"Simulation1 Name": "Simulation1 description",
                                           "Simulation2 Name": "Simulation2 description"...}
    """

    case_description = simulation_description[label]

    return case_description


def get_simulation_line_style(label):
    """
    Defines the style of the line data in a graph depending on it's name
    SimulationsLineStyle must have be set with the define_simulations_line_style function


    These line style are defined in a global dictionnary :
        global SimulationsLineStyle
        SimulationsLineStyle = {"Case 1": {"color": "COLORNAME", "marker": "MARKERNAME", "markersize": NUMBER, "linestyle": "LINESTYLE", "linewidth": NUMBER},
                                "Case 2": {"color": "COLORNAME", "marker": "MARKERNAME", "markersize": NUMBER, "linestyle": "LINESTYLE", "linewidth": NUMBER}
                                }
        If one of the settings isn't declared, it will be set to the default value (None)

    """

    simulation_style_dictionary = {}

    # Only select a custom color if there is a label
    if label is not None:
        # Selects the color from the colormap if the graph_label is in SimulationColors
        if label in SimulationsLineStyle:

            simulation_style_dictionary = SimulationsLineStyle[label]

    return simulation_style_dictionary


# %% Graph visual functions setup


def define_COP_contour(COP_contourInputFileName, InputFileType):
    """
    Function that can be used to read a file containing the coordinates of the contour that is plot by the function COP_graph (COP_contour argument)
    It creates a numpy array
    Only the x and y coordinates are used to draw the contour

    Reads the file that contains the coordinates of the contour
    COP_contourInputFileName : string containing the name of the file that contains the coordinates (without the extension)

    InputFileType : string containing the file type (pp...)
                    FOR NOW CAN ONLY READ PICKED POINTS FILES ARE SUPPORTED (.pp)

    ---------------------------------
    return
    COP_contour : Array containing the coordinates of the points of the contour
                Dimension (npoints,3 or 2)
                Column 1 : x
                Column 2 : y
                Column 3 : z (not used by the COP_graph function)

    """
    # import numpy as np
    # Liste des formats supportés (ajouter des formats si cette fonction est modifiée)
    supported_formats = ["pp"]

    # Message d'erreur si le format utilisé n'est pas supporté
    if InputFileType not in supported_formats:
        raise ValueError(
            f"Les fichiers .{InputFileType} ne sont pas supportés par la fonction define_COP_contour \n Les formats supportés sont : {supported_formats}")
        return

    COP_contour = read_picked_points(
        COP_contourInputFileName + "." + InputFileType)

    return COP_contour


def draw_graph_annotation(annotation_values, x, y, **kwargs):
    """
    Draws the peak value of a dataset in an annotation

    x = Liste des coordonnées en x
    y = Liste des coordonnées en y

    annotation_values : Liste contenant les coordonnées à afficher dans le pic

    dans les kwargs :
    annotation_mode = string
    choisit quel type d'annotation de pic est activé
        annotation on the maximum : "max"
        annotation on the minimum : "min"
        annotation on the highest peak : "max_peak"
        annotation on the lowest peak : "min_peak"




        Par Défaut : "max"

    Par défaut : Max activé et les autres désactivés

    n_interpolation_points : int, Default = None
    to be used only if the x values are striclty increasing
    None : no interpolation done, keeps the original data

    controls the precision of the value displayed in the annotation by setting the number of interpolation points
    the interpolation function will be calculated with n_interpolation_points and the x, y and annotation_values will have this amount of point
    finds an interpolate function f : y = f(x) and then use it to display the peak value

    ex : if y = annotation_values = [0, 100, 25, 50, 75]
        x = [0, 25, 50, 75, 100]
    only 5 points so the value calculated won't be very precise (the x coordinate will have a 25 precision)
    with n_interpolation_points = 100, the values calculated on x will have a precision of 1



    """

    """
    SCRIPT TO INTERPOLATE NOT TESTED
    """

    # Sets the number of interpolation points to use to calculate the peak
    n_interpolation_points = kwargs.get("n_interpolation_points", None)

    annotation_mode = kwargs.get("annotation_mode", "max")

    implemented_annotation_modes = [
        "min", "max", "min_peak", "max_peak", "first", "last"]

    find_max_index = False
    find_min_index = False
    find_max_peak_index = False
    find_min_peak_index = False
    find_first_point = False
    find_last_point = False

    # Choses which peak annotation to draw
    if annotation_mode == "max":
        find_max_index = True
    elif annotation_mode == "min":
        find_min_index = True
    elif annotation_mode == "max_peak":
        find_max_peak_index = True
    elif annotation_mode == "min_peak":
        find_min_peak_index = True
    elif annotation_mode == "first":
        find_first_point = True
        find_min_peak_index = True
    elif annotation_mode == "last":
        find_last_point = True
    else:
        raise ValueError(
            f"annotation_mode must be one of these modes : {implemented_annotation_modes}")

    # No interpolation is done
    if n_interpolation_points is None:
        interpolated_values = annotation_values
        interpolated_x = x
        interpolated_y = y

    else:
        from scipy import interpolate

        # Creates a list of numbers from the first to the last value of x with a number of points of n_interpolation_points
        interpolated_x = np.linspace(x[0], x[-1], n_interpolation_points)

        try:
            # interpolation function to find the annotation values
            values_interpolation_function = interpolate.CubicSpline(
                x, annotation_values)
        # Error message in case x isn't strictly increasing
        except ValueError:
            raise ValueError(
                "La variable en x doit augmenter strictement pour augmenter la précision du calcul du pic par interpolation")

        interpolated_values = values_interpolation_function(interpolated_x)

        # interpolation function to find the annotation values
        y_interpolation_function = interpolate.CubicSpline(x, y)
        interpolated_y = y_interpolation_function(interpolated_x)

    # When we need to find peaks
    if find_first_point is False and find_last_point is False:
        # Trouve les indices où y a atteint un pic (maximums et minimums)
        peak_indexes = find_peak_indexes(
            interpolated_y, find_max_index, find_min_index, find_max_peak_index, find_min_peak_index)
    # Finds the first point index
    elif find_first_point:
        peak_indexes = [0]
    # Finds the last point index
    elif find_last_point:
        peak_indexes = [len(interpolated_y) - 1]

    # Gets the x and y position of the peaks and the peak value
    Peak_x = interpolated_x[peak_indexes]
    Peak_y = interpolated_y[peak_indexes]
    Peak_Value = interpolated_values[peak_indexes]

    # Gets the color of the last ploted graph
    color = plt.gca().lines[-1].get_color()

    for index in range(len(peak_indexes)):

        plt.annotate(f"{round(Peak_Value[index])}",
                     xy=(Peak_x[index], Peak_y[index]),
                     bbox=dict(boxstyle="round", fc=color, alpha=0.6),
                     arrowprops=dict(
                         arrowstyle="-", connectionstyle="arc3", color="black"),
                     color="black")


def draw_COP_points(coordinates, x, y, points_step):
    """
    Shows where is the COP at t=0s
    And draws a point where the COP is during the movement

    coordinates : Liste ou array contenant l'angle de déplacement angulaire étudié

    points_step spécifie tous les combien de degré d'abduction un point doit être tracé
        Type = int


    Exemple : points_step = 15 : Le COP sera tracé tous les 15° [15, 30, 45, 60....] environ en trouvant les données correspondant aux angles les plus proches
    Le script détect l'angle de début et de fin

    """

    CaseColor = plt.gca().lines[-1].get_color()

    def find_closest_number(Array, Number):
        """
        Fonction pour trouver la position et la valeur de l'élément ayant la valeur la plus proche d'un nombre spécifié dans une liste de nombres

        Exemple : Array = [1, 2, 3, 4]
        Number = 3.2

        ClosestNumber = 3
        NumberIndex = 2

        """
        ClosestNumber = Array[min(
            range(len(Array)), key=lambda i: abs(Array[i] - Number))]
        NumberIndex = np.where(Array == ClosestNumber)[0][0]

        return ClosestNumber, NumberIndex

    # Initialisation des angles
    ClosestAngles = []
    Indexes = []

    # Parcours les valeurs d'angles de la valeur minimale à maximale, avec un pas spécifié
    # Ne sélectionne pas la valeur minimale ni la valeur maximale
    for Angle in range(int(min(coordinates)) + points_step, int(max(coordinates)), points_step):

        # Trouve la valeur dans la liste d'angle qui est la plus proche de l'angle voulu et sa position dans la liste
        ClosestNumber, NumberIndex = find_closest_number(coordinates, Angle)

        ClosestAngles = [*ClosestAngles, ClosestNumber]
        Indexes = [*Indexes, NumberIndex]

    # Sélectionne les valeurs x et y du COP à tracer
    xSelection = x[Indexes]
    ySelection = y[Indexes]

    plt.plot(x[0], y[0], '+', color=CaseColor, markersize=10, mew=2.5)
    plt.plot(xSelection, ySelection, ".", color=CaseColor, markersize=8)

# %% Graph functions


def graph(data, variable_x, variable_y, figure_title="", cases_on=False, compare=False, composante_x="Total", composante_y=["Total"], subplot=None, subplot_title=False, compared_case="", **kwargs):
    """
    Fonction générale qui gère les graphiques


    data : le dictionnaire contenant les data à tracer
         : Par défaut : Un dictionnaire ne contenant qu'une seule simulation
         : Soit un jeu de plusieurs datas (compare = True)

    variable_x : Le nom de la variable placée en x sur le graphique
    variable_y : le nom de la variable placée en y sur le graphique

    composante_y :
                  : type : liste de chaines de charactère
                  : Liste contenant les nom des composantes de la variable à tracer
                  : Par défaut : On trace la composante "Total" donc composante_y = ["Total"]

                : Activer plusieurs composantes :
                Exemple : composante_y = ["composante 1","composante 2","composante 3","Total"....]
                          Si on veut activer x et y entrer : composante_y = ["x","y"]

                : Activer une seule composante :
                Exemple : Si on veut activer y entrer : composante_y = ["y"]


                CAS PARTICULIER COMPOSANTES: Si on compare, on ne peut activer qu'une seule composante
                                           : Si on active plusieurs composantes, on doit comparer la même donnée (un seul cas de simulation)

    Composantes_x : Le nom de la composante de la variable en abscisse
                  : composante_x est une chaîne de charactère contenant le nom de la composante de la variable
                  : Par défaut : "Total"
                  : Si on veut activer y entrer : Composantes_x = "y"

    compare : = True si on veut comparer plusieurs données
              Ne rien mettre (compare = False par défaut) : on veut tracer qu'une seule donnée

    subplot = {"Dimension: [nrows, ncolumns]", :"number": Number_of_the_subplot_selected, "figsize": [horizontal_size_inches, vertical_size_inches], "last_subplot": True}


        subplot["dimension"] = [nrows, ncolumns]
    And defines the active axis as the subplot["number"]=number of the plot


    subplot["figsize"] : Optional argument to set the size of the figure
                         subplot["figsize"] = [horizontal_size_inches, vertical_size_inches]
                         : default : [14, 10] inches for 2D ; [7, 5] for [1,1] subplot

    subplot["dimension"] and figsize : are only to be set for subplot["number"] = 1
    They are not taken in account otherwise

    subplot["LastPart"] : bool = Optional argument : Controls if the legend and figure title are drawn
                        : It's automatically set to True if we reach the maximum subplot Number
                        : But it can be overwritten so that the legend is drawn even if one of the subplot is empty


    Example : Dimension = [2,2]
              the grah numbers are 1 2
                                   3 4

              Number = 3 corresponds to subplot [1,0]

            : To plot on a graph with 2 line and 3 columns on the graph in the center
            subplot = {"dimension":[3,3],"number":5}

    **kwargs : contient d'autres paramètres comme
             label : si jamais on veut ajouter un label à une donnée d'un graphique qui n'en aurait ou qui en aurait un autre
             add_graph = True : Si jamais on veut ajouter un autre graphique sur le dernier graphique tracé
                               : False par défaut, les nouvelles données seront tracées en effaçant les anciennes sur le subplot en cours
             legend_on : bool : argument contrôlant l'affichage de la légende
                       : True (par défaut) la légende s'affiche
                       : False La légende ne s'affiche pas'
             LegendLocation = dictionary, controls where the legend is drawn outside the figure

                           location string of matplotlib 'upper right', 'center left'...

                           Default value : lower center (below the figure)
    """

    # get the customlabel if a label arguent is declared, puts None otherwise as a default value
    custom_label = kwargs.get("label", None)

    # Get the add_graph variable. Puts it to false by default if it's not declared in the kwargs
    add_graph = kwargs.get("add_graph", False)

    # get the legend_on argument that controls if the legend is drawn or not (Default True)
    legend_on = kwargs.get("legend_on", True)

    # Stores the name of the x variable and y variable in the kwargs
    kwargs["variable_y"] = variable_y
    kwargs["variable_x"] = variable_x

    graph_annotation_on = kwargs.get("graph_annotation_on", False)

    graph_type = "graph"

    # Verifications for when simulationCases are used
    if cases_on:
        # If "all", all cases are selected to be drawn
        if cases_on == "all":
            cases_on = list(data.keys())

        elif type(cases_on) is str:
            raise ValueError(
                "cases_on doit être une liste si 'all' n'est pas utilisé")
            return

        # Vérifie qu'on n'active pas plusieurs cas tout en comparant
        if len(cases_on) > 1 and compare:
            raise ValueError(
                "On ne peut pas comparer plusieurs simulations et plusieurs cas en même temps")
            return

        # Vérifie qu'on ne dessine pas plusieurs variables tout en dessinant plusieurs cas
        if len(cases_on) > 1 and len(composante_y) > 1:
            raise ValueError(
                "On ne peut pas dessiner plusieurs cas et plusieurs composantes en même temps")
            return

    # Vérification qu'on ne dessine pas plusieurs variables tout en comparant
    if compare and len(composante_y) > 1:
        raise ValueError(
            "On ne peut pas comparer plusieurs simulations et dessiner plusieurs variables")
        return

    # flatten the data into a simulation cases form if compare=True
    if compare:
        # stores the case name that is being compared
        compared_case = cases_on[0]

        # resets the cases_on names to all the simulations names
        data, cases_on = flatten_compared_simulations(data, cases_on[0])
    else:
        compared_case = ""

    # Gets the figure size
    figsize = kwargs.get("figsize", None)

    fig = subplot_setup(subplot, figsize, add_graph)

    # Selects the data to graph
    x_description, y_description = graph_select_data_to_plot(data, composante_x, composante_y, cases_on, compare, custom_label, compared_case, graph_type, **kwargs)

    # # S'il n'y a qu'une composante à tracer
    # if len(composante_y) == 1:

    #     # Prend la valeur de la composante comme elle est seule
    #     composante_y = composante_y[0]

    #     # if compare is False:
    #     # était dans compare is false (avec ajout tabluation)
    #     if cases_on is False:
    #         label = None
    #         plot_graph_functions(data, data[variable_x][composante_x],
    #                    data[variable_y][composante_y], graph_type, label=label, custom_label=custom_label, **kwargs)

    #     # If the graph used is CasesGraph
    #     else:
    #         for Case in cases_on:
    #             label = Case

    #             plot_graph_functions(data[Case], data[Case][variable_x][composante_x], data[Case]
    #                        [variable_y][composante_y], graph_type, label=label, custom_label=custom_label, **kwargs)

    #     # elif compare:

    #     #     ListSimulations = list(data.keys())

    #     #     for Simulation in ListSimulations:
    #     #         # Definds the color of this simulation depending on its name
    #     #         label = Simulation

    #     #         if cases_on is False:
    #     #             plot_graph_functions(data[Simulation], data[Simulation][variable_x][composante_x], data[Simulation]
    #     #                        [variable_y][composante_y], graph_type, label=label, custom_label=custom_label, **kwargs)

    #     #         # When we compare, we compare only one case between several simulations
    #     #         elif len(cases_on) == 1:
    #     #             plot_graph_functions(data[Simulation][cases_on[0]], data[Simulation][cases_on[0]][variable_x][composante_x],
    #     #                        data[Simulation][cases_on[0]][variable_y][composante_y], graph_type, label=label, custom_label=custom_label, **kwargs)

    # # Si plusieurs composantes sont activées
    # else:

    #    # # On ne peut comparer que si on active la même donnée, donc seulement une seule composante
    #     # if compare is False:
    #     for Composante in composante_y:
    #         label = Composante

    #         if cases_on is False:
    #             plot_graph_functions(data, data[variable_x][composante_x], data[variable_y]
    #                        [Composante], graph_type, label=label, custom_label=custom_label, **kwargs)

    #         # On peut tracer plusieurs composantes seulement si un seul cas de simulation est activé
    #         elif len(cases_on) == 1:
    #             plot_graph_functions(data, data[cases_on[0]][variable_x][composante_x], data[cases_on[0]]
    #                        [variable_y][Composante], graph_type, label=label, custom_label=custom_label, **kwargs)

    # # # Axis Labels from the variable description
    # # if compare:
    # #     if cases_on is False:
    # #         plt.xlabel(data[ListSimulations[0]][variable_x]["Description"])
    # #         plt.ylabel(data[ListSimulations[0]][variable_y]["Description"])
    # #     else:
    # #         plt.xlabel(data[ListSimulations[0]][cases_on[0]]
    # #                    [variable_x]["Description"])
    # #         plt.ylabel(data[ListSimulations[0]][cases_on[0]]
    # #                    [variable_y]["Description"])

    # # elif compare is False:

    # # était dans le compare is false
    # if cases_on is False:
    #     plt.xlabel(data[variable_x]["Description"])
    #     plt.ylabel(data[variable_y]["Description"])
    # else:
    #     plt.xlabel(data[cases_on[0]][variable_x]["Description"])
    #     plt.ylabel(data[cases_on[0]][variable_y]["Description"])

    if subplot is None:
        plt.title(figure_title)

        # unsuperpose the annotations if activated
        if graph_annotation_on:

            # Calls the function that will move the annotations to avoid superposition
            unsuperpose_plot_annotations(**kwargs)

        # shows the legend if activated
        if legend_on:
            legend_setup(fig, graph_type, **kwargs)

        # Traces the axis labels
        plt.xlabel(x_description)
        plt.ylabel(y_description)

        # Setups the grid and the axes ticks of the graph
        graph_grid_setup(fig, **kwargs)

    # Dans le cas d'un subplot
    else:

        # If a subplot title is entered, draws it (subplot_title isn't a bool)
        if subplot_title:
            plt.title(subplot_title)

        # unsuperpose the annotations if activated
        if graph_annotation_on:

            # Calls the function that will move the annotations to avoid superposition
            unsuperpose_plot_annotations(**kwargs)

        # last_subplot can be entered in the subplot dictionary to oblige the legend to draw even if a subplot is empty
        # This statement has the priority over the test on the number of dimension
        if "last_subplot" in subplot:
            last_subplot = subplot["last_subplot"]

        # Tests if the number of subplot corresponds to the last subplot number to control if the legend and title are drawn or not
        elif subplot["number"] == subplot["dimension"][0] * subplot["dimension"][1]:
            last_subplot = True
        # Case where no legend and figure title will be drawn
        else:
            last_subplot = False

        # Setups the grid and the axes ticks of the graph
        graph_grid_setup(fig, last_subplot, **kwargs)

        # Displays the legend and figure title only if it's the last subplot drawn
        if last_subplot:

            # Trace le titre de la figure
            plt.suptitle(figure_title)

            # Traces the axis labels
            plt.xlabel(x_description)
            plt.ylabel(y_description)

            # Ajuste les distances entre les subplots quand ils sont tous tracés
            plt.tight_layout()

            # shows the legend if activated
            if legend_on:
                legend_setup(fig, graph_type, **kwargs)


def muscle_part_graph(data, MuscleName, MusclePart, variable_x, variable_y, figure_title="", composante_x="Total", composante_y=["Total"], compare=False, subplot=None, subplot_title=False, cases_on=False, MusclePartInformation=False, fig=None, compared_case="", **kwargs):
    """
    Fonction qui gère trace la variable d'une seule fibre musculaire

    lastPart = statement pour dire qu'on dessine la dernière musclepart pour ne tracer la légende qu'à ce moment là


    data : le dictionnaire contenant les data à tracer
         : Par défaut : Un dictionnaire ne contenant qu'une seule simulation
         : Soit un jeu de plusieurs datas (compare = True)

    variable_x : Le nom de la variable placée en x sur le graphique
    variable_y : le nom de la variable placée en y sur le graphique

    composante_y :
                  : type : liste de chaines de charactère
                  : Liste contenant les nom des composantes de la variable à tracer
                  : Par défaut : On trace la composante "Total" donc composante_y = ["Total"]

                : Activer plusieurs composantes :
                Exemple : composante_y = ["composante 1","composante 2","composante 3","Total"....]
                          Si on veut activer x et y entrer : composante_y = ["x","y"]

                : Activer une seule composante :
                Exemple : Si on veut activer y entrer : composante_y = ["y"]

                CAS PARTICULIER COMPOSANTES: Si on compare, on ne peut activer qu'une seule composante
                                           : Si on active plusieurs composantes, on doit comparer la même donnée (un seul cas de simulation)

    Composantes_x : Le nom de la composante de la variable en abscisse
                  : composante_x est une chaîne de charactère contenant le nom de la composante de la variable
                  : Par défaut : "Total"
                  : Si on veut activer y entrer : Composantes_x = "y"

    muscle_part_on  : Liste contenant les numéros des parties à tracer
                  : active ou non de graph la variable totale du muscle ou la variable d'une des parties du muscle
                  : "allparts" toutes les parties on sans le total
                  : "all" toutes les parties avec le total

                  : Défault = False : trace la variable totale du muscle entier
                  : muscle_part_on = liste des numéros de la partie du muscle à tracer


    compare : = True si on veut comparer plusieurs données
              Ne rien mettre (compare = False par défaut) : on veut tracer qu'une seule donnée

    subplot = {"Dimension: [nrows, ncolumns]", :"number": Number_of_the_subplot_selected, "figsize": [horizontal_size_inches, vertical_size_inches], "last_subplot": True}


        subplot["dimension"] = [nrows, ncolumns]
    And defines the active axis as the subplot["number"]=number of the plot


    subplot["figsize"] : Optional argument to set the size of the figure
                         subplot["figsize"] = [horizontal_size_inches, vertical_size_inches]
                         : default : [14, 10] inches for 2D ; [7, 5] for [1,1] subplot

    subplot["dimension"] and figsize : are only to be set for subplot["number"] = 1
    They are not taken in account otherwise

    subplot["LastPart"] : bool = Optional argument : Controls if the legend and figure title are drawn
                        : It's automatically set to True if we reach the maximum subplot Number
                        : But it can be overwritten so that the legend is drawn even if one of the subplot is empty


    Example : Dimension = [2,2]
              the grah numbers are 1 2
                                   3 4

              Number = 3 corresponds to subplot [1,0]

            : To plot on a graph with 2 line and 3 columns on the graph in the center
            subplot = {"dimension":[3,3],"number":5}

    **kwargs : contient d'autres paramètres comme
             label : si jamais on veut ajouter un label à une donnée d'un graphique qui n'en aurait ou qui en aurait un autre
             add_graph = True : Si jamais on veut ajouter un autre graphique sur le dernier graphique tracé
                               : False par défaut, les nouvelles données seront tracées en effaçant les anciennes sur le subplot en cours
             legend_on : bool : argument contrôlant l'affichage de la légende
                       : True (par défaut) la légende s'affiche
                       : False La légende ne s'affiche pas'
            LegendLocation = dictionary, controls where the legend is drawn outside the figure

                          location string of matplotlib 'upper right', 'center left'...

                          Default value : lower center (below the figure
    """

    # get the customlabel if a label arguent is declared, puts None otherwise as a default value
    custom_label = kwargs.get("label", None)

    # get the legend_on argument that controls if the legend is drawn or not (Default True)
    legend_on = kwargs.get("legend_on", True)

    graph_type = "muscle_graph"

    # Stores the name of the x variable and y variable in the kwargs
    kwargs["variable_y"] = variable_y
    kwargs["variable_x"] = variable_x

    graph_annotation_on = kwargs.get("graph_annotation_on", False)

    # Name of the dictionnary key where the muscles are stored
    # By default it's muscles but in case of an edge muscle it is stored in GHReactions
    if "Edge muscle" in MuscleName:
        MuscleFolder = "GHReactions"
    else:
        MuscleFolder = "Muscles"

    # Initialise les informations sur les muscles parts si elle n'a pas été spécifiée (c'est à dire qu'il n'y a qu'une seule musclePart à dessiner)
    if MusclePartInformation is False:
        MusclePartInformation = {"LastPart": True,
                                 "Total Number Muscle Parts": 1}

    # Parcours toutes les parties de muscles à tracer
    # Selects the data to graph
    x_description, y_description = muscle_graph_select_data_to_plot(data, composante_x, composante_y, cases_on, compare, custom_label, compared_case, graph_type, MusclePartInformation, MusclePart, MuscleFolder, MuscleName, **kwargs)

    # # S'il n'y a qu'une composante à tracer
    # if len(composante_y) == 1:

    #     # Prend la valeur de la composante comme elle est seule
    #     composante_y = composante_y[0]

    #     # if compare is False:
    #     # était dans compare is false
    #     if cases_on is False:

    #         # Si plus d'une muscle part est tracée, on met une legende avec le nom de la musclepart
    #         if MusclePartInformation["Total Number Muscle Parts"] > 1:
    #             label = MusclePart

    #         # Si seulement une muscle part est activée et qu'on ne compare pas, on ne met pas de légende
    #         else:
    #             label = None

    #         plot_graph_functions(data, data[variable_x][composante_x], data[MuscleFolder][MuscleName]
    #                              [MusclePart][variable_y][composante_y], graph_type, label=label, custom_label=custom_label, **kwargs)

    #     # with simulation cases
    #     else:
    #         # On ne peut tracer qu'une seule donnée, donc on doit avoir soit un seul Case de sélectionné et n>=1 muscle parts
    #         # Ou on peut avoir plusieurs Case de sélectionnés mais une seule muscle part à tracer
    #         if len(cases_on) == 1 or MusclePartInformation["Total Number Muscle Parts"] == 1:

    #             for Case in cases_on:

    #                 # La légende est le nom du case si il n'y a qu'une seule muscle part à tracer et plus d'un Case sélectionné
    #                 if len(cases_on) > 1 and MusclePartInformation["Total Number Muscle Parts"] == 1:
    #                     label = Case

    #                 # La légende est le nom de la muscle part s'il n'y a qu'un seul case et plusieurs Muscle part à tracer
    #                 elif len(cases_on) == 1 and MusclePartInformation["Total Number Muscle Parts"] > 1:
    #                     label = MusclePart

    #                 # Si les deux sont 1, on ne met pas de légende
    #                 else:
    #                     label = None

    #                 plot_graph_functions(data[Case], data[Case][variable_x][composante_x], data[Case][MuscleFolder][MuscleName]
    #                                      [MusclePart][variable_y][composante_y], graph_type, label=label, custom_label=custom_label, **kwargs)

    #     # elif compare:

    #     #     # Si on a plusieurs simulations, on ne peut afficher qu'une seule donnée sur le graphique, donc qu'une seule muscle part
    #     #     if MusclePartInformation["Total Number Muscle Parts"] == 1:
    #     #         ListSimulations = list(data.keys())

    #     #         for Simulation in ListSimulations:
    #     #             label = Simulation

    #     #             if cases_on is False:
    #     #                 plot_graph_functions(data[Simulation], data[Simulation][variable_x][composante_x], data[Simulation][MuscleFolder]
    #     #                            [MuscleName][MusclePart][variable_y][composante_y], graph_type, label=label, custom_label=custom_label, **kwargs)

    #     #             # When we compare, we compare only one case between several simulations
    #     #             elif len(cases_on) == 1:
    #     #                 plot_graph_functions(data[Simulation][cases_on[0]], data[Simulation][cases_on[0]][variable_x][composante_x], data[Simulation][cases_on[0]]
    #     #                            [MuscleFolder][MuscleName][MusclePart][variable_y][composante_y], graph_type, label=label, custom_label=custom_label, **kwargs)

    # # Si plusieurs composantes sont activées
    # else:

    #     # Si on a plusieurs composantes, on ne peut afficher qu'une seule donnée sur le graphique, donc qu'une seule muscle part
    #     if MusclePartInformation["Total Number Muscle Parts"] == 1:

    #         # On ne peut comparer plusieurs simulations que si on active la même donnée, on ne peut pas afficher plusieurs composantes avec plusieurs simulations
    #         if compare is False:

    #             for Composante in composante_y:
    #                 label = Composante

    #                 if cases_on is False:
    #                     plot_graph_functions(data, data[variable_x][composante_x], data[MuscleFolder][MuscleName]
    #                                          [MusclePart][variable_y][Composante], graph_type, label=label, custom_label=custom_label, **kwargs)

    #                 # On peut tracer plusieurs composantes seulement si un seul cas de simulation est activé
    #                 elif len(cases_on) == 1:
    #                     plot_graph_functions(data, data[cases_on[0]][variable_x][composante_x], data[cases_on[0]][MuscleFolder]
    #                                          [MuscleName][MusclePart][variable_y][Composante], graph_type, label=label, custom_label=custom_label, **kwargs)

    # Si on trace la dernière muscle part, trace les axes, la légende, les titres etc...
    if MusclePartInformation["LastPart"]:

        # Axis Labels from the variable description
        # if compare:
        #     if cases_on is False:
        #         plt.xlabel(data[ListSimulations[0]][variable_x]["Description"])
        #         plt.ylabel(data[ListSimulations[0]][MuscleFolder]
        #                    [MuscleName][MusclePart][variable_y]["Description"])
        #     else:
        #         plt.xlabel(data[ListSimulations[0]][cases_on[0]]
        #                    [variable_x]["Description"])
        #         plt.ylabel(data[ListSimulations[0]][cases_on[0]][MuscleFolder]
        #                    [MuscleName][MusclePart][variable_y]["Description"])
        # elif compare is False:
        # était dans compare is false

        # if cases_on is False:
        #     plt.xlabel(data[variable_x]["Description"])
        #     plt.ylabel(data[MuscleFolder][MuscleName]
        #                [MusclePart][variable_y]["Description"])
        # else:
        #     plt.xlabel(data[cases_on[0]][variable_x]["Description"])
        #     plt.ylabel(data[cases_on[0]][MuscleFolder][MuscleName]
        #                [MusclePart][variable_y]["Description"])

        if subplot is None:
            plt.title(figure_title)

            # unsuperpose the annotations if activated
            if graph_annotation_on:

                # Calls the function that will move the annotations to avoid superposition
                unsuperpose_plot_annotations(**kwargs)

            # shows the legend if activated
            if legend_on:
                legend_setup(fig, graph_type, **kwargs)

            # Traces the axis labels
            plt.xlabel(x_description)
            plt.ylabel(y_description)

            # Setups the grid and the axes ticks of the graph
            graph_grid_setup(fig, **kwargs)

        # Dans le cas d'un subplot
        else:

            # If a subplot title is entered, draws it (subplot_title isn't a bool)
            if not type(subplot_title) is bool:
                plt.title(subplot_title)

            # If a subplot title is entered, draws it (subplot_title isn't a bool)
            if not type(subplot_title) is bool:
                plt.title(subplot_title)

            # unsuperpose the annotations if activated
            if graph_annotation_on:

                # Calls the function that will move the annotations to avoid superposition
                unsuperpose_plot_annotations(**kwargs)

            # last_subplot can be entered in the subplot dictionary to oblige the legend to draw even if a subplot is empty
            # This statement has the priority over the test on the number of dimension
            if "last_subplot" in subplot:
                last_subplot = subplot["last_subplot"]
            # Tests if the number of subplot corresponds to the last subplot number to control if the legend and title are drawn or not
            elif subplot["number"] == subplot["dimension"][0] * subplot["dimension"][1]:
                last_subplot = True
            # Case where no legend and figure title will be drawn
            else:
                last_subplot = False

            # Setups the grid and the axes ticks of the graph
            graph_grid_setup(fig, last_subplot, **kwargs)

            # Displays the legend and figure title only if it's the last subplot drawn
            if last_subplot:
                # Trace le titre de la figure
                plt.suptitle(figure_title)

                # Traces the axis labels
                plt.xlabel(x_description)
                plt.ylabel(y_description)

                # Ajuste les distances entre les subplots quand ils sont tous tracés
                plt.tight_layout()

                # shows the legend if activated
                if legend_on:

                    legend_setup(fig, graph_type, **kwargs)


def muscle_graph(data, MuscleName, variable_x, variable_y, figure_title="", cases_on=False, compare=False, composante_x="Total", composante_y=["Total"], muscle_part_on=False, subplot=None, subplot_title=False, **kwargs):
    """
    Draws all the parts of a Muscle that were selected


    data : le dictionnaire contenant les data à tracer
         : Par défaut : Un dictionnaire ne contenant qu'une seule simulation
         : Soit un jeu de plusieurs datas (compare = True)

    variable_x : Le nom de la variable placée en x sur le graphique
    variable_y : le nom de la variable placée en y sur le graphique

    composante_y :
                  : type : liste de chaines de charactère
                  : Liste contenant les nom des composantes de la variable à tracer
                  : Par défaut : On trace la composante "Total" donc composante_y = ["Total"]

                : Activer plusieurs composantes :
                Exemple : composante_y = ["composante 1","composante 2","composante 3","Total"....]
                          Si on veut activer x et y entrer : composante_y = ["x","y"]

                : Activer une seule composante :
                Exemple : Si on veut activer y entrer : composante_y = ["y"]


                CAS PARTICULIER COMPOSANTES: Si on compare, on ne peut activer qu'une seule composante
                                           : Si on active plusieurs composantes, on doit comparer la même donnée (un seul cas de simulation)

    Composantes_x : Le nom de la composante de la variable en abscisse
                  : composante_x est une chaîne de charactère contenant le nom de la composante de la variable
                  : Par défaut : "Total"
                  : Si on veut activer y entrer : Composantes_x = "y"

    muscle_part_on  : Liste contenant les numéros des parties à tracer
                  : active ou non de graph la variable totale du muscle ou la variable d'une des parties du muscle
                  : "allparts" toutes les parties on sans le total
                  : "all" toutes les parties avec le total

                  : Défault = False : trace la variable totale du muscle entier
                  : muscle_part_on = numéro de la partie du muscle à tracer


    compare : = True si on veut comparer plusieurs données
              Ne rien mettre (compare = False par défaut) : on veut tracer qu'une seule donnée

    subplot = {"Dimension: [nrows, ncolumns]", :"number": Number_of_the_subplot_selected, "figsize": [horizontal_size_inches, vertical_size_inches], "last_subplot": True}


        subplot["dimension"] = [nrows, ncolumns]
    And defines the active axis as the subplot["number"]=number of the plot


    subplot["figsize"] : Optional argument to set the size of the figure
                         subplot["figsize"] = [horizontal_size_inches, vertical_size_inches]
                         : default : [14, 10] inches for 2D ; [7, 5] for [1,1] subplot

    subplot["dimension"] and figsize : are only to be set for subplot["number"] = 1
    They are not taken in account otherwise

    subplot["LastPart"] : bool = Optional argument : Controls if the legend and figure title are drawn
                        : It's automatically set to True if we reach the maximum subplot Number
                        : But it can be overwritten so that the legend is drawn even if one of the subplot is empty


    Example : Dimension = [2,2]
              the grah numbers are 1 2
                                   3 4

              Number = 3 corresponds to subplot [1,0]

            : To plot on a graph with 2 line and 3 columns on the graph in the center
            subplot = {"dimension":[3,3],"number":5}

    **kwargs : contient d'autres paramètres comme
             label : si jamais on veut ajouter un label à une donnée d'un graphique qui n'en aurait ou qui en aurait un autre
             add_graph = True : Si jamais on veut ajouter un autre graphique sur le dernier graphique tracé
                               : False par défaut, les nouvelles données seront tracées en effaçant les anciennes sur le subplot en cours
             legend_on : bool : argument contrôlant l'affichage de la légende
                       : True (par défaut) la légende s'affiche
                       : False La légende ne s'affiche pas'
             LegendLocation = dictionary, controls where the legend is drawn outside the figure

                           location string of matplotlib 'upper right', 'center left'...

                           Default value : lower center (below the figure)

    """

    # Get add_graph function. Puts it to false by default if it's not declared in the kwargs
    add_graph = kwargs.get("add_graph", False)

    # Verifications for when simulationCases are used
    if cases_on:
        # Active tous les cas présents dans data
        if cases_on == "all":
            cases_on = list(data.keys())

        # Vérifie que Cases est toujours une liste si 'all' n'est pas utilisé
        elif not type(cases_on) is list:
            raise ValueError(
                "cases_on doit être une liste si 'all' n'est pas utilisé")
            return

        # Vérifie qu'on n'active pas plusieurs cas tout en comparant
        if len(cases_on) > 1 and compare:
            raise ValueError(
                "On ne peut pas comparer plusieurs simulations et plusieurs cas en même temps")
            return

        # Vérifie qu'on ne dessine pas plusieurs variables tout en dessinant plusieurs cas
        if len(cases_on) > 1 and len(composante_y) > 1:
            raise ValueError(
                "On ne peut pas dessiner plusieurs cas et plusieurs composantes en même temps")
            return

    # flatten the data into a simulation cases form if compare=True
    if compare:
        # stores the case name that is being compared
        compared_case = cases_on[0]

        # resets the cases_on names to all the simulations names
        data, cases_on = flatten_compared_simulations(data, cases_on[0])
    else:
        compared_case = ""

    # Name of the folder where muscles are stored
    MuscleFolder = "Muscles"

    # Construit la liste des parties de muscle à tracer
    # Sans cas de simulation selon le cas (avec/sans des cas, avec/sans comparaison)
    if cases_on is False:
        # if compare is False:
        # était dans compare is false

        try:

            MuscleParts = list(data[MuscleFolder][MuscleName].keys())
        except:
            print(f"{MuscleName} is not a muscle in the the result dictionary")

        # else:
        #     ListSimulations = list(data.keys())
        #     MuscleParts = list(data[ListSimulations[0]][MuscleFolder][MuscleName].keys())

    # Dans les cas où on a des cas de simulation
    else:
        # if compare is False:
        # était dans compare is false

        try:
            MuscleParts = list(data[cases_on[0]][MuscleFolder][MuscleName].keys())
        except:
            print(f"{MuscleName} is not a muscle in the result dictionary")

        # else:
        #     ListSimulations = list(data.keys())
        #     MuscleParts = list(data[ListSimulations[0]][cases_on[0]][MuscleFolder][MuscleName].keys())

    # Si toutes les parties sont activées, fais une liste avec le nom de toutes les parties sauf le muscle total
    # n'enlève pas la partie totale si le muscle n'a pas de partie
    if muscle_part_on == "allparts" and (not len(MuscleParts) == 1):

        # Enlève le muscle total de la liste
        MuscleParts.remove(MuscleName)

    # Dans le cas où on a entré une liste des numéros des parties
    elif isinstance(muscle_part_on, list):

        # Recrée les noms des parties à tracer en parcourant les numéros entrés
        ListMuscleParts = [
            f"{MuscleName} {MusclePartNumber}" for MusclePartNumber in muscle_part_on]

        # Stores the new value of muscleparts to draw
        MuscleParts = ListMuscleParts

    # Si on ne veut tracer qu'un seul muscle
    elif muscle_part_on is False:
        MuscleParts = [MuscleName]

    # Vérification qu'on ne dessine pas plusieurs parties de muscles tout en comparant
    if compare and len(MuscleParts) > 1:
        raise ValueError(
            "On ne peut pas comparer plusieurs simulations et dessiner plusieurs parties de muscles en même temps")
        return

    # Vérification qu'on ne dessine pas plusieurs parties de muscles tout en dessinant plusieurs composantes
    if compare and len(composante_y) > 1:
        raise ValueError(
            "On ne peut pas dessiner plusieurs composantes et dessiner plusieurs parties de muscles en même temps")
        return
    # Vérifie qu'on ne dessine pas plusieurs parties de muscles tout comparant plusieurs cas de simulation
    if type(cases_on) is list:
        if len(cases_on) > 1 and len(MuscleParts) > 1:
            raise ValueError(
                "On ne peut pas dessiner plusieurs cas de simulation et dessiner plusieurs parties de muscles en même temps")
            return

    # Gets the figure size
    figsize = kwargs.get("figsize", None)

    # subplot is setup here to be able to draw every part of a muscle on the same figure
    fig = subplot_setup(subplot, figsize, add_graph)

    # Initialisation du dictionnaire contenant les informations sur le nombre de muscles parts qui seront tracées
    MusclePartInformation = {}

    # Nombre de muscle parts qui seront tracées sur le même graphique
    MusclePartInformation["Total Number Muscle Parts"] = len(MuscleParts)

    # Parcours les parties de muscles à tracer
    for MusclePart in MuscleParts:

        # Si on trace la dernière partie de muscle pour ne tracer la légende et les axes qu'à ce moment là
        if MusclePart == MuscleParts[-1]:
            MusclePartInformation["LastPart"] = True
        else:
            MusclePartInformation["LastPart"] = False

        muscle_part_graph(data, MuscleName, MusclePart, variable_x, variable_y, figure_title, composante_x,
                          composante_y, compare, subplot, subplot_title, cases_on, MusclePartInformation, fig=fig, compared_case=compared_case, **kwargs)


def COP_graph(data, COP_contour=None, variable="COP", figure_title="", composantes=["x", "y"], cases_on=False, subplot=None, compare=False, subplot_title=False, draw_COP_points_on=True, **kwargs):
    """

    Fait le graphique de la position d'un centre de pression et trace un contour (contour d'un implant ou de la surface de contact par exemple)
    data doit avoir une variable appelée "COP"
    COP doit avoir la séquence : "SequenceComposantes": ["AP", "IS", "ML"]
    Trace la composante antéropostérieure (AP) en abscisse (Antérieur = positif) et inférosupérieure (IS) en ordonnée (Supérieur = Positif)

    draw_GH_reactions_nodes= ARGUMENT PERSONNEL POUR TRACER EN PLUS DES POINT SUR LE CONTOUR, PEUT ÊTRE SUPPRIMÉ

    COP_contour : numpy.array contenant les coordonnées x, y du contour à tracer (Peut être créé par la fonction define_COP_contour si on veut lire un fichier contenant ces coordonnées)
               : Dimension (npoints,3 or 2)
                 Column 1 : x
                 Column 2 : y
                 Column 3 : z (not used by the COP_graph function)

    variable : string : The name of the variable to draw
    (Default "COP")

    composantes : list : ["composante_x", "composante_y"]
                : Composantes de la variable à tracer
                (Default ["x", "y"])

    data : le dictionnaire contenant les data à tracer
         : Par défaut : Un dictionnaire ne contenant qu'une seule simulation
         : Soit un jeu de plusieurs datas (compare = True)

    graph_annotation_on : bool : Contrôle l'affichage ou non des angles de pic de COP (Vrai par défaut)

    draw_COP_points_on : bool : active ou non le traçage des points sur le COP

    compare : = True si on veut comparer plusieurs données
              Ne rien mettre (compare = False par défaut) : on veut tracer qu'une seule donnée

    subplot = {"Dimension: [nrows, ncolumns]", :"number": Number_of_the_subplot_selected, "figsize": [horizontal_size_inches, vertical_size_inches], "last_subplot": True}


        subplot["dimension"] = [nrows, ncolumns]
    And defines the active axis as the subplot["number"]=number of the plot


    subplot["figsize"] : Optional argument to set the size of the figure
                         subplot["figsize"] = [horizontal_size_inches, vertical_size_inches]
                         : default : [14, 10] inches for 2D ; [7, 5] for [1,1] subplot

    subplot["dimension"] and figsize : are only to be set for subplot["number"] = 1
    They are not taken in account otherwise

    subplot["LastPart"] : bool = Optional argument : Controls if the legend and figure title are drawn
                        : It's automatically set to True if we reach the maximum subplot Number
                        : But it can be overwritten so that the legend is drawn even if one of the subplot is empty


    Example : Dimension = [2,2]
              the grah numbers are 1 2
                                   3 4

              Number = 3 corresponds to subplot [1,0]

            : To plot on a graph with 2 line and 3 columns on the graph in the center
            subplot = {"dimension":[3,3],"number":5}

    **kwargs : contient d'autres paramètres comme
             label : si jamais on veut ajouter un label à une donnée d'un graphique qui n'en aurait ou qui en aurait un autre
             add_graph = True : Si jamais on veut ajouter un autre graphique sur le dernier graphique tracé
                               : False par défaut, les nouvelles données seront tracées en effaçant les anciennes sur le subplot en cours
             legend_on : bool : argument contrôlant l'affichage de la légende
                       : True (par défaut) la légende s'affiche
                       : False La légende ne s'affiche pas'
                       : Argument utile seulement si on trace dans la dernière case d'un subplot ou dans un subplot 1x1 (il n'a pas d'effet autrement)
             LegendLocation = dictionary, controls where the legend is drawn outside the figure

                           location string of matplotlib 'upper right', 'center left'...

                           Default value : lower center (below the figure)

    """

    # get the customlabel if a label arguent is declared, puts None otherwise as a default value
    custom_label = kwargs.get("label", None)

    # Get add_graph function. Puts it to false by default if it's not declared in the kwargs
    add_graph = kwargs.get("add_graph", False)

    # get the legend_on argument that controls if the legend is drawn or not (Default True)
    legend_on = kwargs.get("legend_on", True)

    graph_type = "COP_graph"

    # Adds this special argument to the kwargs so that it is taken in account in other functions
    kwargs["draw_COP_points_on"] = draw_COP_points_on

    # Takes the components of the variable and the name of the variable
    composante_x = composantes[0]
    composante_y = [composantes[1]]
    variable_x = variable
    variable_y = variable

    # Stores the name of the x variable and y variable in the kwargs
    kwargs["variable_y"] = variable_y
    kwargs["variable_x"] = variable_x

    graph_annotation_on = kwargs.get("graph_annotation_on", True)
    # Overwrites the old value in case it was set to True by default
    # Because here value by default is true not false like other graph functions
    kwargs["graph_annotation_on"] = graph_annotation_on

    # Verifications for when simulationCases are used
    if cases_on:
        # Active tous les cas présents dans data
        if cases_on == "all":
            cases_on = list(data.keys())

        # Vérifie que Cases est toujours une liste si 'all' n'est pas utilisé
        elif not type(cases_on) is list:
            raise ValueError(
                "cases_on doit être une liste si 'all' n'est pas utilisé")
            return

        # Vérifie qu'on n'active pas plusieurs cas tout en comparant
        if len(cases_on) > 1 and compare:
            raise ValueError(
                "On ne peut pas comparer plusieurs simulations et plusieurs cas en même temps")
            return

    # flatten the data into a simulation cases form if compare=True
    if compare:
        # stores the case name that is being compared
        compared_case = cases_on[0]

        # resets the cases_on names to all the simulations names
        data, cases_on = flatten_compared_simulations(data, cases_on[0])
    else:
        compared_case = ""

    # Gets the figure size
    figsize = kwargs.get("figsize", None)

    fig = subplot_setup(subplot, figsize, add_graph)

    # Draws a contour only if there is one and sets the axis to be of equal ratio to keep the shape of the contour
    if COP_contour is not None:
        plt.plot(COP_contour[:, 0], COP_contour[:, 1], color='tab:blue')

        # Sets the aspect ratio between x and y axis to be equal
        # And makes the axis sizes adjustable
        plt.gca().set_aspect('equal', adjustable="datalim")

    # Selects the data to graph
    x_description, y_description = graph_select_data_to_plot(data, composante_x, composante_y, cases_on, compare, custom_label, compared_case, graph_type, **kwargs)

    # # if compare is False:
    #     # était dans compare is false
    # if cases_on is False:
    #     label = None
    #     plot_graph_functions(data, data[variable_x][composante_x], data[variable_y][composante_y],
    #                          graph_type, label=label, custom_label=custom_label, **kwargs)

    # # If the graph used is CasesGraph
    # else:
    #     for Case in cases_on:
    #         label = Case
    #         plot_graph_functions(data[Case], data[Case][variable_x][composante_x], data[Case][variable_y][composante_y], graph_type,
    #                              label=label, custom_label=custom_label, **kwargs)

    # # elif compare:

    # #     ListSimulations = list(data.keys())

    # #     for Simulation in ListSimulations:
    # #         label = Simulation

    # #         if cases_on is False:
    # #             plot_graph_functions(data[Simulation], data[Simulation][variable_x][composante_x], data[Simulation][variable_y][composante_y],
    # #                        graph_type, label=label, custom_label=custom_label, **kwargs)

    # #         # When we compare, we compare only one case between several simulations
    # #         elif len(cases_on) == 1:
    # #             plot_graph_functions(data[Simulation][cases_on[0]], data[Simulation][cases_on[0]][variable_x][composante_x], data[Simulation][cases_on[0]][variable_y]
    # #                        [composante_y], graph_type, label=label, custom_label=custom_label, **kwargs)

    if subplot is None:
        plt.title(figure_title)

        # unsuperpose the annotations if activated
        if graph_annotation_on:

            # Calls the function that will move the annotations to avoid superposition
            unsuperpose_plot_annotations(**kwargs)

        # shows the legend if activated
        if legend_on:
            legend_setup(fig, graph_type, **kwargs)

        # traces the axis labels
        plt.xlabel("<-----Postérieur              Antérieur----->")
        plt.ylabel("<----- Inférieur            Supérieur ----->")

        # Setups the grid and the axes ticks of the graph
        graph_grid_setup(fig, **kwargs)

    # Dans le cas d'un subplot
    else:

        # If a subplot title is entered, draws it
        if subplot_title:
            plt.title(subplot_title)

        # last_subplot can be entered in the subplot dictionary to oblige the legend to draw even if a subplot is empty
        # This statement has the priority over the test on the number of dimension
        if "last_subplot" in subplot:
            last_subplot = subplot["last_subplot"]

        # Tests if the number of subplot corresponds to the last subplot number to control if the legend and title are drawn or not
        elif subplot["number"] == subplot["dimension"][0] * subplot["dimension"][1]:
            last_subplot = True
        # Case where no legend and figure title will be drawn
        else:
            last_subplot = False

        # Setups the grid and the axes ticks of the graph
        graph_grid_setup(fig, last_subplot, **kwargs)

        # unsuperpose the annotations if activated
        if graph_annotation_on:

            # Calls the function that will move the annotations to avoid superposition
            unsuperpose_plot_annotations(**kwargs)
        # Displays the legend and figure title only if it's the last subplot drawn
        if last_subplot:

            # Trace le titre de la figure
            plt.suptitle(figure_title)

            # traces the axis labels
            plt.xlabel("<-----Postérieur              Antérieur----->")
            plt.ylabel("<----- Inférieur            Supérieur ----->")

            # Ajuste les distances entre les subplots quand ils sont tous tracés
            plt.tight_layout()

            # shows the legend if activated
            if legend_on:

                legend_setup(fig, graph_type, **kwargs)


# %% select data to plot

def check_result_dictionary_data_structure(data, cases_on, compare):
    """
    function used in the graph functions to check that the arguments compare and cases_on are used according to the result dictionary structure

    it checks if cases_on is used when simulation cases exis
    it checks if compare=True is used when we compare simulation with simulation cases
    """

    # counter that counts how deep the variables are stored which indicates the data structure
    # 0 : no simulation cases
    # 1 : simulation cases
    # 2 : compared simulation cases
    # 3 : error
    variables_deepness_counter = 0

    tested_data = data

    # searches for the entry "Loaded Variables" to know the data structure
    while "Loaded Variables" not in list(tested_data.keys()) and variables_deepness_counter < 3:
        # increases the ccuonter
        variables_deepness_counter += 1

        # goes one step deeper in the result dictionary
        tested_data = tested_data[list(tested_data.keys())[0]]

    # Check that the arguments entered match the result data structure
    # No simulation cases data structure
    if variables_deepness_counter == 0:
        if cases_on or compare:
            raise ValueError(f"For a result dictionary without simulation cases, cases_on and compare must both be False. \n While cases_on = {cases_on} and compare = {compare} were entered.")

    # Simulation cases data structure
    elif variables_deepness_counter == 1:
        if not cases_on or compare:
            raise ValueError(f"For a result dictionary with simulation cases, a cases_on list must be entered and compare must be False. \n While cases_on = {cases_on} and compare = {compare} were entered.")

    # Simulation cases comparison data structure
    elif variables_deepness_counter == 2:
        if not cases_on or not compare:
            raise ValueError(f"For a result dictionary that compares simulation cases, a cases_on list must be entered and compare must be True. \n While cases_on = {cases_on} and compare = {compare} were entered.")

    # When the variables are deeper than 2 levels
    else:
        raise ValueError("The result dictionary used doesn't have a correct data structure. The variables are {variables_deepness_counter} levels deep while 2 is the maximum!")


def graph_select_data_to_plot(data, composante_x, composante_y, cases_on, compare, custom_label, compared_case, graph_type, **kwargs):
    """
    selects the dictionary that contains the data, x and y data to be entered in plot_graph_functions

    prints errors (without stopping the graph) if some data to be grahped don't exist

    -------------------------------
    returns
    x_description and y_description : str
    the descriptions of the x and y variables to be set as xlabel and ylabel in the graph function

    """

    """
    AJOUTER UNE FONCTION QUI TEST LA STRUCTURE DES DICTIONAIRES ET FAIT ERREUR SELON LE CAS

    def check_graphed_data_structure(data, cases_on, compare)
    """

    # First checks that the results data structure match the argument entered in the graph function
    check_result_dictionary_data_structure(data, cases_on, compare)

    variable_x = kwargs.get("variable_x")
    variable_y = kwargs.get("variable_y")

    # initialises an empty dictionary with an empty description
    x_data = {"Description": ""}
    y_data = {"Description": ""}

    # error texts
    exc_x_error_text = ["The value : ", "doesn't exist in the x variable in : Results_dictionary"]
    exc_y_error_text = ["The value : ", "doesn't exist in the y variable in : Results_dictionary"]

    if compare:
        exc_x_error_text[1] += f"/{compared_case}"
        exc_y_error_text[1] += f"/{compared_case}"
        exc_case_error_text = ["The case : ", "doesn't exist in the simulation :", "in the results dictionary"]
    else:
        exc_case_error_text = ["The case : ", "doesn't exist in the results dictionary"]

    # S'il n'y a qu'une composante à tracer
    if len(composante_y) == 1:

        # Prend la valeur de la composante comme elle est seule
        composante_y = composante_y[0]

        # The program stops if there is an error because only one value to graph
        if cases_on is False:
            # defines the y_error text as empty
            error_y_text = ""

            label = None

            # checks that x exists in data
            try:

                x_data = data[variable_x]
                x = x_data[composante_x]

                # checks that y exists in data
                try:

                    y_data = data[variable_y]
                    y = y_data[composante_y]

                    plot_graph_functions(data, x, y, graph_type, label=label, custom_label=custom_label, **kwargs)

                except Exception as exc_y:
                    # Stores the y_error text to print as an error later
                    error_y_text = f"{exc_y_error_text[0]} {str(exc_y)} {exc_y_error_text[1]}/{variable_y}/{composante_y} \n"
                    raise ValueError("")
                    sys.exit(1)

            except Exception as exc_x:
                # If no error on y
                if not error_y_text:
                    raise ValueError(f"{exc_x_error_text[0]} {str(exc_x)} {exc_x_error_text[1]}/{variable_x}/{composante_x} \n")
                # if there is an error on y, raise the y error text string
                else:
                    raise ValueError(error_y_text)
                sys.exit(1)

        # If the data has simulation cases
        else:
            for Case in cases_on:
                label = Case

                # checks that the case exists in data
                try:
                    case_data = data[Case]

                    # checks that x exists in data
                    try:

                        x_data = data[Case][variable_x]
                        x = x_data[composante_x]

                        # checks that y exists in data
                        try:

                            y_data = data[Case][variable_y]
                            y = y_data[composante_y]

                            plot_graph_functions(case_data, x, y, graph_type, label=label, custom_label=custom_label, **kwargs)

                        except Exception as exc_y:
                            print(f"{exc_y_error_text[0]} {str(exc_y)} {exc_y_error_text[1]}/{Case}/{variable_y}/{composante_y} \n")

                    except Exception as exc_x:
                        print(f"{exc_x_error_text[0]} {str(exc_x)} {exc_x_error_text[1]}/{Case}/{variable_x}/{composante_x} \n")

                except Exception:
                    if compare:
                        raise ValueError(f"{exc_case_error_text[0]}{compared_case} {exc_case_error_text[1]} {Case} {exc_case_error_text[2]}\n")
                    else:
                        raise ValueError(f"{exc_case_error_text[0]}{Case}{exc_case_error_text[1]} \n")
                    sys.exit(1)

    # Si plusieurs composantes sont activées
    else:

        # On ne peut comparer que si on active la même donnée, donc seulement une seule composante
        # if compare is False:
        for Composante in composante_y:
            label = Composante

            if cases_on is False:

                # checks that x exists in data
                try:

                    x_data = data[variable_x]
                    x = x_data[composante_x]

                    # checks that y exists in data
                    try:

                        y_data = data[variable_y]
                        y = y_data[Composante]

                        plot_graph_functions(data, x, y, graph_type, label=label, custom_label=custom_label, **kwargs)

                    except Exception as exc_y:
                        print(f"{exc_y_error_text[0]} {str(exc_y)} {exc_y_error_text[1]}/{variable_y}/{Composante} \n")

                except Exception as exc_x:
                    print(f"{exc_x_error_text[0]} {str(exc_x)} {exc_x_error_text[1]}/{variable_x}/{composante_x} \n")

            # On peut tracer plusieurs composantes seulement si un seul cas de simulation est activé
            elif len(cases_on) == 1:

                # get the case name
                Case = cases_on[0]

                # checks that the case exists in data
                try:
                    case_data = data[Case]

                    # checks that x exists in data
                    try:

                        x_data = data[Case][variable_x]
                        x = x_data[composante_x]

                        # checks that y exists in data
                        try:

                            y_data = data[Case][variable_y]
                            y = y_data[Composante]

                            plot_graph_functions(case_data, x, y, graph_type, label=label, custom_label=custom_label, **kwargs)

                        except Exception as exc_y:
                            print(f"{exc_y_error_text[0]} {str(exc_y)} {exc_y_error_text[1]}/{Case}/{variable_y}/{Composante} \n")

                    except Exception as exc_x:
                        print(f"{exc_x_error_text[0]} {str(exc_x)} {exc_x_error_text[1]}/{Case}/{variable_x}/{composante_x} \n")

                except Exception:
                    if compare:
                        raise ValueError(f"{exc_case_error_text[0]}{compared_case} {exc_case_error_text[1]} {Case} {exc_case_error_text[2]}\n")
                    else:
                        raise ValueError(f"{exc_case_error_text[0]}{Case}{exc_case_error_text[1]} \n")
                    sys.exit(1)

    # Returns the description for each axis to be applied later
    x_description = x_data["Description"]
    y_description = y_data["Description"]

    return x_description, y_description


def muscle_graph_select_data_to_plot(data, composante_x, composante_y, cases_on, compare, custom_label, compared_case, graph_type, MusclePartInformation, MusclePart, MuscleFolder, MuscleName, **kwargs):
    """
    selects the dictionary that contains the data, x and y data to be entered in plot_graph_functions

    prints errors (without stopping the graph) if some data to be grahped don't exist

    -------------------------------
    returns
    x_description and y_description : str
    the descriptions of the x and y variables to be set as xlabel and ylabel in the graph function

    """

    """
    AJOUTER UNE FONCTION QUI TEST LA STRUCTURE DES DICTIONAIRES ET FAIT ERREUR SELON LE CAS

    def check_graphed_data_structure(data, cases_on, compare)
    """

    # First checks that the results data structure match the argument entered in the graph function
    check_result_dictionary_data_structure(data, cases_on, compare)

    variable_x = kwargs.get("variable_x")
    variable_y = kwargs.get("variable_y")

    # initialises an empty dictionary with an empty description
    x_data = {"Description": ""}
    y_data = {"Description": ""}

    # error texts
    exc_x_error_text = ["The value : ", "doesn't exist in the x variable in : Results_dictionary"]
    exc_y_error_text = ["The value : ", "doesn't exist in the y variable in : Results_dictionary"]

    if compare:
        exc_x_error_text[1] += f"/{compared_case}"
        exc_y_error_text[1] += f"/{compared_case}"
        exc_case_error_text = ["The case : ", "doesn't exist in the simulation :", "in the results dictionary"]
    else:
        exc_case_error_text = ["The case : ", "doesn't exist in the results dictionary"]

    # S'il n'y a qu'une composante à tracer
    if len(composante_y) == 1:

        # Prend la valeur de la composante comme elle est seule
        composante_y = composante_y[0]

        # The program stops if there is an error because only one value to graph
        if cases_on is False:

            error_y_text = ""

            # Si plus d'une muscle part est tracée, on met une legende avec le nom de la musclepart
            if MusclePartInformation["Total Number Muscle Parts"] > 1:
                label = MusclePart

            # Si seulement une muscle part est activée et qu'on ne compare pas, on ne met pas de légende
            else:
                label = None

            # checks that x exists in data
            try:

                x_data = data[variable_x]
                x = x_data[composante_x]

                # checks that y exists in data
                try:

                    y_data = data[MuscleFolder][MuscleName][MusclePart][variable_y]
                    y = y_data[composante_y]

                    plot_graph_functions(data, x, y, graph_type, label=label, custom_label=custom_label, **kwargs)

                except Exception as exc_y:
                    # Stores the y_error text to print as an error later
                    error_y_text = f"{exc_y_error_text[0]} {str(exc_y)} {exc_y_error_text[1]}/{MuscleFolder}/{MuscleName}/{MusclePart}/{variable_y}/{composante_y} \n"
                    raise ValueError("")

                    sys.exit()

            except Exception as exc_x:
                # If no error on y
                if not error_y_text:
                    raise ValueError(f"{exc_x_error_text[0]} {str(exc_x)} {exc_x_error_text[1]}/{variable_x}/{composante_x} \n")
                # if there is an error on y, raise the y error text string
                else:
                    raise ValueError(error_y_text)
                sys.exit(1)

        # If the data has simulation cases
        else:
            # On ne peut tracer qu'une seule donnée, donc on doit avoir soit un seul Case de sélectionné et n>=1 muscle parts
            # Ou on peut avoir plusieurs Case de sélectionnés mais une seule muscle part à tracer
            if len(cases_on) == 1 or MusclePartInformation["Total Number Muscle Parts"] == 1:
                for Case in cases_on:
                    # La légende est le nom du case si il n'y a qu'une seule muscle part à tracer et plus d'un Case sélectionné
                    if len(cases_on) > 1 and MusclePartInformation["Total Number Muscle Parts"] == 1:
                        label = Case

                    # La légende est le nom de la muscle part s'il n'y a qu'un seul case et plusieurs Muscle part à tracer
                    elif len(cases_on) == 1 and MusclePartInformation["Total Number Muscle Parts"] > 1:
                        label = MusclePart

                    # Si les deux sont 1, on ne met pas de légende
                    else:
                        label = None

                    # checks that the case exists in data
                    try:
                        case_data = data[Case]

                        # checks that x exists in data
                        try:

                            x_data = data[Case][variable_x]
                            x = x_data[composante_x]

                            # checks that y exists in data
                            try:

                                y_data = data[Case][MuscleFolder][MuscleName][MusclePart][variable_y]
                                y = y_data[composante_y]

                                plot_graph_functions(case_data, x, y, graph_type, label=label, custom_label=custom_label, **kwargs)

                            except Exception as exc_y:
                                print(f"{exc_y_error_text[0]} {str(exc_y)} {exc_y_error_text[1]}/{Case}/{MuscleFolder}/{MuscleName}/{MusclePart}/{variable_y}/{composante_y} \n")

                        except Exception as exc_x:
                            print(f"{exc_x_error_text[0]} {str(exc_x)} {exc_x_error_text[1]}/{Case}/{variable_x}/{composante_x} \n")

                    except Exception:
                        if compare:
                            raise ValueError(f"{exc_case_error_text[0]}{compared_case} {exc_case_error_text[1]} {Case} {exc_case_error_text[2]}\n")
                        else:
                            raise ValueError(f"{exc_case_error_text[0]}{Case}{exc_case_error_text[1]} \n")
                        sys.exit(1)

    # Si plusieurs composantes sont activées
    else:

        # On ne peut comparer que si on active la même donnée, donc seulement une seule composante
        # if compare is False:
        for Composante in composante_y:
            label = Composante

            if cases_on is False:

                # checks that x exists in data
                try:

                    x_data = data[variable_x]
                    x = x_data[composante_x]

                    # checks that y exists in data
                    try:

                        y_data = data[MuscleFolder][MuscleName][MusclePart][variable_y]
                        y = y_data[Composante]

                        plot_graph_functions(data, x, y, graph_type, label=label, custom_label=custom_label, **kwargs)

                    except Exception as exc_y:
                        print(f"{exc_y_error_text[0]} {str(exc_y)} {exc_y_error_text[1]}/{MuscleFolder}/{MuscleName}/{MusclePart}/{variable_y}/{Composante} \n")

                except Exception as exc_x:
                    print(f"{exc_x_error_text[0]} {str(exc_x)} {exc_x_error_text[1]}/{variable_x}/{composante_x} \n")

            # On peut tracer plusieurs composantes seulement si un seul cas de simulation est activé
            elif len(cases_on) == 1:

                # get the case name
                Case = cases_on[0]

                # checks that the case exists in data
                try:
                    case_data = data[Case]

                    # checks that x exists in data
                    try:

                        x_data = data[Case][variable_x]
                        x = x_data[composante_x]

                        # checks that y exists in data
                        try:

                            y_data = data[Case][MuscleFolder][MuscleName][MusclePart][variable_y][variable_y]
                            y = y_data[Composante]

                            plot_graph_functions(case_data, x, y, graph_type, label=label, custom_label=custom_label, **kwargs)

                        except Exception as exc_y:
                            print(f"{exc_y_error_text[0]} {str(exc_y)} {exc_y_error_text[1]}/{Case}/{MuscleFolder}/{MuscleName}/{MusclePart}/{variable_y}/{Composante} \n")

                    except Exception as exc_x:
                        print(f"{exc_x_error_text[0]} {str(exc_x)} {exc_x_error_text[1]}/{Case}/{variable_x}/{composante_x} \n")

                except Exception:
                    if compare:
                        raise ValueError(f"{exc_case_error_text[0]}{compared_case} {exc_case_error_text[1]} {Case} {exc_case_error_text[2]}\n")
                    else:
                        raise ValueError(f"{exc_case_error_text[0]}{Case}{exc_case_error_text[1]} \n")
                    sys.exit(1)

    """
    old code
    """
    # # S'il n'y a qu'une composante à tracer
    # if len(composante_y) == 1:

    #     # Prend la valeur de la composante comme elle est seule
    #     composante_y = composante_y[0]

    #     # if compare is False:
    #     # était dans compare is false
    #     if cases_on is False:

    #         # Si plus d'une muscle part est tracée, on met une legende avec le nom de la musclepart
    #         if MusclePartInformation["Total Number Muscle Parts"] > 1:
    #             label = MusclePart

    #         # Si seulement une muscle part est activée et qu'on ne compare pas, on ne met pas de légende
    #         else:
    #             label = None

    #         plot_graph_functions(data, data[variable_x][composante_x], data[MuscleFolder][MuscleName]
    #                              [MusclePart][variable_y][composante_y], graph_type, label=label, custom_label=custom_label, **kwargs)

    #     # with simulation cases
    #     else:
    #         # On ne peut tracer qu'une seule donnée, donc on doit avoir soit un seul Case de sélectionné et n>=1 muscle parts
    #         # Ou on peut avoir plusieurs Case de sélectionnés mais une seule muscle part à tracer
    #         if len(cases_on) == 1 or MusclePartInformation["Total Number Muscle Parts"] == 1:

    #             for Case in cases_on:

    #                 # La légende est le nom du case si il n'y a qu'une seule muscle part à tracer et plus d'un Case sélectionné
    #                 if len(cases_on) > 1 and MusclePartInformation["Total Number Muscle Parts"] == 1:
    #                     label = Case

    #                 # La légende est le nom de la muscle part s'il n'y a qu'un seul case et plusieurs Muscle part à tracer
    #                 elif len(cases_on) == 1 and MusclePartInformation["Total Number Muscle Parts"] > 1:
    #                     label = MusclePart

    #                 # Si les deux sont 1, on ne met pas de légende
    #                 else:
    #                     label = None

    #                 plot_graph_functions(data[Case], data[Case][variable_x][composante_x], data[Case][MuscleFolder][MuscleName]
    #                                      [MusclePart][variable_y][composante_y], graph_type, label=label, custom_label=custom_label, **kwargs)

    #     # elif compare:

    #     #     # Si on a plusieurs simulations, on ne peut afficher qu'une seule donnée sur le graphique, donc qu'une seule muscle part
    #     #     if MusclePartInformation["Total Number Muscle Parts"] == 1:
    #     #         ListSimulations = list(data.keys())

    #     #         for Simulation in ListSimulations:
    #     #             label = Simulation

    #     #             if cases_on is False:
    #     #                 plot_graph_functions(data[Simulation], data[Simulation][variable_x][composante_x], data[Simulation][MuscleFolder]
    #     #                            [MuscleName][MusclePart][variable_y][composante_y], graph_type, label=label, custom_label=custom_label, **kwargs)

    #     #             # When we compare, we compare only one case between several simulations
    #     #             elif len(cases_on) == 1:
    #     #                 plot_graph_functions(data[Simulation][cases_on[0]], data[Simulation][cases_on[0]][variable_x][composante_x], data[Simulation][cases_on[0]]
    #     #                            [MuscleFolder][MuscleName][MusclePart][variable_y][composante_y], graph_type, label=label, custom_label=custom_label, **kwargs)

    # # Si plusieurs composantes sont activées
    # else:

    #     # Si on a plusieurs composantes, on ne peut afficher qu'une seule donnée sur le graphique, donc qu'une seule muscle part
    #     if MusclePartInformation["Total Number Muscle Parts"] == 1:

    #         # # On ne peut comparer plusieurs simulations que si on active la même donnée, on ne peut pas afficher plusieurs composantes avec plusieurs simulations
    #         # if compare is False:

    #         for Composante in composante_y:
    #             label = Composante

    #             if cases_on is False:
    #                 plot_graph_functions(data, data[variable_x][composante_x], data[MuscleFolder][MuscleName]
    #                                      [MusclePart][variable_y][Composante], graph_type, label=label, custom_label=custom_label, **kwargs)

    #             # On peut tracer plusieurs composantes seulement si un seul cas de simulation est activé
    #             elif len(cases_on) == 1:
    #                 plot_graph_functions(data, data[cases_on[0]][variable_x][composante_x], data[cases_on[0]][MuscleFolder]
    #                                      [MuscleName][MusclePart][variable_y][Composante], graph_type, label=label, custom_label=custom_label, **kwargs)

    # Returns the description for each axis to be applied later
    x_description = x_data["Description"]
    y_description = y_data["Description"]

    return x_description, y_description


# %% old graph functions without flatten for compare 10 november


def COP_graph_old(data, COP_contour=None, variable="COP", figure_title="", composantes=["x", "y"], cases_on=False, subplot=None, compare=False, subplot_title=False, draw_COP_points_on=True, **kwargs):
    """

    Fait le graphique de la position d'un centre de pression et trace un contour (contour d'un implant ou de la surface de contact par exemple)
    data doit avoir une variable appelée "COP"
    COP doit avoir la séquence : "SequenceComposantes": ["AP", "IS", "ML"]
    Trace la composante antéropostérieure (AP) en abscisse (Antérieur = positif) et inférosupérieure (IS) en ordonnée (Supérieur = Positif)

    draw_GH_reactions_nodes= ARGUMENT PERSONNEL POUR TRACER EN PLUS DES POINT SUR LE CONTOUR, PEUT ÊTRE SUPPRIMÉ

    COP_contour : numpy.array contenant les coordonnées x, y du contour à tracer (Peut être créé par la fonction define_COP_contour si on veut lire un fichier contenant ces coordonnées)
               : Dimension (npoints,3 or 2)
                 Column 1 : x
                 Column 2 : y
                 Column 3 : z (not used by the COP_graph function)

    variable : string : The name of the variable to draw
    (Default "COP")

    composantes : list : ["composante_x", "composante_y"]
                : Composantes de la variable à tracer
                (Default ["x", "y"])

    data : le dictionnaire contenant les data à tracer
         : Par défaut : Un dictionnaire ne contenant qu'une seule simulation
         : Soit un jeu de plusieurs datas (compare = True)

    graph_annotation_on : bool : Contrôle l'affichage ou non des angles de pic de COP (Vrai par défaut)

    draw_COP_points_on : bool : active ou non le traçage des points sur le COP

    compare : = True si on veut comparer plusieurs données
              Ne rien mettre (compare = False par défaut) : on veut tracer qu'une seule donnée

    subplot = {"Dimension: [nrows, ncolumns]", :"number": Number_of_the_subplot_selected, "figsize": [horizontal_size_inches, vertical_size_inches], "last_subplot": True}


        subplot["dimension"] = [nrows, ncolumns]
    And defines the active axis as the subplot["number"]=number of the plot


    subplot["figsize"] : Optional argument to set the size of the figure
                         subplot["figsize"] = [horizontal_size_inches, vertical_size_inches]
                         : default : [14, 10] inches for 2D ; [7, 5] for [1,1] subplot

    subplot["dimension"] and figsize : are only to be set for subplot["number"] = 1
    They are not taken in account otherwise

    subplot["LastPart"] : bool = Optional argument : Controls if the legend and figure title are drawn
                        : It's automatically set to True if we reach the maximum subplot Number
                        : But it can be overwritten so that the legend is drawn even if one of the subplot is empty


    Example : Dimension = [2,2]
              the grah numbers are 1 2
                                   3 4

              Number = 3 corresponds to subplot [1,0]

            : To plot on a graph with 2 line and 3 columns on the graph in the center
            subplot = {"dimension":[3,3],"number":5}

    **kwargs : contient d'autres paramètres comme
             label : si jamais on veut ajouter un label à une donnée d'un graphique qui n'en aurait ou qui en aurait un autre
             add_graph = True : Si jamais on veut ajouter un autre graphique sur le dernier graphique tracé
                               : False par défaut, les nouvelles données seront tracées en effaçant les anciennes sur le subplot en cours
             legend_on : bool : argument contrôlant l'affichage de la légende
                       : True (par défaut) la légende s'affiche
                       : False La légende ne s'affiche pas'
                       : Argument utile seulement si on trace dans la dernière case d'un subplot ou dans un subplot 1x1 (il n'a pas d'effet autrement)
             LegendLocation = dictionary, controls where the legend is drawn outside the figure

                           location string of matplotlib 'upper right', 'center left'...

                           Default value : lower center (below the figure)

    """

    # get the customlabel if a label arguent is declared, puts None otherwise as a default value
    custom_label = kwargs.get("label", None)

    # Get add_graph function. Puts it to false by default if it's not declared in the kwargs
    add_graph = kwargs.get("add_graph", False)

    # get the legend_on argument that controls if the legend is drawn or not (Default True)
    legend_on = kwargs.get("legend_on", True)

    graph_type = "COP_graph"

    # Adds this special argument to the kwargs so that it is taken in account in other functions
    kwargs["draw_COP_points_on"] = draw_COP_points_on

    # Takes the components of the variable and the name of the variable
    composante_x = composantes[0]
    composante_y = composantes[1]
    variable_x = variable
    variable_y = variable

    # Stores the name of the x variable and y variable in the kwargs
    kwargs["variable_y"] = variable_y
    kwargs["variable_x"] = variable_x

    graph_annotation_on = kwargs.get("graph_annotation_on", True)
    # Overwrites the old value in case it was set to True by default
    # Because here value by default is true not false like other graph functions
    kwargs["graph_annotation_on"] = graph_annotation_on

    # Verifications for when simulationCases are used
    if cases_on:
        # Active tous les cas présents dans data
        if cases_on == "all":
            cases_on = list(data.keys())

        # Vérifie que Cases est toujours une liste si 'all' n'est pas utilisé
        elif not type(cases_on) is list:
            raise ValueError(
                "cases_on doit être une liste si 'all' n'est pas utilisé")
            return

        # Vérifie qu'on n'active pas plusieurs cas tout en comparant
        if len(cases_on) > 1 and compare:
            raise ValueError(
                "On ne peut pas comparer plusieurs simulations et plusieurs cas en même temps")
            return

    # Gets the figure size
    figsize = kwargs.get("figsize", None)

    fig = subplot_setup(subplot, figsize, add_graph)

    # Draws a contour only if there is one and sets the axis to be of equal ratio to keep the shape of the contour
    if COP_contour is not None:
        plt.plot(COP_contour[:, 0], COP_contour[:, 1], color='tab:blue')

        # Sets the aspect ratio between x and y axis to be equal
        # And makes the axis sizes adjustable
        plt.gca().set_aspect('equal', adjustable="datalim")

    if compare is False:

        if cases_on is False:
            label = None
            plot_graph_functions(data, data[variable_x][composante_x], data[variable_y][composante_y],
                                 graph_type, label=label, custom_label=custom_label, **kwargs)

        # If the graph used is CasesGraph
        else:
            for Case in cases_on:
                label = Case
                plot_graph_functions(data[Case], data[Case][variable_x][composante_x], data[Case][variable_y][composante_y], graph_type,
                                     label=label, custom_label=custom_label, **kwargs)

    elif compare:

        ListSimulations = list(data.keys())

        for Simulation in ListSimulations:
            label = Simulation

            if cases_on is False:
                plot_graph_functions(data[Simulation], data[Simulation][variable_x][composante_x], data[Simulation][variable_y][composante_y],
                                     graph_type, label=label, custom_label=custom_label, **kwargs)

            # When we compare, we compare only one case between several simulations
            elif len(cases_on) == 1:
                plot_graph_functions(data[Simulation][cases_on[0]], data[Simulation][cases_on[0]][variable_x][composante_x], data[Simulation][cases_on[0]][variable_y]
                                     [composante_y], graph_type, label=label, custom_label=custom_label, **kwargs)

    plt.xlabel("<-----Postérieur              Antérieur----->")
    plt.ylabel("<----- Inférieur            Supérieur ----->")

    if subplot is None:
        plt.title(figure_title)

        # unsuperpose the annotations if activated
        if graph_annotation_on:

            # Calls the function that will move the annotations to avoid superposition
            unsuperpose_plot_annotations(**kwargs)

        # shows the legend if activated
        if legend_on:
            legend_setup(fig, graph_type, **kwargs)

        # Setups the grid and the axes ticks of the graph
        graph_grid_setup(fig, **kwargs)

    # Dans le cas d'un subplot
    else:

        # If a subplot title is entered, draws it
        if subplot_title:
            plt.title(subplot_title)

        # last_subplot can be entered in the subplot dictionary to oblige the legend to draw even if a subplot is empty
        # This statement has the priority over the test on the number of dimension
        if "last_subplot" in subplot:
            last_subplot = subplot["last_subplot"]

        # Tests if the number of subplot corresponds to the last subplot number to control if the legend and title are drawn or not
        elif subplot["number"] == subplot["dimension"][0] * subplot["dimension"][1]:
            last_subplot = True
        # Case where no legend and figure title will be drawn
        else:
            last_subplot = False

        # Setups the grid and the axes ticks of the graph
        graph_grid_setup(fig, last_subplot, **kwargs)

        # unsuperpose the annotations if activated
        if graph_annotation_on:

            # Calls the function that will move the annotations to avoid superposition
            unsuperpose_plot_annotations(**kwargs)
        # Displays the legend and figure title only if it's the last subplot drawn
        if last_subplot:

            # Trace le titre de la figure
            plt.suptitle(figure_title)

            # Ajuste les distances entre les subplots quand ils sont tous tracés
            plt.tight_layout()

            # shows the legend if activated
            if legend_on:

                legend_setup(fig, graph_type, **kwargs)


def graph_old(data, variable_x, variable_y, figure_title, cases_on=False, compare=False, composante_x="Total", composante_y=["Total"], subplot=None, subplot_title=False, **kwargs):
    """
    Fonction générale qui gère les graphiques


    data : le dictionnaire contenant les data à tracer
         : Par défaut : Un dictionnaire ne contenant qu'une seule simulation
         : Soit un jeu de plusieurs datas (compare = True)

    variable_x : Le nom de la variable placée en x sur le graphique
    variable_y : le nom de la variable placée en y sur le graphique

    composante_y :
                  : type : liste de chaines de charactère
                  : Liste contenant les nom des composantes de la variable à tracer
                  : Par défaut : On trace la composante "Total" donc composante_y = ["Total"]

                : Activer plusieurs composantes :
                Exemple : composante_y = ["composante 1","composante 2","composante 3","Total"....]
                          Si on veut activer x et y entrer : composante_y = ["x","y"]

                : Activer une seule composante :
                Exemple : Si on veut activer y entrer : composante_y = ["y"]


                CAS PARTICULIER COMPOSANTES: Si on compare, on ne peut activer qu'une seule composante
                                           : Si on active plusieurs composantes, on doit comparer la même donnée (un seul cas de simulation)

    Composantes_x : Le nom de la composante de la variable en abscisse
                  : composante_x est une chaîne de charactère contenant le nom de la composante de la variable
                  : Par défaut : "Total"
                  : Si on veut activer y entrer : Composantes_x = "y"

    compare : = True si on veut comparer plusieurs données
              Ne rien mettre (compare = False par défaut) : on veut tracer qu'une seule donnée

    subplot = {"Dimension: [nrows, ncolumns]", :"number": Number_of_the_subplot_selected, "figsize": [horizontal_size_inches, vertical_size_inches], "last_subplot": True}


        subplot["dimension"] = [nrows, ncolumns]
    And defines the active axis as the subplot["number"]=number of the plot


    subplot["figsize"] : Optional argument to set the size of the figure
                         subplot["figsize"] = [horizontal_size_inches, vertical_size_inches]
                         : default : [14, 10] inches for 2D ; [7, 5] for [1,1] subplot

    subplot["dimension"] and figsize : are only to be set for subplot["number"] = 1
    They are not taken in account otherwise

    subplot["LastPart"] : bool = Optional argument : Controls if the legend and figure title are drawn
                        : It's automatically set to True if we reach the maximum subplot Number
                        : But it can be overwritten so that the legend is drawn even if one of the subplot is empty


    Example : Dimension = [2,2]
              the grah numbers are 1 2
                                   3 4

              Number = 3 corresponds to subplot [1,0]

            : To plot on a graph with 2 line and 3 columns on the graph in the center
            subplot = {"dimension":[3,3],"number":5}

    **kwargs : contient d'autres paramètres comme
             label : si jamais on veut ajouter un label à une donnée d'un graphique qui n'en aurait ou qui en aurait un autre
             add_graph = True : Si jamais on veut ajouter un autre graphique sur le dernier graphique tracé
                               : False par défaut, les nouvelles données seront tracées en effaçant les anciennes sur le subplot en cours
             legend_on : bool : argument contrôlant l'affichage de la légende
                       : True (par défaut) la légende s'affiche
                       : False La légende ne s'affiche pas'
             LegendLocation = dictionary, controls where the legend is drawn outside the figure

                           location string of matplotlib 'upper right', 'center left'...

                           Default value : lower center (below the figure)
    """

    # get the customlabel if a label arguent is declared, puts None otherwise as a default value
    custom_label = kwargs.get("label", None)

    # Get the add_graph variable. Puts it to false by default if it's not declared in the kwargs
    add_graph = kwargs.get("add_graph", False)

    # get the legend_on argument that controls if the legend is drawn or not (Default True)
    legend_on = kwargs.get("legend_on", True)

    # Stores the name of the x variable and y variable in the kwargs
    kwargs["variable_y"] = variable_y
    kwargs["variable_x"] = variable_x

    graph_annotation_on = kwargs.get("graph_annotation_on", False)

    graph_type = "graph"

    # Verifications for when simulationCases are used
    if cases_on:
        # If "all", all cases are selected to be drawn
        if cases_on == "all":
            cases_on = list(data.keys())

        elif type(cases_on) is str:
            raise ValueError(
                "cases_on doit être une liste si 'all' n'est pas utilisé")
            return

        # Vérifie qu'on n'active pas plusieurs cas tout en comparant
        if len(cases_on) > 1 and compare:
            raise ValueError(
                "On ne peut pas comparer plusieurs simulations et plusieurs cas en même temps")
            return

        # Vérifie qu'on ne dessine pas plusieurs variables tout en dessinant plusieurs cas
        if len(cases_on) > 1 and len(composante_y) > 1:
            raise ValueError(
                "On ne peut pas dessiner plusieurs cas et plusieurs composantes en même temps")
            return

    # Vérification qu'on ne dessine pas plusieurs variables tout en comparant
    if compare and len(composante_y) > 1:
        raise ValueError(
            "On ne peut pas comparer plusieurs simulations et dessiner plusieurs composantes")
        return

    # Gets the figure size
    figsize = kwargs.get("figsize", None)

    fig = subplot_setup(subplot, figsize, add_graph)

    # S'il n'y a qu'une composante à tracer
    if len(composante_y) == 1:

        # Prend la valeur de la composante comme elle est seule
        composante_y = composante_y[0]

        if compare is False:

            if cases_on is False:
                label = None
                plot_graph_functions(data, data[variable_x][composante_x],
                                     data[variable_y][composante_y], graph_type, label=label, custom_label=custom_label, **kwargs)

            # If the graph used is CasesGraph
            else:
                for Case in cases_on:
                    label = Case

                    plot_graph_functions(data[Case], data[Case][variable_x][composante_x], data[Case]
                                         [variable_y][composante_y], graph_type, label=label, custom_label=custom_label, **kwargs)

        elif compare:

            ListSimulations = list(data.keys())

            for Simulation in ListSimulations:
                # Definds the color of this simulation depending on its name
                label = Simulation

                if cases_on is False:
                    plot_graph_functions(data[Simulation], data[Simulation][variable_x][composante_x], data[Simulation]
                                         [variable_y][composante_y], graph_type, label=label, custom_label=custom_label, **kwargs)

                # When we compare, we compare only one case between several simulations
                elif len(cases_on) == 1:
                    plot_graph_functions(data[Simulation][cases_on[0]], data[Simulation][cases_on[0]][variable_x][composante_x],
                                         data[Simulation][cases_on[0]][variable_y][composante_y], graph_type, label=label, custom_label=custom_label, **kwargs)

    # Si plusieurs composantes sont activées
    else:

        # On ne peut comparer que si on active la même donnée, donc seulement une seule composante
        if compare is False:
            for Composante in composante_y:
                label = Composante

                if cases_on is False:
                    plot_graph_functions(data, data[variable_x][composante_x], data[variable_y]
                                         [Composante], graph_type, label=label, custom_label=custom_label, **kwargs)

                # On peut tracer plusieurs composantes seulement si un seul cas de simulation est activé
                elif len(cases_on) == 1:
                    plot_graph_functions(data, data[cases_on[0]][variable_x][composante_x], data[cases_on[0]]
                                         [variable_y][Composante], graph_type, label=label, custom_label=custom_label, **kwargs)

    # Axis Labels from the variable description
    if compare:
        if cases_on is False:
            plt.xlabel(data[ListSimulations[0]][variable_x]["Description"])
            plt.ylabel(data[ListSimulations[0]][variable_y]["Description"])
        else:
            plt.xlabel(data[ListSimulations[0]][cases_on[0]]
                       [variable_x]["Description"])
            plt.ylabel(data[ListSimulations[0]][cases_on[0]]
                       [variable_y]["Description"])
    elif compare is False:
        if cases_on is False:
            plt.xlabel(data[variable_x]["Description"])
            plt.ylabel(data[variable_y]["Description"])
        else:
            plt.xlabel(data[cases_on[0]][variable_x]["Description"])
            plt.ylabel(data[cases_on[0]][variable_y]["Description"])

    if subplot is None:
        plt.title(figure_title)

        # unsuperpose the annotations if activated
        if graph_annotation_on:

            # Calls the function that will move the annotations to avoid superposition
            unsuperpose_plot_annotations(**kwargs)

        # shows the legend if activated
        if legend_on:
            legend_setup(fig, graph_type, **kwargs)

        # Setups the grid and the axes ticks of the graph
        graph_grid_setup(fig, **kwargs)

    # Dans le cas d'un subplot
    else:

        # If a subplot title is entered, draws it (subplot_title isn't a bool)
        if subplot_title:
            plt.title(subplot_title)

        # unsuperpose the annotations if activated
        if graph_annotation_on:

            # Calls the function that will move the annotations to avoid superposition
            unsuperpose_plot_annotations(**kwargs)

        # last_subplot can be entered in the subplot dictionary to oblige the legend to draw even if a subplot is empty
        # This statement has the priority over the test on the number of dimension
        if "last_subplot" in subplot:
            last_subplot = subplot["last_subplot"]

        # Tests if the number of subplot corresponds to the last subplot number to control if the legend and title are drawn or not
        elif subplot["number"] == subplot["dimension"][0] * subplot["dimension"][1]:
            last_subplot = True
        # Case where no legend and figure title will be drawn
        else:
            last_subplot = False

        # Setups the grid and the axes ticks of the graph
        graph_grid_setup(fig, last_subplot, **kwargs)

        # Displays the legend and figure title only if it's the last subplot drawn
        if last_subplot:

            # Trace le titre de la figure
            plt.suptitle(figure_title)

            # Ajuste les distances entre les subplots quand ils sont tous tracés
            plt.tight_layout()

            # shows the legend if activated
            if legend_on:
                legend_setup(fig, graph_type, **kwargs)


def muscle_part_graph_old(data, MuscleName, MusclePart, variable_x, variable_y, figure_title, composante_x="Total", composante_y=["Total"], compare=False, subplot=None, subplot_title=False, cases_on=False, MusclePartInformation=False, fig=None, **kwargs):
    """
    Fonction qui gère trace la variable d'une seule fibre musculaire

    lastPart = statement pour dire qu'on dessine la dernière musclepart pour ne tracer la légende qu'à ce moment là


    data : le dictionnaire contenant les data à tracer
         : Par défaut : Un dictionnaire ne contenant qu'une seule simulation
         : Soit un jeu de plusieurs datas (compare = True)

    variable_x : Le nom de la variable placée en x sur le graphique
    variable_y : le nom de la variable placée en y sur le graphique

    composante_y :
                  : type : liste de chaines de charactère
                  : Liste contenant les nom des composantes de la variable à tracer
                  : Par défaut : On trace la composante "Total" donc composante_y = ["Total"]

                : Activer plusieurs composantes :
                Exemple : composante_y = ["composante 1","composante 2","composante 3","Total"....]
                          Si on veut activer x et y entrer : composante_y = ["x","y"]

                : Activer une seule composante :
                Exemple : Si on veut activer y entrer : composante_y = ["y"]

                CAS PARTICULIER COMPOSANTES: Si on compare, on ne peut activer qu'une seule composante
                                           : Si on active plusieurs composantes, on doit comparer la même donnée (un seul cas de simulation)

    Composantes_x : Le nom de la composante de la variable en abscisse
                  : composante_x est une chaîne de charactère contenant le nom de la composante de la variable
                  : Par défaut : "Total"
                  : Si on veut activer y entrer : Composantes_x = "y"

    muscle_part_on  : Liste contenant les numéros des parties à tracer
                  : active ou non de graph la variable totale du muscle ou la variable d'une des parties du muscle
                  : "allparts" toutes les parties on sans le total
                  : "all" toutes les parties avec le total

                  : Défault = False : trace la variable totale du muscle entier
                  : muscle_part_on = liste des numéros de la partie du muscle à tracer


    compare : = True si on veut comparer plusieurs données
              Ne rien mettre (compare = False par défaut) : on veut tracer qu'une seule donnée

    subplot = {"Dimension: [nrows, ncolumns]", :"number": Number_of_the_subplot_selected, "figsize": [horizontal_size_inches, vertical_size_inches], "last_subplot": True}


        subplot["dimension"] = [nrows, ncolumns]
    And defines the active axis as the subplot["number"]=number of the plot


    subplot["figsize"] : Optional argument to set the size of the figure
                         subplot["figsize"] = [horizontal_size_inches, vertical_size_inches]
                         : default : [14, 10] inches for 2D ; [7, 5] for [1,1] subplot

    subplot["dimension"] and figsize : are only to be set for subplot["number"] = 1
    They are not taken in account otherwise

    subplot["LastPart"] : bool = Optional argument : Controls if the legend and figure title are drawn
                        : It's automatically set to True if we reach the maximum subplot Number
                        : But it can be overwritten so that the legend is drawn even if one of the subplot is empty


    Example : Dimension = [2,2]
              the grah numbers are 1 2
                                   3 4

              Number = 3 corresponds to subplot [1,0]

            : To plot on a graph with 2 line and 3 columns on the graph in the center
            subplot = {"dimension":[3,3],"number":5}

    **kwargs : contient d'autres paramètres comme
             label : si jamais on veut ajouter un label à une donnée d'un graphique qui n'en aurait ou qui en aurait un autre
             add_graph = True : Si jamais on veut ajouter un autre graphique sur le dernier graphique tracé
                               : False par défaut, les nouvelles données seront tracées en effaçant les anciennes sur le subplot en cours
             legend_on : bool : argument contrôlant l'affichage de la légende
                       : True (par défaut) la légende s'affiche
                       : False La légende ne s'affiche pas'
            LegendLocation = dictionary, controls where the legend is drawn outside the figure

                          location string of matplotlib 'upper right', 'center left'...

                          Default value : lower center (below the figure
    """

    # get the customlabel if a label arguent is declared, puts None otherwise as a default value
    custom_label = kwargs.get("label", None)

    # get the legend_on argument that controls if the legend is drawn or not (Default True)
    legend_on = kwargs.get("legend_on", True)

    graph_type = "muscle_graph"

    # Stores the name of the x variable and y variable in the kwargs
    kwargs["variable_y"] = variable_y
    kwargs["variable_x"] = variable_x

    graph_annotation_on = kwargs.get("graph_annotation_on", False)

    # Name of the dictionnary key where the muscles are stored
    # By default it's muscles but in case of an edge muscle it is stored in GHReactions
    if "Edge muscle" in MuscleName:
        MuscleFolder = "GHReactions"
    else:
        MuscleFolder = "Muscles"

    # Initialise les informations sur les muscles parts si elle n'a pas été spécifiée (c'est à dire qu'il n'y a qu'une seule musclePart à dessiner)
    if MusclePartInformation is False:
        MusclePartInformation = {"LastPart": True,
                                 "Total Number Muscle Parts": 1}

    # Parcours toutes les parties de muscles à tracer

    # S'il n'y a qu'une composante à tracer
    if len(composante_y) == 1:

        # Prend la valeur de la composante comme elle est seule
        composante_y = composante_y[0]

        if compare is False:

            if cases_on is False:

                # Si plus d'une muscle part est tracée, on met une legende avec le nom de la musclepart
                if MusclePartInformation["Total Number Muscle Parts"] > 1:
                    label = MusclePart

                # Si seulement une muscle part est activée et qu'on ne compare pas, on ne met pas de légende
                else:
                    label = None

                plot_graph_functions(data, data[variable_x][composante_x], data[MuscleFolder][MuscleName]
                                     [MusclePart][variable_y][composante_y], graph_type, label=label, custom_label=custom_label, **kwargs)

            else:
                # On ne peut tracer qu'une seule donnée, donc on doit avoir soit un seul Case de sélectionné et n>=1 muscle parts
                # Ou on peut avoir plusieurs Case de sélectionnés mais une seule muscle part à tracer
                if len(cases_on) == 1 or MusclePartInformation["Total Number Muscle Parts"] == 1:

                    for Case in cases_on:

                        # La légende est le nom du case si il n'y a qu'une seule muscle part à tracer et plus d'un Case sélectionné
                        if len(cases_on) > 1 and MusclePartInformation["Total Number Muscle Parts"] == 1:
                            label = Case

                        # La légende est le nom de la muscle part s'il n'y a qu'un seul case et plusieurs Muscle part à tracer
                        elif len(cases_on) == 1 and MusclePartInformation["Total Number Muscle Parts"] > 1:
                            label = MusclePart

                        # Si les deux sont 1, on ne met pas de légende
                        else:
                            label = None

                        plot_graph_functions(data[Case], data[Case][variable_x][composante_x], data[Case][MuscleFolder][MuscleName]
                                             [MusclePart][variable_y][composante_y], graph_type, label=label, custom_label=custom_label, **kwargs)

        elif compare:

            # Si on a plusieurs simulations, on ne peut afficher qu'une seule donnée sur le graphique, donc qu'une seule muscle part
            if MusclePartInformation["Total Number Muscle Parts"] == 1:
                ListSimulations = list(data.keys())

                for Simulation in ListSimulations:
                    label = Simulation

                    if cases_on is False:
                        plot_graph_functions(data[Simulation], data[Simulation][variable_x][composante_x], data[Simulation][MuscleFolder]
                                             [MuscleName][MusclePart][variable_y][composante_y], graph_type, label=label, custom_label=custom_label, **kwargs)

                    # When we compare, we compare only one case between several simulations
                    elif len(cases_on) == 1:
                        plot_graph_functions(data[Simulation][cases_on[0]], data[Simulation][cases_on[0]][variable_x][composante_x], data[Simulation][cases_on[0]]
                                             [MuscleFolder][MuscleName][MusclePart][variable_y][composante_y], graph_type, label=label, custom_label=custom_label, **kwargs)

    # Si plusieurs composantes sont activées
    else:

        # Si on a plusieurs composantes, on ne peut afficher qu'une seule donnée sur le graphique, donc qu'une seule muscle part
        if MusclePartInformation["Total Number Muscle Parts"] == 1:

            # On ne peut comparer plusieurs simulations que si on active la même donnée, on ne peut pas afficher plusieurs composantes avec plusieurs simulations
            if compare is False:

                for Composante in composante_y:
                    label = Composante

                    if cases_on is False:
                        plot_graph_functions(data, data[variable_x][composante_x], data[MuscleFolder][MuscleName]
                                             [MusclePart][variable_y][Composante], graph_type, label=label, custom_label=custom_label, **kwargs)

                    # On peut tracer plusieurs composantes seulement si un seul cas de simulation est activé
                    elif len(cases_on) == 1:
                        plot_graph_functions(data, data[cases_on[0]][variable_x][composante_x], data[cases_on[0]][MuscleFolder]
                                             [MuscleName][MusclePart][variable_y][Composante], graph_type, label=label, custom_label=custom_label, **kwargs)

    # Si on trace la dernière muscle part, trace les axes, la légende, les titres etc...
    if MusclePartInformation["LastPart"]:

        # Axis Labels from the variable description
        if compare:
            if cases_on is False:
                plt.xlabel(data[ListSimulations[0]][variable_x]["Description"])
                plt.ylabel(data[ListSimulations[0]][MuscleFolder]
                           [MuscleName][MusclePart][variable_y]["Description"])
            else:
                plt.xlabel(data[ListSimulations[0]][cases_on[0]]
                           [variable_x]["Description"])
                plt.ylabel(data[ListSimulations[0]][cases_on[0]][MuscleFolder]
                           [MuscleName][MusclePart][variable_y]["Description"])
        elif compare is False:
            if cases_on is False:
                plt.xlabel(data[variable_x]["Description"])
                plt.ylabel(data[MuscleFolder][MuscleName]
                           [MusclePart][variable_y]["Description"])
            else:
                plt.xlabel(data[cases_on[0]][variable_x]["Description"])
                plt.ylabel(data[cases_on[0]][MuscleFolder][MuscleName]
                           [MusclePart][variable_y]["Description"])

        if subplot is None:
            plt.title(figure_title)

            # unsuperpose the annotations if activated
            if graph_annotation_on:

                # Calls the function that will move the annotations to avoid superposition
                unsuperpose_plot_annotations(**kwargs)

            # shows the legend if activated
            if legend_on:
                legend_setup(fig, graph_type, **kwargs)

            # Setups the grid and the axes ticks of the graph
            graph_grid_setup(fig, **kwargs)

        # Dans le cas d'un subplot
        else:

            # If a subplot title is entered, draws it (subplot_title isn't a bool)
            if not type(subplot_title) is bool:
                plt.title(subplot_title)

            # If a subplot title is entered, draws it (subplot_title isn't a bool)
            if not type(subplot_title) is bool:
                plt.title(subplot_title)

            # unsuperpose the annotations if activated
            if graph_annotation_on:

                # Calls the function that will move the annotations to avoid superposition
                unsuperpose_plot_annotations(**kwargs)

            # last_subplot can be entered in the subplot dictionary to oblige the legend to draw even if a subplot is empty
            # This statement has the priority over the test on the number of dimension
            if "last_subplot" in subplot:
                last_subplot = subplot["last_subplot"]
            # Tests if the number of subplot corresponds to the last subplot number to control if the legend and title are drawn or not
            elif subplot["number"] == subplot["dimension"][0] * subplot["dimension"][1]:
                last_subplot = True
            # Case where no legend and figure title will be drawn
            else:
                last_subplot = False

            # Setups the grid and the axes ticks of the graph
            graph_grid_setup(fig, last_subplot, **kwargs)

            # Displays the legend and figure title only if it's the last subplot drawn
            if last_subplot:
                # Trace le titre de la figure
                plt.suptitle(figure_title)

                # Ajuste les distances entre les subplots quand ils sont tous tracés
                plt.tight_layout()

                # shows the legend if activated
                if legend_on:

                    legend_setup(fig, graph_type, **kwargs)


def muscle_graph_old(data, MuscleName, variable_x, variable_y, figure_title, cases_on=False, compare=False, composante_x="Total", composante_y=["Total"], muscle_part_on=False, subplot=None, subplot_title=False, **kwargs):
    """
    Draws all the parts of a Muscle that were selected


    data : le dictionnaire contenant les data à tracer
         : Par défaut : Un dictionnaire ne contenant qu'une seule simulation
         : Soit un jeu de plusieurs datas (compare = True)

    variable_x : Le nom de la variable placée en x sur le graphique
    variable_y : le nom de la variable placée en y sur le graphique

    composante_y :
                  : type : liste de chaines de charactère
                  : Liste contenant les nom des composantes de la variable à tracer
                  : Par défaut : On trace la composante "Total" donc composante_y = ["Total"]

                : Activer plusieurs composantes :
                Exemple : composante_y = ["composante 1","composante 2","composante 3","Total"....]
                          Si on veut activer x et y entrer : composante_y = ["x","y"]

                : Activer une seule composante :
                Exemple : Si on veut activer y entrer : composante_y = ["y"]


                CAS PARTICULIER COMPOSANTES: Si on compare, on ne peut activer qu'une seule composante
                                           : Si on active plusieurs composantes, on doit comparer la même donnée (un seul cas de simulation)

    Composantes_x : Le nom de la composante de la variable en abscisse
                  : composante_x est une chaîne de charactère contenant le nom de la composante de la variable
                  : Par défaut : "Total"
                  : Si on veut activer y entrer : Composantes_x = "y"

    muscle_part_on  : Liste contenant les numéros des parties à tracer
                  : active ou non de graph la variable totale du muscle ou la variable d'une des parties du muscle
                  : "allparts" toutes les parties on sans le total
                  : "all" toutes les parties avec le total

                  : Défault = False : trace la variable totale du muscle entier
                  : muscle_part_on = numéro de la partie du muscle à tracer


    compare : = True si on veut comparer plusieurs données
              Ne rien mettre (compare = False par défaut) : on veut tracer qu'une seule donnée

    subplot = {"Dimension: [nrows, ncolumns]", :"number": Number_of_the_subplot_selected, "figsize": [horizontal_size_inches, vertical_size_inches], "last_subplot": True}


        subplot["dimension"] = [nrows, ncolumns]
    And defines the active axis as the subplot["number"]=number of the plot


    subplot["figsize"] : Optional argument to set the size of the figure
                         subplot["figsize"] = [horizontal_size_inches, vertical_size_inches]
                         : default : [14, 10] inches for 2D ; [7, 5] for [1,1] subplot

    subplot["dimension"] and figsize : are only to be set for subplot["number"] = 1
    They are not taken in account otherwise

    subplot["LastPart"] : bool = Optional argument : Controls if the legend and figure title are drawn
                        : It's automatically set to True if we reach the maximum subplot Number
                        : But it can be overwritten so that the legend is drawn even if one of the subplot is empty


    Example : Dimension = [2,2]
              the grah numbers are 1 2
                                   3 4

              Number = 3 corresponds to subplot [1,0]

            : To plot on a graph with 2 line and 3 columns on the graph in the center
            subplot = {"dimension":[3,3],"number":5}

    **kwargs : contient d'autres paramètres comme
             label : si jamais on veut ajouter un label à une donnée d'un graphique qui n'en aurait ou qui en aurait un autre
             add_graph = True : Si jamais on veut ajouter un autre graphique sur le dernier graphique tracé
                               : False par défaut, les nouvelles données seront tracées en effaçant les anciennes sur le subplot en cours
             legend_on : bool : argument contrôlant l'affichage de la légende
                       : True (par défaut) la légende s'affiche
                       : False La légende ne s'affiche pas'
             LegendLocation = dictionary, controls where the legend is drawn outside the figure

                           location string of matplotlib 'upper right', 'center left'...

                           Default value : lower center (below the figure)

    """

    # Get add_graph function. Puts it to false by default if it's not declared in the kwargs
    add_graph = kwargs.get("add_graph", False)

    # Verifications for when simulationCases are used
    if cases_on:
        # Active tous les cas présents dans data
        if cases_on == "all":
            cases_on = list(data.keys())

        # Vérifie que Cases est toujours une liste si 'all' n'est pas utilisé
        elif not type(cases_on) is list:
            raise ValueError(
                "cases_on doit être une liste si 'all' n'est pas utilisé")
            return

        # Vérifie qu'on n'active pas plusieurs cas tout en comparant
        if len(cases_on) > 1 and compare:
            raise ValueError(
                "On ne peut pas comparer plusieurs simulations et plusieurs cas en même temps")
            return

        # Vérifie qu'on ne dessine pas plusieurs variables tout en dessinant plusieurs cas
        if len(cases_on) > 1 and len(composante_y) > 1:
            raise ValueError(
                "On ne peut pas dessiner plusieurs cas et plusieurs composantes en même temps")
            return

    # Name of the dictionnary key where the muscles are stored
    # By default it's muscles but in case of an edge muscle it is stored in GHReactions
    if "Edge Muscle" in MuscleName:
        MuscleFolder = "GHReactions"
    else:
        MuscleFolder = "Muscles"

    # Construit la liste des parties de muscle à tracer

    # Sans cas de simulation selon le cas (avec/sans des cas, avec/sans comparaison)
    if cases_on is False:
        if compare is False:
            MuscleParts = list(data[MuscleFolder][MuscleName].keys())
        else:
            ListSimulations = list(data.keys())
            MuscleParts = list(data[ListSimulations[0]]
                               [MuscleFolder][MuscleName].keys())

    # Dans les cas où on a des cas de simulation
    else:
        if compare is False:
            MuscleParts = list(
                data[cases_on[0]][MuscleFolder][MuscleName].keys())
        else:
            ListSimulations = list(data.keys())
            MuscleParts = list(
                data[ListSimulations[0]][cases_on[0]][MuscleFolder][MuscleName].keys())

    # Si toutes les parties sont activées, fais une liste avec le nom de toutes les parties sauf le muscle total
    # n'enlève pas la partie totale si le muscle n'a pas de partie
    if muscle_part_on == "allparts" and (not len(MuscleParts) == 1):

        # Enlève le muscle total de la liste
        MuscleParts.remove(MuscleName)

    # Dans le cas où on a entré une liste des numéros des parties
    elif isinstance(muscle_part_on, list):

        # Recrée les noms des parties à tracer en parcourant les numéros entrés
        ListMuscleParts = [
            f"{MuscleName} {MusclePartNumber}" for MusclePartNumber in muscle_part_on]

        # Stores the new value of muscleparts to draw
        MuscleParts = ListMuscleParts

    # Si on ne veut tracer qu'un seul muscle
    elif muscle_part_on is False:
        MuscleParts = [MuscleName]

    # Vérification qu'on ne dessine pas plusieurs parties de muscles tout en comparant
    if compare and len(MuscleParts) > 1:
        raise ValueError(
            "On ne peut pas comparer plusieurs simulations et dessiner plusieurs parties de muscles en même temps")
        return

    # Vérification qu'on ne dessine pas plusieurs parties de muscles tout en dessinant plusieurs composantes
    if compare and len(composante_y) > 1:
        raise ValueError(
            "On ne peut pas dessiner plusieurs composantes et dessiner plusieurs parties de muscles en même temps")
        return
    # Vérifie qu'on ne dessine pas plusieurs parties de muscles tout comparant plusieurs cas de simulation
    if type(cases_on) is list:
        if len(cases_on) > 1 and len(MuscleParts) > 1:
            raise ValueError(
                "On ne peut pas dessiner plusieurs cas de simulation et dessiner plusieurs parties de muscles en même temps")
            return

    # Gets the figure size
    figsize = kwargs.get("figsize", None)

    # subplot is setup here to be able to draw every part of a muscle on the same figure
    fig = subplot_setup(subplot, figsize, add_graph)

    # Initialisation du dictionnaire contenant les informations sur le nombre de muscles parts qui seront tracées
    MusclePartInformation = {}

    # Nombre de muscle parts qui seront tracées sur le même graphique
    MusclePartInformation["Total Number Muscle Parts"] = len(MuscleParts)

    # Parcours les parties de muscles à tracer
    for MusclePart in MuscleParts:

        # Si on trace la dernière partie de muscle pour ne tracer la légende et les axes qu'à ce moment là
        if MusclePart == MuscleParts[-1]:
            MusclePartInformation["LastPart"] = True
        else:
            MusclePartInformation["LastPart"] = False

        muscle_part_graph(data, MuscleName, MusclePart, variable_x, variable_y, figure_title, composante_x,
                          composante_y, compare, subplot, subplot_title, cases_on, MusclePartInformation, fig=fig, **kwargs)


# %% CODE POUR TESTER TOUTES LES COMBINAISONS ET LES MESSAGES D'ERREUR

# import Anybody_LoadOutput as LoadOutput

# COP_contour = define_COP_contour("COP_contour")

# # Imports the module to import .h5 and AnyFileOut

# SaveDatadir = 'SaveData/Variation_CSA/'

# SimulationCases = ["Case 1", "Case 2", "Case 3", "Case 4", "Case 5"]

# CasesFilesList1 = ['Results-31-07-case1-droit-GlenoidAxisTilt-EnFace-CustomForce',
#                     'Results-31-07-case2-droit-GlenoidAxisTilt-EnFace-CustomForce',
#                     'Results-31-07-case3-droit-GlenoidAxisTilt-EnFace-CustomForce',
#                     'Results-31-07-case4-droit-GlenoidAxisTilt-EnFace-CustomForce',
#                     'Results-31-07-case5-droit-GlenoidAxisTilt-EnFace-CustomForce']


# CasesFilesList2 = [
#     'Results-17-07-case1-antero-Droit-EnFace21',
#     'Results-17-07-case2-antero-Droit-EnFace21',
#     'Results-17-07-case3-antero-Droit-EnFace21',
#     '',
#     'Results-17-07-case5-antero-EnFace21'
# ]


# CasesOffcompareOff = LoadOutput.LoadResultsh5(SaveDatadir, 'Results-31-07-case5-droit-GlenoidAxisTilt-EnFace-CustomForce', AddConstants=True)

# cases_oncompareOff = LoadOutput.LoadSimulationCases(SaveDatadir, CasesFilesList1, SimulationCases, AddConstants=True)

# cases_oncompareOn = LoadOutput.LoadSimulations(SaveDatadir, [CasesFilesList1, CasesFilesList2], ["simulation 1", "simulation 2"], AddConstants=True, SimulationCases=SimulationCases)

# CasesOffcompareOn = LoadOutput.LoadSimulations(SaveDatadir, CasesFilesList1, ["simulation 1", "simulation 2", "simulation 3", "simulation 4", "simulation 5"], AddConstants=True)


# %%                                 TEST GRAPH
# %% PAS DE CAS DE SIMULATION

# # compare = False
# graph(CasesOffcompareOff, "Abduction", "ForceContact", "Force contact cases_on = False compare = False")
# graph(CasesOffcompareOff, "Abduction", "ForceContact", "Force contact cases_on = False compare = False en x", composante_y=["x"])
# graph(CasesOffcompareOff, "Abduction", "ForceContact", "Force contact cases_on = False compare = False plusieurs composantes", composante_y=["x", "y", "z", "Total"])


# # compare = True

# graph(CasesOffcompareOn, "Abduction", "ForceContact", "Force contact cases_on = False compare = True", compare=True)
# graph(CasesOffcompareOn, "Abduction", "ForceContact", "Force contact cases_on = False compare = True en x", composante_y=["x"], compare=True)

# # Message d'erreur fait
# graph(CasesOffcompareOn, "Abduction", "ForceContact", "Force contact cases_on = False compare = True plusieurs composantes", composante_y=["x", "y", "z", "Total"], compare=True)

# %% CAS DE SIMULATION

# # compare = False

# # 1 Case
# graph(cases_oncompareOff, "Abduction", "ForceContact", "Force contact cases_on = True compare = False Un case", cases_on=["Case 1"])
# graph(cases_oncompareOff, "Abduction", "ForceContact", "Force contact cases_on = True compare = False en x Un case", composante_y=["x"], cases_on=["Case 1"])
# graph(cases_oncompareOff, "Abduction", "ForceContact", "Force contact cases_on = True compare = False plusieurs composantes Un case", composante_y=["x", "y", "z", "Total"], cases_on=["Case 1"])

# # Plusieurs cases mais une seule variable_y
# graph(cases_oncompareOff, "Abduction", "ForceContact", "Force contact cases_on = True compare = False plusieurs cases", cases_on=["Case 1", "Case 2"])
# graph(cases_oncompareOff, "Abduction", "ForceContact", "Force contact cases_on = True compare = False en x plusieurs cases", composante_y=["x"], cases_on=["Case 1", "Case 2"])

# # Message d'erreur fait
# graph(cases_oncompareOff, "Abduction", "ForceContact", "Force contact cases_on = True compare = False plusieurs composantes plusieurs cases", composante_y=["x", "y", "z", "Total"], cases_on=["Case 1", "Case 2"])


# # All cases mais une seule variable_y
# graph(cases_oncompareOff, "Abduction", "ForceContact", "Force contact cases_on = True compare = False cases all", cases_on='all')
# graph(cases_oncompareOff, "Abduction", "ForceContact", "Force contact cases_on = True compare = False en x cases all", composante_y=["x"], cases_on='all')

# # Message d'erreur fait
# graph(cases_oncompareOff, "Abduction", "ForceContact", "Force contact cases_on = True compare = False plusieurs composantes cases all", composante_y=["x", "y", "z", "Total"], cases_on='all')


# # compare = True

# # 1 Case
# graph(cases_oncompareOn, "Abduction", "ForceContact", "Force contact cases_on = True compare = True Un case", cases_on=["Case 1"], compare=True)
# graph(cases_oncompareOn, "Abduction", "ForceContact", "Force contact cases_on = True compare = True en x Un case", composante_y=["x"], cases_on=["Case 1"], compare=True)

# # Message d'erreur fait
# graph(cases_oncompareOn, "Abduction", "ForceContact", "Force contact cases_on = True compare = True plusieurs composantes Un case", composante_y=["x", "y", "z", "Total"], cases_on=["Case 1"], compare=True)


# # Plusieurs cases mais une seule variable_y
# # Message d'erreut fait
# graph(cases_oncompareOn, "Abduction", "ForceContact", "Force contact cases_on = True compare = True plusieurs cases", cases_on=["Case 1", "Case 2"], compare=True)
# graph(cases_oncompareOn, "Abduction", "ForceContact", "Force contact cases_on = True compare = True en x plusieurs cases", composante_y=["x"], cases_on=["Case 1", "Case 2"], compare=True)
# graph(cases_oncompareOn, "Abduction", "ForceContact", "Force contact cases_on = True compare = True plusieurs composantes plusieurs cases", composante_y=["x", "y", "z", "Total"], cases_on=["Case 1", "Case 2"], compare=True)


# # All cases mais une seule variable_y
# # Message d'erreut fait
# graph(cases_oncompareOn, "Abduction", "ForceContact", "Force contact cases_on = True compare = True cases all", cases_on='all', compare=True)
# graph(cases_oncompareOn, "Abduction", "ForceContact", "Force contact cases_on = True compare = True en x cases all", composante_y=["x"], cases_on='all', compare=True)
# graph(cases_oncompareOn, "Abduction", "ForceContact", "Force contact cases_on = True compare = True plusieurs composantes cases all", composante_y=["x", "y", "z", "Total"], cases_on='all', compare=True)


# %%                                 TEST muscle_graph
# %% PAS DE CAS DE SIMULATION

# # Juste une partie ou le total
# muscle_graph(CasesOffcompareOff, "deltoideus lateral", "Abduction", "Fm", "Force muscle cases_on = False compare = False")
# muscle_graph(CasesOffcompareOff, "deltoideus lateral", "Abduction", "Fm", "Force muscle cases_on = False compare = False", muscle_part_on=[2])

# # Plusieurs part
# muscle_graph(CasesOffcompareOff, "deltoideus lateral", "Abduction", "Fm", "Force muscle cases_on = False compare = False", muscle_part_on=[1, 2])
# muscle_graph(CasesOffcompareOff, "deltoideus lateral", "Abduction", "Fm", "Force muscle cases_on = False compare = False", muscle_part_on="all")
# muscle_graph(CasesOffcompareOff, "deltoideus lateral", "Abduction", "Fm", "Force muscle cases_on = False compare = False", muscle_part_on="allparts")


# # compare True
# # Juste une partie ou le total
# muscle_graph(CasesOffcompareOn, "deltoideus lateral", "Abduction", "Fm", "Force muscle cases_on = False compare = True", compare=True)
# muscle_graph(CasesOffcompareOn, "deltoideus lateral", "Abduction", "Fm", "Force muscle cases_on = False compare = False", muscle_part_on=[2], compare=True)

# # Plusieurs part
# # Message d'erreur fait
# muscle_graph(CasesOffcompareOn, "deltoideus lateral", "Abduction", "Fm", "Force muscle cases_on = False compare = True", muscle_part_on=[1, 2], compare=True)
# muscle_graph(CasesOffcompareOn, "deltoideus lateral", "Abduction", "Fm", "Force muscle cases_on = False compare = True", muscle_part_on="all", compare=True)
# muscle_graph(CasesOffcompareOn, "deltoideus lateral", "Abduction", "Fm", "Force muscle cases_on = False compare = True", muscle_part_on="allparts", compare=True)


# %% CAS DE SIMULATION


# # compare = False

# # 1 Case
# # Juste une partie ou le total
# muscle_graph(cases_oncompareOff, "deltoideus lateral", "Abduction", "Fm", "Force muscle cases_on = True compare = False", cases_on=["Case 1"])
# muscle_graph(cases_oncompareOff, "deltoideus lateral", "Abduction", "Fm", "Force muscle cases_on = True compare = False", muscle_part_on=[2], cases_on=["Case 1"])

# # Plusieurs part 1 case
# muscle_graph(cases_oncompareOff, "deltoideus lateral", "Abduction", "Fm", "Force muscle cases_on = True compare = False", muscle_part_on=[1, 2], cases_on=["Case 1"])
# muscle_graph(cases_oncompareOff, "deltoideus lateral", "Abduction", "Fm", "Force muscle cases_on = True compare = False", muscle_part_on="all", cases_on=["Case 1"])
# muscle_graph(cases_oncompareOff, "deltoideus lateral", "Abduction", "Fm", "Force muscle cases_on = True compare = False", muscle_part_on="allparts", cases_on=["Case 1"])


# # plusieurs Cases
# # Juste une partie ou le total
# muscle_graph(cases_oncompareOff, "deltoideus lateral", "Abduction", "Fm", "Force muscle cases_on = True compare = False", cases_on=["Case 1", "Case 2"])
# muscle_graph(cases_oncompareOff, "deltoideus lateral", "Abduction", "Fm", "Force muscle cases_on = True compare = False", muscle_part_on=[2], cases_on=["Case 1", "Case 2"])
# muscle_graph(cases_oncompareOff, "deltoideus lateral", "Abduction", "Fm", "Force muscle cases_on = True compare = False", cases_on="all")
# muscle_graph(cases_oncompareOff, "deltoideus lateral", "Abduction", "Fm", "Force muscle cases_on = True compare = False", muscle_part_on=[2], cases_on="all")


# # Plusieurs part, plusieurs case
# # Message d'erreur fait
# muscle_graph(cases_oncompareOff, "deltoideus lateral", "Abduction", "Fm", "Force muscle cases_on = True compare = False", muscle_part_on=[1, 2], cases_on="all")
# muscle_graph(cases_oncompareOff, "deltoideus lateral", "Abduction", "Fm", "Force muscle cases_on = True compare = False", muscle_part_on="all", cases_on="all")
# muscle_graph(cases_oncompareOff, "deltoideus lateral", "Abduction", "Fm", "Force muscle cases_on = True compare = False", muscle_part_on="allparts", cases_on="all")


# # compare True
# # 1 Case
# # Juste une partie ou le total
# muscle_graph(cases_oncompareOn, "deltoideus lateral", "Abduction", "Fm", "Force muscle cases_on = True compare = True", compare=True, cases_on=["Case 1"])
# muscle_graph(cases_oncompareOn, "deltoideus lateral", "Abduction", "Fm", "Force muscle cases_on = True compare = True", muscle_part_on=[2], compare=True, cases_on=["Case 1"])

# # Plusieurs part
# # Message d'erreur fait
# muscle_graph(cases_oncompareOn, "deltoideus lateral", "Abduction", "Fm", "Force muscle cases_on = True compare = True", muscle_part_on=[1, 2], compare=True, cases_on=["Case 1"])
# muscle_graph(cases_oncompareOn, "deltoideus lateral", "Abduction", "Fm", "Force muscle cases_on = True compare = True", muscle_part_on="all", compare=True, cases_on=["Case 1"])
# muscle_graph(cases_oncompareOn, "deltoideus lateral", "Abduction", "Fm", "Force muscle cases_on = True compare = True", muscle_part_on="allparts", compare=True, cases_on=["Case 1"])


# # Plusieurs Case
# # Juste une partie ou le total
# # Message d'erreut fait
# muscle_graph(cases_oncompareOn, "deltoideus lateral", "Abduction", "Fm", "Force muscle cases_on = True compare = True", compare=True, cases_on=["Case 1", "Case 2"])
# muscle_graph(cases_oncompareOn, "deltoideus lateral", "Abduction", "Fm", "Force muscle cases_on = True compare = True", muscle_part_on=[2], compare=True, cases_on=["Case 1", "Case 2"])

# # Plusieurs part un cas
# # Message d'erreur fait
# muscle_graph(cases_oncompareOn, "deltoideus lateral", "Abduction", "Fm", "Force muscle cases_on = True compare = True", muscle_part_on=[1, 2], compare=True, cases_on=["Case 1", "Case 2"])
# muscle_graph(cases_oncompareOn, "deltoideus lateral", "Abduction", "Fm", "Force muscle cases_on = True compare = True", muscle_part_on="all", compare=True, cases_on=["Case 1", "Case 2"])
# muscle_graph(cases_oncompareOn, "deltoideus lateral", "Abduction", "Fm", "Force muscle cases_on = True compare = True", muscle_part_on="allparts", compare=True, cases_on=["Case 1", "Case 2"])

# %%                                 TEST COP_graph
# # Pas de cas de simulation

# # compare = False
# COP_graph(CasesOffcompareOff, "COP_graph Cases False compare False", COP_contour)
# COP_graph(CasesOffcompareOn, "COP_graph Cases False compare True", COP_contour, compare=True)

# # Cas de simulation
# COP_graph(cases_oncompareOff, "COP_graph Cases False compare False", COP_contour, cases_on="all")

# # Message d'erreur fait
# COP_graph(cases_oncompareOn, "COP_graph Cases False compare True", COP_contour, compare=True, cases_on="all")
# COP_graph(cases_oncompareOn, "COP_graph Cases False compare True", COP_contour, compare=True, cases_on="Case 1")
