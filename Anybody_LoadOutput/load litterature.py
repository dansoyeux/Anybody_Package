# -*- coding: utf-8 -*-
"""
Created on Mon Nov 13 11:03:46 2023

@author: Dan
"""

from scipy.interpolate import CubicSpline
from Anybody_Package.Anybody_LoadOutput.Tools import array_to_dictionary


import numpy as np
import pandas as pd


def seperate_data_frame_by_category(data_frame):
    """
    seperates a pandas data_frame by categories

    the first line of the entered dataframe has a category name or nan values.
    groups the data from the column where there is a category name to the column before the next category name.


    returns a dictionay where each key is the name of that category


    ex : col numbers :    1       2    3    4    5       6    7      8      9    10
         line 1 :       "cat_1", NaN, NaN, NaN "cat_2", NaN, NaN, "cat_3", NaN, NaN

        it will return a dictionary named seperated_categories_dictionary

        seperated_categories_dictionary["cat_1"] : column 1 to 4
        seperated_categories_dictionary["cat_2"] : column 5 to 7
        seperated_categories_dictionary["cat_3"] : column 8 to 10
        """
    # Selects the line that stores the category names
    category_line = data_frame.iloc[0, :]

    # gets the indexes of each column
    category_line_index = category_line.index.to_numpy()

    # list containing the index of the columns that have a category name (not NaN)
    category_first_value = category_line_index[category_line.notna().to_numpy()]

    # list of all categories
    category_name_list = category_line[category_line.notna().to_numpy()].tolist()

    seperated_categories_dictionary = {}

    # Gets the data for each category
    for category_index, category_name in enumerate(category_name_list):
        # select the data for each author
        # for every authors except the last
        if not category_name == category_name_list[-1]:
            current_category_data = data_frame.loc[:, category_first_value[category_index]: category_first_value[category_index + 1] - 1]
        # for the last author
        else:
            current_category_data = data_frame.loc[:, category_first_value[category_index]: category_line_index[-1]]

        # deletes the row that has author names
        current_category_data = current_category_data.iloc[1: len(current_category_data), :]

        seperated_categories_dictionary[category_name] = current_category_data

    return seperated_categories_dictionary


def variable_data_frame_to_dictionary(variable_data, x_variable_interpolate_value, variable_informations):
    """
    function that takes the variables and returs it as a dictionary

    variable_data : line 0 : Variable names
                    line 1 : component names
                    lines 2:end : values
    """

    variable_names_list = list(variable_informations.keys())
    variable_x_name = variable_names_list[0]
    variable_y_name = variable_names_list[1]

    # line with variables
    variable_line = variable_data.iloc[0, :].tolist()

    # resets the y variable component sequence for this author
    variable_y_composante_sequence = []
    variable_component_index = 0

    # number of component for the y variable
    n_composantes = len([i for i in variable_line if i == variable_x_name])

    # Creates an array that will store every y_interpolated_component_value
    y_variable_array = np.zeros([n_interpolate_points, n_composantes])

    # Goes through every number of columns for this author
    for variable_column_index in range(0, len(variable_line)):
        current_column_variable = variable_data.iloc[0, variable_column_index]

        # if the current column is the x variable
        if current_column_variable == variable_x_name:
            x_variable_array = variable_data.iloc[2:len(variable_data), variable_column_index].dropna().to_numpy(dtype=float)

            x_variable_component = [variable_data.iloc[1, variable_column_index]]

        # for the y variable and interpolates it using the x variable
        elif current_column_variable == variable_y_name:

            # stores the current component name
            variable_y_composante_sequence.append(variable_data.iloc[1, variable_column_index])

            # stores the current component value
            y_variable_component_array = variable_data.iloc[2:len(variable_data), variable_column_index].dropna().to_numpy(dtype=float)

            # Interpolates the y variable using min and max x variable entered so that every y component correspond to the same x values
            Interpolation_Function = CubicSpline(x_variable_array, y_variable_component_array)

            # gets the interpolated y value
            interpolated_y_component_value = Interpolation_Function(x_variable_interpolate_value)

            # stores this value in the y variable array
            y_variable_array[:, int(variable_component_index)] = interpolated_y_component_value

            # increases the component index
            variable_component_index += 1

    loaded_variables = variable_informations
    loaded_variables[variable_x_name]["SequenceComposantes"] = x_variable_component
    loaded_variables[variable_y_name]["SequenceComposantes"] = variable_y_composante_sequence

    # Transforms the obtained arrays to a dictionary
    variable_x_dictionary = array_to_dictionary(x_variable_interpolate_value, **loaded_variables[variable_x_name])
    variable_y_dictionary = array_to_dictionary(y_variable_array, **loaded_variables[variable_y_name])

    return variable_x_dictionary, variable_y_dictionary, loaded_variables


def get_excel_sheet_variable_informations(variable_informations_data):
    """
    function that puts in a dictionary every information about the x and y variables from the current sheet variable informations data

    variable descriptions and multiply factors

    variable x min and max for interpolation


    return
        variable_informations = {variable_x_name: {"VariableDescription": variable_x_description, "MultiplyFactor": variable_x_multiply_factor},
                                 variable_y_name: {"VariableDescription": variable_y_description, "MultiplyFactor": variable_y_multiply_factor}
                                 }

        interpolation_informations= {"n_interpolate_points": n_interpolate_points, "min_x": variable_x_min, "max_x": variable_x_max}


    """
    x_variable_index = int(variable_informations_data[variable_informations_data.loc[:, 0] == "Variable x"].index.values[0])
    y_variable_index = int(variable_informations_data[variable_informations_data.loc[:, 0] == "Variable y"].index.values[0])

    x_variable_informations = variable_informations_data.loc[x_variable_index:y_variable_index - 1, 1]
    y_variable_informations = variable_informations_data.loc[y_variable_index:len(variable_informations_data), 1].dropna()

    variable_x_name = x_variable_informations.iloc[0]
    variable_x_description = x_variable_informations.iloc[1]
    variable_x_multiply_factor = x_variable_informations.iloc[2]
    variable_x_min = x_variable_informations.iloc[3]
    variable_x_max = x_variable_informations.iloc[4]
    n_interpolate_points = x_variable_informations.iloc[5]

    variable_y_name = y_variable_informations.iloc[0]
    variable_y_description = y_variable_informations.iloc[1]
    variable_y_multiply_factor = y_variable_informations.iloc[2]

    variable_informations = {variable_x_name: {"VariableDescription": variable_x_description, "MultiplyFactor": variable_x_multiply_factor},
                             variable_y_name: {"VariableDescription": variable_y_description, "MultiplyFactor": variable_y_multiply_factor}
                             }

    interpolation_informations = {"n_interpolate_points": n_interpolate_points, "min_x": variable_x_min, "max_x": variable_x_max}

    return variable_informations, interpolation_informations, variable_x_name, variable_y_name


"""
------------------------------------------------------------------------------------------
------------------------------------------------------------------------------------------
------------------------------------------------------------------------------------------
------------------------------------------------------------------------------------------
ajouter aussi une entrée LoadedVariables
------------------------------------------------------------------------------------------
------------------------------------------------------------------------------------------
------------------------------------------------------------------------------------------
"""


"""
Arguments
"""
file_name = "Anybody_Package/Template/Template_importation_littérature"


# nombre de points interpolation (100 par défaut)
n_interpolate_points = 100

# valeur par défaut
extension = "xlsx"

"""
temporary values
"""
#
# current_sheet_name = "Translation"
current_sheet_name = "Muscle.Activity"


"""
script
"""

"""
variables à prendre dans varible infos
"""
# min et max de la variable en x. sera utilisé pour interpolation
# est stockée dans variable informations
# min_x = 0
# max_x = 165

# variable_x_name = "Abduction"
# y_variable_name = "Translation"
# y_variable_name = "Activity"

# x_variable_description = "Angle d'abduction [°]"
# y_variable_description = "Translation de la tête humérale [mm]"

# x_multiply_factor = 1
# y_multiply_factor = 1


ExcelFile = pd.ExcelFile(f"{file_name}.{extension}")


def get_sheet_variables(ExcelFile, current_sheet_name):

    data = pd.read_excel(ExcelFile, current_sheet_name, header=None)

    # gets the variables informations
    variable_informations_data = data.iloc[:, 0: 2]

    variable_informations, interpolation_informations, variable_x_name, variable_y_name = get_excel_sheet_variable_informations(variable_informations_data)

    x_variable_interpolate_value = np.linspace(interpolation_informations["min_x"], interpolation_informations["max_x"], interpolation_informations["n_interpolate_points"])

    variables_data = data.iloc[:, 2:len(data)]

    result_dictionary = {}

    # seperates the dataframe by author names
    author_data = seperate_data_frame_by_category(variables_data)

    # goes through
    for author_name, current_author_data in author_data.items():

        # in case of a muscle variable, seperates the data by muscle name
        if "Muscle" in current_sheet_name:
            # seperates the data by variables
            muscle_author_data = seperate_data_frame_by_category(current_author_data)

            # initializes the result dictionary with a "Muscles" directory to store muscle variables
            result_dictionary[author_name] = {"Muscles": {}}

            # goes through each muscle data for the current author
            for muscle_name, current_muscle_data in muscle_author_data.items():
                variable_x_dictionary, variable_y_dictionary, loaded_variables = variable_data_frame_to_dictionary(current_muscle_data, x_variable_interpolate_value, variable_informations)

                # Puts the result
                result_dictionary[author_name]["Muscles"][muscle_name] = {muscle_name: {variable_y_name: variable_y_dictionary}}

            result_dictionary[author_name][variable_x_name] = variable_x_dictionary

        else:
            # transforms the variables data into a result dictionary
            variable_x_dictionary, variable_y_dictionary, loaded_variables = variable_data_frame_to_dictionary(current_author_data, x_variable_interpolate_value, variable_informations)

            # for a normal variable
            result_dictionary[author_name] = {variable_x_name: variable_x_dictionary, variable_y_name: variable_y_dictionary}

        result_dictionary[author_name]["Loaded Variables"] = loaded_variables

    return result_dictionary


a = get_sheet_variables(ExcelFile, current_sheet_name)


ExcelFile.close()
