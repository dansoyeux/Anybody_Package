import numpy as np

def get_result_dictionary_data_structure(result_dictionary):
    """
    returns a deepnest counter that indicates the data structure of the result dictionary entered
    and indicates if the source of the data is Anybody or from the literature
    --------------------------------------------------------------------------
    return
    variables_deepness_counter : (int) counter that counts how deep the variables are stored which indicates the data structure
    0 : no simulation cases
    1 : simulation cases
    2 : compared simulation cases
    3+ : error

    data_source : (str) The data source ("Anybody" or "Literature")
    """

    # counter that counts how deep the variables are stored which indicates the data structure
    # 0 : no simulation cases
    # 1 : simulation cases
    # 2 : compared simulation cases
    # 3 : error
    variables_deepness_counter = 0

    # searches for the entry "Loaded Variables" to know the data structure
    while "Loaded Variables" not in list(result_dictionary.keys()) and variables_deepness_counter < 3:
        # increases the ccuonter
        variables_deepness_counter += 1

        # goes one step deeper in the result dictionary
        result_dictionary = result_dictionary[list(result_dictionary.keys())[0]]

    if variables_deepness_counter > 2:
        raise ValueError("The result dictionary used doesn't have a correct data structure. The variables are {variables_deepness_counter} levels deep while 2 is the maximum!")

    # Gets the source of the data (anybody or the literature)
    data_source = result_dictionary["Loaded Variables"]["Data Source"]

    return variables_deepness_counter, data_source


def get_result_dictionary_variables_informations(result_dictionary):
    """
    Returns the names of the variables and muscles variables, their component sequence and description

    ----------------------------------------------
    return
        muscle_variables_informations = {"muscle_variable_1_name": {"SequenceComposantes": [......], "Description": "DESCRIPTION"}}
        variables_informations = {"variable_1_name": {"SequenceComposantes": [......], "Description": "DESCRIPTION"}}
    """

    variables_informations = {}
    muscle_variables_informations = {}

    # gets the result dictionary structure
    variables_deepness_counter, data_source = get_result_dictionary_data_structure(result_dictionary)

    if variables_deepness_counter == 0:
        case_result_dictionary = result_dictionary

    # Simulation cases
    elif variables_deepness_counter == 1:
        first_case = list(result_dictionary.keys())[0]
        case_result_dictionary = result_dictionary[first_case]

    # Compared simulation cases
    elif variables_deepness_counter == 2:
        first_simulation = list(result_dictionary.keys())[0]
        first_case = list(result_dictionary[first_simulation].keys())[0]
        case_result_dictionary = result_dictionary[first_simulation][first_case]

    # List of entries normal variables
    variables_list = list(case_result_dictionary.keys())

    # Removes entries that are not variables
    if "Muscles" in variables_list:
        variables_list.remove("Muscles")
    if "Loaded Variables" in variables_list:
        variables_list.remove("Loaded Variables")
    if "Model informations" in variables_list:
        variables_list.remove("Model informations")

    # gets the variables description and component sequence
    for variable in variables_list:
        variables_informations[variable] = {"SequenceComposantes": case_result_dictionary[variable]["SequenceComposantes"],
                                            "Description": case_result_dictionary[variable]["Description"]}

    first_muscle = list(case_result_dictionary["Muscles"].keys())[0]

    muscle_variables_list = list(case_result_dictionary["Muscles"][first_muscle][first_muscle].keys())

    # gets the variables description and component sequence
    for muscle_variable in muscle_variables_list:
        muscle_variables_informations[muscle_variable] = {"SequenceComposantes": case_result_dictionary["Muscles"][first_muscle][first_muscle][muscle_variable]["SequenceComposantes"],
                                                          "Description": case_result_dictionary["Muscles"][first_muscle][first_muscle][muscle_variable]["Description"]}

    return variables_informations, muscle_variables_informations


def transform_vector(vector, rotation_matrix, translation_vect=None, inverse_transform=False):
    """
    Transforme un vecteur avec une matrice de rotation pour chaque pas de temps et le bouge d'un certain vecteur pour chaque pas de temps

    rotation_matrix est la matrice de rotation à chaque pas de temps (nstep,3,3)

    translation_vect : Vecteur de translation colonne à chaque pas de temps (nstep,1)

    Le vecteur à transformer à chaque pas de temps est une matrice (nstep,3)

    inverse_transform : False : Transformation directe : RotMat * vector + TranslationVector
                       True : si on veut faire une transformation inverse (vector = vector * (RotMat - TranslationVector))

    """

    nstep = vector.shape[0]
    transformed_vector = np.zeros([nstep, 3])

    # Dans le cas où on veut juste faire une rotation
    if translation_vect is None:
        # Transformation directe
        if inverse_transform is False:
            for step in range(nstep):
                transformed_vector[step, :] = np.dot(rotation_matrix[step, :, :], vector[step, :])

        # Transformation inverse
        else:
            for step in range(nstep):
                transformed_vector[step, :] = np.dot(vector[step, :], rotation_matrix[step, :, :])

    # Rotation and translation
    else:
        # Transformation directe
        if inverse_transform is False:
            for step in range(nstep):
                transformed_vector[step, :] = np.dot(rotation_matrix[step, :, :], vector[step, :]) + translation_vect[step, :]

        # Transformation inverse
        else:
            for step in range(nstep):
                transformed_vector[step, :] = np.dot(vector[step, :] - translation_vect[step, :], rotation_matrix[step, :, :])

    return transformed_vector


def array_to_dictionary(Array, VariableDescription='', SequenceComposantes='', MultiplyFactor=1, Composantes_Inverse_Direction=False, first_value=False, offset=False, vect_dir=False, total_on=True, **kwargs):
    """
    Met en forme un array 2D (nstep,ndim) sous la forme d'un dictionnaire :
        "Description" : Description qui sera utilisée par les graphiques

        SequenceComposantes : sépare ensuite les composantes dans un dictionnaire selon la séquence précisée (forme de liste ["composante 1","Composante 2"...])
        Par défaut : La séquence par défaut est ['x','y','z']
                   : WARNING : par défaut, les vecteurs sont séparés en 3 composantes maximum

        "Total" : Calcule la valeur totale de la variable à chaque step de simulation
                : Cette valeur n'est pas calculée si la séquence de composante contient la valeur "Total"

        total_on : (bool) activates the calculation of the total

        Composantes_Inverse_Direction : list : example SequenceComposantes = ['x','y','z'], if you want to revert the direction of 'y' only
                                      : Composantes_Inverse_Direction = [False, True, True]
                                      Default (inversion not activated) False

       offset : list, values are floats
              : Offsets each components of the array by a value
              : The list contains the offset for each component of the Array (in the sequence order)
              : [offset_x, offset_y, offset_z] for SequenceComposantes = ['x','y','z']
              : The offset must be in the units wanted after applying the multiplyfactor
                  (if we want the offset to be 1 millimiter for a measure that was first in meters then converted in millimeter, the offset to enter is 1 millimiter)
              : the dimension of this list is equal to the number of components this Array has.
              : only total : len(first_value) = 1
              : one component : len(first_value) = 1
              : n components : len(first_value) = n

       first_value : list, values are a float or False
              : activates the fact that all data are offset so that the first value of each component is a certain value
              : the target value must be in the units wanted after applying the multiplyfactor
                  (if we want the first value to be 1 millimiter for a measure that was first in meters then converted in millimeter, the first value to enter is 1 millimiter)
              : put False so the component is not offseted
              : each member of the list is the value the first value of a component must be offset to (in the sequence order)
              : [first_value_x, first_value_y, first_value_z] for SequenceComposantes = ['x','y','z']
              : the dimension of this list is equal to the number of components this Array has.
              : only total : len(first_value) = 1
              : one component : len(first_value) = 1
              : n components : len(first_value) = n

              : Example : A value originally in meter, needs to be converted to mm and also offset to 1mm as the first value
                        : first_value = [1]

        vect_dir : bool : activates the fact that the output array is divided by its norm to get the direction of a vector
                        : (only available for vectors)
    """

    VariableOutput = {}
    VariableOutput["Description"] = VariableDescription
    VariableOutput["SequenceComposantes"] = []

    if offset and first_value:
        raise ValueError("Cannot activate both the argument 'offset' and 'first_value' at the same time")

    # Makes 1 column 2D array into 1D array so that every data with only one component have consistent shape
    if Array.ndim == 2 and Array.shape[1] == 1:
        Array = Array.flatten()

    # If the array is a matrix but it has only one column or one line for each timesteps, recize it to a 2D array
    if Array.ndim == 3:
        # Only one column
        if Array.shape[1] == 1:
            Array = Array[:, 0, :]

        # Only one line
        elif Array.shape[2] == 1:
            Array = Array[:, :, 0]

    # If the output is a vector (ndim=1) or has only one column, puts the output in total and no components are created
    if Array.ndim == 1 or Array.shape[1] == 1:

        # In the case we want to offset all the values so that they all start by a specific value
        if first_value:

            # the offset is the first value of the vector array minus the offset of this component
            # the offset is divided by the multiply factor to take in account the target coordinates units
            first_value_offset = Array[0] - first_value[0] / MultiplyFactor

            # Offsets the vector
            Array = Array - first_value_offset

        # In the case we want to add offset the data by a certain value
        elif offset:

            # Offsets the vector and takes into account the multiply factor
            Array = Array + offset[0] / MultiplyFactor

        # Name the component "Total" if no component name were entered
        if SequenceComposantes == "":
            Composante = "Total"
        else:
            Composante = SequenceComposantes[0]

        # Stores the vector
        VariableOutput[Composante] = Array * MultiplyFactor

        # np.isfinite
        VariableOutput["SequenceComposantes"].append(Composante)

        # error message
        if vect_dir:
            raise ValueError("The argument 'vect_dir' can only be used for 3D vectors, not 1D values")

    # If the output is 2D
    elif Array.ndim == 2:

        # In the case we want to add offset the data by a certain value
        if offset:
            # Goes through each column
            for col_index in range(Array.shape[1]):
                component_offset = offset[col_index]

                # Offsets the column and takes into account the multiply factor
                Array[:, col_index] = Array[:, col_index] + component_offset / MultiplyFactor

        # In the case we want to offset all the values so that they all start by a specific value
        elif first_value:
            # Goes through each column
            for col_index in range(Array.shape[1]):

                # The component is not offset if we put False
                if first_value[col_index] is False:
                    current_first_value_offset = 0
                else:
                    current_first_value_offset = first_value[col_index]

                # the offset for this component is the first value of each column minus the offset of this component
                # the offset is divided by the multiply factor to take in account the target coordinates units
                component_offset = Array[0, col_index] - current_first_value_offset / MultiplyFactor

                # Offsets the column
                Array[:, col_index] = Array[:, col_index] - component_offset

        # Sets the sequence to the default and adapts to the number of columns in the array
        if SequenceComposantes == '':
            # Si la séquence n'est pas spécifiée, met xyz en séquence par défaut en s'adaptant au nombre de composantes
            DefaultSequence = ['x', 'y', 'z']
            SequenceComposantes = DefaultSequence[0:Array.shape[1]]

        # Calculates the total of the component at each timestep if the total is not already calculated and is activated
        if "Total" not in SequenceComposantes and total_on:
            VariableOutput["Total"] = np.linalg.norm(Array, axis=1) * MultiplyFactor
            VariableOutput["SequenceComposantes"].append("Total")

        # Sets the default Component multiply factor and adapts to the number of columns in the array
        if Composantes_Inverse_Direction is False:
            DefaultComposantes_MultiplyFactor = [1, 1, 1, 1]
            Composantes_MultiplyFactor = DefaultComposantes_MultiplyFactor[0:Array.shape[1]]
        else:
            Composantes_MultiplyFactor = []
            # Multiply by -1 if inverse is true and by 1 if inverse is false
            for inverse in Composantes_Inverse_Direction:
                if inverse:
                    Composantes_MultiplyFactor.append(-1)
                elif inverse is False:
                    Composantes_MultiplyFactor.append(1)
                else:
                    raise ValueError("Composantes_Inverse_Direction must be filled with True or False")

        # if vect_dir is activated, the component values are the direction of the direction vector (value/norm_vector)
        if vect_dir:
            # each column of the array are divided by a column vector containing the total
            Array = Array / VariableOutput["Total"][:, None]

            # Resets the total to a vector filled with ones
            VariableOutput["Total"] = np.ones(len(VariableOutput["Total"]))

        # Parcours le nom des composantes dans l'ordre spécifié
        # Et multiplie par le facteur multiplicatif de la composante
        for col, Composante in enumerate(SequenceComposantes):
            VariableOutput[Composante] = Array[:, col] * MultiplyFactor * Composantes_MultiplyFactor[col]

        # Stores the variable component sequence
        VariableOutput["SequenceComposantes"] = [*VariableOutput["SequenceComposantes"], *SequenceComposantes]

    # Returns the array if it is a matrix
    else:
        VariableOutput = Array
    return VariableOutput


def AddVariableDictionaries(VariableDictionary_1, VariableDictionary_2):
    """
    Add 2 dictionaries created by array_to_dictionary that have the same SequenceComposantes and add them

    Returns a dictionary containing the added dictionaries components
    """

    Added_Dictionaries = VariableDictionary_1.copy()

    SequenceComposantes_1 = VariableDictionary_1["SequenceComposantes"]
    SequenceComposantes_2 = VariableDictionary_2["SequenceComposantes"]

    if not SequenceComposantes_1 == SequenceComposantes_2:

        raise ValueError("Les deux dictionnaires doivent avoir la même séquence de composantes")
        return

    # # Addition des totaux
    # Added_Dictionaries["Total"] = VariableDictionary_1["Total"] + VariableDictionary_2["Total"]

    # # Si les dictionnaires ont des composantes, les additionne une à une
    # if not SequenceComposantes_1 == [""]:
    #     for Composante in SequenceComposantes_1:
    #         Added_Dictionaries[Composante] = VariableDictionary_1[Composante] + VariableDictionary_2[Composante]

    # # Addition des totaux
    # Added_Dictionaries["Total"] = VariableDictionary_1["Total"] + VariableDictionary_2["Total"]

    # Si les dictionnaires ont des composantes, les additionne une à une
    # if not SequenceComposantes_1 == [""]:
    for Composante in SequenceComposantes_1:
        Added_Dictionaries[Composante] = VariableDictionary_1[Composante] + VariableDictionary_2[Composante]

    return Added_Dictionaries


def CleanFailedSimulationSteps(Output, Failed):
    """
    From an array of data of any size, deletes the data that failed during an anybody simulation

    Failed : removes the values in the results in case the simulation failed after a certain time
    Failed : is the first step number that failed (if the step number goes from 0 to nstep)
    """
    # Shape of the Output
    shapeOutput = np.shape(Output)

    # Shape of the part without fails
    shapeClean = np.copy(shapeOutput)
    shapeClean[0] = Failed

    # Shape of the part with failes
    shapeFailed = np.copy(shapeOutput)
    shapeFailed[0] = shapeOutput[0] - Failed

    # Creates arrays full of True to select non-failed steps
    Clean = np.full(tuple(shapeClean), True)
    # Creates arrays full of False to select failed steps
    Failed = np.full(tuple(shapeFailed), False)

    # Creates a mask of the same size of the output
    mask = np.append(Clean, Failed, axis=0)

    # Select the data from output without fails
    CleanOutput = Output[mask]

    # shape the output back to it's original shape
    CleanOutput = np.reshape(CleanOutput, shapeClean)

    return CleanOutput


def combine_variable(variable_dictionaries_list, operation="mean"):
    """
    From a list of variable dictionaries, combines each variable component

    variable_dictionaries : dict  = [variable_dictionary_1, variable_dictionary_2, variable_dictionary_3]

    operation : str : name of the operation done
              : list of combination posible : ["mean", "total", "min" "max"]
              : for "mean" also calculates the standard deviation
    ------------------------------------------------
    return
    combined_variable : dict : variable dictionary with the combined values of every
    """
    # makes a copy of the first variable
    combined_variable = variable_dictionaries_list[0].copy()

    Sequence_Composantes = variable_dictionaries_list[0]["SequenceComposantes"]

    list_operation = ["mean", "total", "min" "max"]
    if operation not in list_operation:
        raise ValueError(f"The combine variables operation '{operation}' is not part of the operations posible : list_operation")

    # goes through each component data
    for composante in Sequence_Composantes:
        # initializes the array that will contain the data for the current component
        composante_data = np.zeros((len(variable_dictionaries_list[0][composante]), len(variable_dictionaries_list)))

        # goes through each dictionary data and takes the value of the current component
        for variable_dictionary_index, variable_dictionary_data in enumerate(variable_dictionaries_list):
            composante_data[:, variable_dictionary_index] = variable_dictionary_data[composante]

        # makes wanted operation on all components
        # For the mean, also calculates the standard deviation
        if operation == "mean":
            combined_variable[composante] = np.mean(composante_data, axis=1)

            # Calculates the standard deviaton for the component and calls it "sd_component"
            combined_variable[f"sd_{composante}"] = np.std(composante_data, axis=1)

        elif operation == "total":
            combined_variable[composante] = np.sum(composante_data, axis=1)
        elif operation == "max":
            combined_variable[composante] = np.max(composante_data, axis=1)
        elif operation == "min":
            combined_variable[composante] = np.min(composante_data, axis=1)

    # adds the standard deviation to Sequence_Composantes
    if operation == "mean":
        sd_Sequence_Composantes = [f"sd_{composante}" for composante in Sequence_Composantes]

        combined_variable["SequenceComposantes"].extend(sd_Sequence_Composantes)

    return combined_variable


def save_results_to_file(result_dictionary, save_directory_path, save_file_name):
    """
    Function to save a result dictionary to a .pkl file

    This file can later be loaded with the function load_results_from_file

    result_dictionary : Any variable containing information
    save_directory_path : string : contains the path of the directory in which the file must be saved in
                  : the directory must already exist
                  : To save the file in the current directory : save_directory_path = ""

    save_file_name : string : The name of the .pkl file
    """

    import pickle

    # Creates the Full File path
    file_path = f"{save_directory_path}/{save_file_name}"

    # Saves the variable
    with open(file_path, 'wb') as file:

        # A new file will be created
        pickle.dump(result_dictionary, file)

        file.close()


def load_results_from_file(save_directory_path, save_file_name):
    """
    Loads a result dictionary from a .pkl file (that can be created by the save_results_to_file)

    save_directory_path : string : contains the path of the directory in which the file must be saved in
                  : the directory must already exist
                  : To save the file in the current directory : save_directory_path = ""

    save_file_name : string : The name of the .pkl file

    ---------------------------------
    return
    Variable : The loaded variable that was stored in the pickle file

    """

    import pickle

    # Creates the Full File path
    file_path = f"{save_directory_path}/{save_file_name}"

    # Open the file
    with open(file_path, 'rb') as file:

        # Call load method to deserialze
        result_dictionary = pickle.load(file)

        file.close()

    return result_dictionary


def save_result_variable_to_sheet(result_dictionary, variable_name, SequenceComposantes, description, xlsxwriter, muscle_variable=False):

    import pandas as pd

    # gets the result dictionary structure
    variables_deepness_counter, data_source = get_result_dictionary_data_structure(result_dictionary)

    workbook = xlsxwriter.book

    # Format of the description titles
    description_title_format = workbook.add_format({"align": "center", 'bold': True})
    description_title_format.set_bg_color('#C0C0C0')
    description_title_format.set_font_size(14)
    description_title_format.set_border(2)

    # Format of the description cells
    description_format = workbook.add_format({"align": "center"})
    description_format.set_border(2)

    # builds a dictionary filled with each column of a data frame containing the values of the current variable
    variable_data = {}

    # sheet named depending on the type of variable
    if muscle_variable:
        sheet_name = f"Muscle.{variable_name}"
    else:
        sheet_name = variable_name

    # empty dictionary that will contain the data of the current variable, that will be transformed to a dataframe then to excel
    variable_data = {}

    # The excel sheet will have a different structure depending on the type of data structure
    if variables_deepness_counter == 0:
        # Muscle variable
        if muscle_variable:
            # list of muscles
            muscle_list = list(result_dictionary["Muscles"].keys())

            for muscle in muscle_list:
                for index, composante in enumerate(SequenceComposantes):
                    variable_data[f"{muscle}_{composante}"] = [composante] + result_dictionary["Muscles"][muscle][muscle][variable_name][composante].tolist()

                    # for the first component, the column also contains the muscle name on top
                    if index == 0:
                        variable_data[f"{muscle}_{composante}"].insert(0, muscle)
                    else:
                        variable_data[f"{muscle}_{composante}"].insert(0, "")

        # Normal variable
        else:
            for composante in SequenceComposantes:
                variable_data[composante] = [composante] + result_dictionary[variable_name][composante].tolist()

    if variables_deepness_counter == 1:
        # Muscle variable
        if muscle_variable:
            # list of muscles
            first_case = list(result_dictionary.keys())[0]
            muscle_list = list(result_dictionary[first_case]["Muscles"].keys())

            for muscle in muscle_list:

                # lists each variable for each simulation case
                for index_case, case_name in enumerate(result_dictionary):
                    for index, composante in enumerate(SequenceComposantes):
                        variable_data[f"{muscle}_{case_name}_{composante}"] = [composante] + result_dictionary[case_name]["Muscles"][muscle][muscle][variable_name][composante].tolist()

                        # inserts the name of the case on top of the first component of the variable
                        if index == 0:
                            variable_data[f"{muscle}_{case_name}_{composante}"].insert(0, case_name)
                        else:
                            variable_data[f"{muscle}_{case_name}_{composante}"].insert(0, "")

                        # inserts the name of the muscle on top of the first case name
                        if index_case == 0:
                            variable_data[f"{muscle}_{case_name}_{composante}"].insert(0, muscle)
                        else:
                            variable_data[f"{muscle}_{case_name}_{composante}"].insert(0, "")

        # Normal variable
        else:
            # lists each variable for each simulation case
            for case_name in result_dictionary:
                for index, composante in enumerate(SequenceComposantes):
                    variable_data[f"{case_name}_{composante}"] = [composante] + result_dictionary[case_name][variable_name][composante].tolist()

                    # inserts the name of the case on topof the first component of the variable
                    if index == 0:
                        variable_data[f"{case_name}_{composante}"].insert(0, case_name)
                    else:
                        variable_data[f"{case_name}_{composante}"].insert(0, "")

    # compared simulation cases. variables data are grouped by cases for each simulation
    if variables_deepness_counter == 2:
        # Muscle variable
        if muscle_variable:
            # list of muscles
            first_simulation = list(result_dictionary.keys())[0]
            case_list = list(result_dictionary[first_simulation].keys())
            first_case = case_list[0]
            muscle_list = list(result_dictionary[first_simulation][first_case]["Muscles"].keys())

            for muscle in muscle_list:

                # lists each variable for each simulation case
                for index_case, case_name in enumerate(case_list):
                    for index_simulation, simulation in enumerate(result_dictionary):
                        for index, composante in enumerate(SequenceComposantes):
                            variable_data[f"{muscle}_{case_name}_{simulation}_{composante}"] = [composante] + result_dictionary[simulation][case_name]["Muscles"][muscle][muscle][variable_name][composante].tolist()

                            # inserts the name of the simulation on top of the first component of the variable
                            if index == 0:
                                variable_data[f"{muscle}_{case_name}_{simulation}_{composante}"].insert(0, simulation)
                            else:
                                variable_data[f"{muscle}_{case_name}_{simulation}_{composante}"].insert(0, "")

                            # inserts the name of the case on top of the first simulation name
                            if index_simulation == 0:
                                variable_data[f"{muscle}_{case_name}_{simulation}_{composante}"].insert(0, case_name)
                            else:
                                variable_data[f"{muscle}_{case_name}_{simulation}_{composante}"].insert(0, "")

                            # inserts the name of the muscle on top of the first case name
                            if index_case == 0 and index_simulation == 0:
                                variable_data[f"{muscle}_{case_name}_{simulation}_{composante}"].insert(0, muscle)
                            else:
                                variable_data[f"{muscle}_{case_name}_{simulation}_{composante}"].insert(0, "")

        # Normal variable
        else:
            # lists each variable for each simulation case
            # list of muscles
            first_simulation = list(result_dictionary.keys())[0]
            case_list = list(result_dictionary[first_simulation].keys())

            # goes through each simulation cases
            for case_name in case_list:
                # goes through each simulation
                for index_simulation, simulation in enumerate(result_dictionary):
                    for index, composante in enumerate(SequenceComposantes):
                        variable_data[f"{case_name}_{simulation}_{composante}"] = [composante] + result_dictionary[simulation][case_name][variable_name][composante].tolist()

                        # inserts the name of the case on top of the first component of the variable
                        if index == 0:
                            variable_data[f"{case_name}_{simulation}_{composante}"].insert(0, simulation)
                        else:
                            variable_data[f"{case_name}_{simulation}_{composante}"].insert(0, "")

                        # inserts the name of the case on top of the first simulation name
                        if index_simulation == 0 and index == 0:
                            variable_data[f"{case_name}_{simulation}_{composante}"].insert(0, case_name)
                        else:
                            variable_data[f"{case_name}_{simulation}_{composante}"].insert(0, "")

    # converts the variable data to a dataframe
    df = pd.DataFrame.from_dict(variable_data)

    # adds 2 empty columns for the descriptions
    df.insert(0, "description_titles", np.nan)
    df.insert(0, "description", np.nan)

    # converts the dataframe to an excel sheet
    df.to_excel(xlsxwriter, index=False, header=False, sheet_name=sheet_name)

    worksheet = xlsxwriter.sheets[sheet_name]

    # adds the description of the variables
    worksheet.write(0, 0, "Variable informations", description_title_format)
    worksheet.write(1, 0, "Variable", description_title_format)
    worksheet.write(2, 0, "Description", description_title_format)

    # adds the description of the variables
    worksheet.write(1, 1, variable_name, description_format)
    worksheet.write(2, 1, description, description_format)

    # autofits the columns
    worksheet.autofit()
    # Sets the column width to a certain amount for the variable informations
    worksheet.set_column(0, 0, 26)


def get_model_informations(result_dictionary: dict, variables_deepness_counter: int) -> dict:
    """
    Creates a dictionary containing pandas.Series of the model informations

    result_dictionary : dict : result_dictionary

    variables_deepness_counter : int : int that indicates the type of the result dictionary (result from the functin get_result_dictionary_data_structure)


    return
    -------
    model_informations = dict contains pandas.Series that store the model informations
                       = {line, category of information 1, informations of the category 1, empty line,
                          category of information 2, informations of the category 2
                          ....
                          }

    """
    import pandas as pd

    def get_case_model_informations(case_result_dictionary: dict) -> pd.DataFrame:
        """
        gets the model information of a result dictionary without simulation cases

        case_result_dictionary : dict result dictionary without simulation cases

        returns
        -----------
        case_model_informations_df : pd.DataFrame Store model informations

        """
        import pandas as pd

        result_model_informations = case_result_dictionary["Model informations"]

        case_model_informations = {}

        for category_name, informations_category in result_model_informations.items():
            # adds an empty entry to make an empty line before the category name

            for information_name, information in informations_category.items():
                case_model_informations[information_name] = pd.Series([information_name, information])

        case_model_informations_df = pd.DataFrame(case_model_informations).T

        return case_model_informations_df

    # Goes through each cases and simulations and add them to the model informations dictionary
    model_informations_df = pd.DataFrame()

    # for a single case result dictionary
    if variables_deepness_counter == 0:

        model_informations_df = get_case_model_informations(result_dictionary)

    # Result with simulation cases
    elif variables_deepness_counter == 1:

        for case_name in result_dictionary:
            case_model_informations = get_case_model_informations(result_dictionary[case_name])

            # adds case names and simulation names to the dataframe
            case_names_df = pd.DataFrame({"case_name": ["Case", case_name]}).T
            case_model_informations = pd.concat([case_names_df, case_model_informations])
            case_model_informations[f"empty_{case_name}"] = ""

            # adds the current case informations to the previous ones
            model_informations_df = pd.concat([model_informations_df, case_model_informations], axis=1)

    # Result with compared simulation cases
    elif variables_deepness_counter == 2:

        for simulation_name, simulation_results in result_dictionary.items():

            for case_name in simulation_results:
                case_model_informations = get_case_model_informations(result_dictionary[simulation_name][case_name])

                # adds case names and simulation names to the dataframe
                case_names_df = pd.DataFrame({"simulation_name": ["Simulation", simulation_name],
                                              "case_name": ["Case", case_name]}).T
                case_model_informations = pd.concat([case_names_df, case_model_informations])
                case_model_informations[f"empty_{case_name}{simulation_name}"] = ""

                # adds the current case informations to the previous ones
                model_informations_df = pd.concat([model_informations_df, case_model_informations], axis=1)

    return model_informations_df


def result_dictionary_to_excel(result_dictionary: dict, excel_file_name: str):
    """
    Function that saves a result dictionary into an excel file

    Uses the package : xlsxwriter
    """
    import pandas as pd

    xlsxwriter = pd.ExcelWriter(f'{excel_file_name}.xlsx', engine='xlsxwriter')

    variables_deepness_counter, data_source = get_result_dictionary_data_structure(result_dictionary)

    # model informations
    model_informations_df = get_model_informations(result_dictionary, variables_deepness_counter)

    # model_informations_df = pd.DataFrame(model_informations).T
    model_informations_df.to_excel(xlsxwriter, index=False, header=False, sheet_name="Model informations")
    worksheet = xlsxwriter.sheets["Model informations"]
    worksheet.set_tab_color("red")
    worksheet.autofit()

    # gets the variables informations
    variables_informations, muscle_variables_informations = get_result_dictionary_variables_informations(result_dictionary)

    for variable in variables_informations:
        description = variables_informations[variable]["Description"]
        SequenceComposantes = variables_informations[variable]["SequenceComposantes"]
        save_result_variable_to_sheet(result_dictionary, variable, SequenceComposantes, description, xlsxwriter, muscle_variable=False)

    # for muscle variables
    for muscle_variable in muscle_variables_informations:
        description = muscle_variables_informations[muscle_variable]["Description"]
        SequenceComposantes = muscle_variables_informations[muscle_variable]["SequenceComposantes"]
        save_result_variable_to_sheet(result_dictionary, muscle_variable, SequenceComposantes, description, xlsxwriter, muscle_variable=True)

    # Close the Pandas Excel writer and output the Excel file.
    xlsxwriter.close()

    print(f"Results exported to the excel file : {excel_file_name}.xlsx")
