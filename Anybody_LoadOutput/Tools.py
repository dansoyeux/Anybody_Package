import numpy as np


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


def array_to_dictionary(Array, VariableDescription='', SequenceComposantes='', MultiplyFactor=1, Composantes_Inverse_Direction=False, offset=False, vect_dir=False, total_on=True, **kwargs):
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

       offset : list, values are a float or False
              : activates the fact that all data are offset so that the first value of each component is a certain value
              : the target value takes into account the MultiplyFactor
              : put False so the component is not offset
              : each member of the list is the value the first value of a component must be offset to (in the sequence order)
              : [first_value_x, first_value_y, first_value_z] for SequenceComposantes = ['x','y','z']
              : the dimension of this list is equal to the number of components this Array has.
              : only total : len(offset) = 1
              : one component : len(offset) = 1
              : n components : len(offset) = n

              : Example : A value originally in meter, needs to be converted to mm and also offset to 1mm as the first value
                        : offset = [1]

        vect_dir : bool : activates the fact that the output array is divided by its norm to get the direction of a vector
                        : (only available for vectors)
    """

    VariableOutput = {}
    VariableOutput["Description"] = VariableDescription
    VariableOutput["SequenceComposantes"] = []

    # Makes 1 column 2D array into 1D array so that every data with only one component have consistent shape
    if Array.ndim == 2 and Array.shape[1] == 1:
        Array = Array.flatten()

    # If the output is a vector (ndim=1) or has only one column, puts the output in total and no components are created
    if Array.ndim == 1 or Array.shape[1] == 1:

        # In the case we want to offset all the values so that they all start by 0
        if offset:

            # the offset is the first value of the vector array minus the offset of this component
            # the offset is divided by the multiply factor to take in account the target coordinates units
            offset = Array[0] - offset[0] / MultiplyFactor

            # Offsets the column
            Array = Array - offset

        # Name the component "Total" if no component name were entered
        if SequenceComposantes == "":
            Composante = "Total"
        else:
            Composante = SequenceComposantes[0]

        # Stores the vector without the nan values
        VariableOutput[Composante] = Array[~np.isnan(Array)] * MultiplyFactor

        VariableOutput["SequenceComposantes"].append(Composante)

        # error message
        if vect_dir:
            raise ValueError("The argument 'vect_dir' can only be used for 3D vectors, not 1D values")

    # If the output is 2D
    elif Array.ndim == 2:

        # In the case we want to offset all the values so that they all start by the value in the offset
        if offset:
            # Goes through each column
            for col_index in range(Array.shape[1]):

                # The component is not offset if we put False
                if offset[col_index] is False:
                    current_offset = 0
                else:
                    current_offset = offset[col_index]

                # the offset for this component is the first value of each column minus the offset of this component
                # the offset is divided by the multiply factor to take in account the target coordinates units
                component_offset = Array[0, col_index] - current_offset / MultiplyFactor

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
        # And deletes nan values
        for col, Composante in enumerate(SequenceComposantes):
            VariableOutput[Composante] = Array[:, col][~np.isnan(Array[:, col])] * MultiplyFactor * Composantes_MultiplyFactor[col]

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
        if operation == "mean":
            combined_variable[composante] = np.mean(composante_data, axis=1)
        elif operation == "total":
            combined_variable[composante] = np.sum(composante_data, axis=1)
        elif operation == "max":
            combined_variable[composante] = np.max(composante_data, axis=1)
        elif operation == "min":
            combined_variable[composante] = np.min(composante_data, axis=1)

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
