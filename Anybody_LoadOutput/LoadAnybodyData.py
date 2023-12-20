import anypytools.h5py_wrapper as h5py2
from anypytools.datautils import read_anyoutputfile
import numpy as np

from Anybody_Package.Anybody_LoadOutput.Tools import array_to_dictionary
from Anybody_Package.Anybody_LoadOutput.Tools import transform_vector
from Anybody_Package.Anybody_LoadOutput.Tools import CleanFailedSimulationSteps

# POUR L'INSTANT, LOAD LES VARIABLES DES H5, LOAD PAS LES VARIABLES DES FICHIERS TEXTES


def Loadh5File(FilePath, Failed=False, GHReactionsShape=False, AddConstants=False):
    """
    Wraps every information about the h5 file in one dictionary


    GHReactionsShape is custom arguments in my simulation

    NOTE : IN THIS SCRIPT, THE H5 FILE IS NOT CLOSED WHICH CAN AFFECT PERFORMANCE A LOT IF A LOT OF h5 ARE LOADED
        : IF THE LoadResultsh5 function is used, the h5 FILE IS CLOSED AT THE END

        : TO CLOSE IT RUN h5File["h5Data"].close() after loading every variable needed


    NOTE 2 : IF AN ERROR SAYING THE FILE WASN'T FOUND, CHECK THE PATH OR IT MEANS THAT THE PATH HAS TOO MANY CHARACTERS
    (TRY TO RENAME THE H5 FILE WITH A SHORTER NAME OR CHECK THAT A DIRECTORY DOESN'T HAVE A VERY LONG NAME)
    """

    import os

    # Loads the h5 file data
    file_extension = "anydata.h5"

    file_full_name = f"{FilePath}.{file_extension}"

    # Gets the absolute path of the file that will be open
    file_absolute_path = os.path.abspath(file_full_name)

    if len(file_absolute_path) > 241:
        raise ValueError(f"The path of the file : \n{FilePath}\nis too long ({len(file_absolute_path)} > 241) so it cannot be opened.\nTry to shorten the name of the file or the names of the directories containing this file or put the file less deep on your harddrive")

    try:
        h5Data = h5py2.File(file_full_name, "r")
    except OSError:
        raise ValueError(f"The file doesn't exist or is damaged and wasn't posible to be opened : \n{FilePath}\nCheck the directory path or the file name or that the h5 file's name finishes with .anydata")

    h5File = {"FilePath": FilePath, "Failed": Failed, "h5Data": h5Data,
              "GHReactions": GHReactionsShape}

    return h5File


def LoadAnyFileOutVariable(FileOutPath, FileType, VariablePath=str, LoadConstantsOnly=False):
    """
    Loads a specific variable from an AnyFileOut file
    Or can load only the constants
    """

    data, dataheader, constantsdata = read_anyoutputfile(
        FileOutPath + "." + FileType)

    if LoadConstantsOnly is False:
        # Constructs a dictionary with all the variables and constants
        DataDictionary = {}
        for index, Variable in enumerate(dataheader):

            # Deletes the anybody path of the variable name to only keep the variable name
            Variable = Variable.replace("Main.Study.FileOut.", "")

            DataDictionary[Variable] = data[:, index]

        # adds the variable to DataDictionary
        for index, Variable in enumerate(constantsdata):

            DataDictionary[Variable] = constantsdata[Variable]

        # Loads every variable and constants
        if VariablePath == "all":
            Output = DataDictionary
        else:
            # Loads a specific variable
            Output = DataDictionary[VariablePath]

    # Loads only the constanst
    if LoadConstantsOnly:
        Output = constantsdata

    return Output


def LoadAnyFileOut(FileOutPath, ConstantsDictionary, FileType="txt", LoadConstantsOnly=False):
    """
    Load an AnyFileOut and creates a Dictionary
    FileType : says if the FileOut is a .txt,.csv...
    LoadConstantsOnly : True if output must only be the constants to complete these missing informations while loading a .h5 file
    Ex : FileOut.txt :
         FileOutPath = File Path and Name
         FileType = txt

    FileOut is a dictionary
    Constants can be stored in different categories (different dictionary keys)

    ConstantsDictionary : = {"CONSTANTS_CATEGORY": [LIST_OF_CATEGORY_CONSTANTS]}
    Ex : to have a mannequin variables category and a simulation parameters category
    ConstantsDictionary =     {"Simulation_Parameters": ["Movement", "Case", "GHReactions", "nstep"],
                               "Mannequin": ["GlenohumeralFlexion", "GlenohumeralAbduction", "GlenohumeralExternalRotation"]}
    """

    FileOut = {}

    """
    # The variables are loaded only if LoadConstantsOnly is False
    # if LoadConstantsOnly == False:
    À FAIRE CE SCRIPT POUR LOAD UNE SÉRIE DE VARIABLE OU TOUTES LES VARIABLES DEPUIS ANYFILEOUT
    ET POUVOIR CHOISIR CONSTANTES, CONSTANTES + VARIABLE, VARIABLE
    """

    # Loads the constants
    constantsdata = LoadAnyFileOutVariable(
        FileOutPath, FileType, LoadConstantsOnly=True)

    # All the constants names in the .txt file
    ConstantsNames = list(dict.keys(constantsdata))

    # Extracts the AnyOutputFile object file in ConstantsDictionary
    AnybodyFileOutPath = ConstantsDictionary["AnybodyFileOutPath"]

    # Makes a copy of ConstantsDictionary to modify it without changing the original dictionnary
    ConstantsDictionary_Copy = ConstantsDictionary.copy()
    # Deletes the AnyOutputFile path from the dictionary
    del ConstantsDictionary_Copy["AnybodyFileOutPath"]

    for Constantindex, ConstantName in enumerate(ConstantsNames):
        ConstantsNames[Constantindex] = ConstantName.replace(f"{AnybodyFileOutPath}.", "")

    # Goes through every constants categories defined in ConstantsDictionary_Copy (the dictionnary containing the variables without the AnybodyFileOutPath)
    for CategoryName, CategoryConstants in ConstantsDictionary_Copy.items():
        FileOut[CategoryName] = {}

        # Goes through all the constants in this category an places it in FileOut[CategoryName]
        for Constant in CategoryConstants:

            # Only loads the constant if it exists
            if Constant in ConstantsNames:
                # Loads the Constant value (the key is AnybodyFileOutPath.Constant)
                FileOut[CategoryName][Constant] = constantsdata[f"{AnybodyFileOutPath}.{Constant}"]

        # Deletes the category if it is an empty dictionary (no constants of this category were found)
        if FileOut[CategoryName] == {}:
            del FileOut[CategoryName]

    return FileOut


def LoadAnyVariable(h5File, VariablePath="", MusclePath="", OutputDictionary=True, select_muscle_RefFrame_output="", rotation_matrix_path="", inverse_rotation=False, select_matrix_line=False, select_matrix_column=False, **kwargs):
    """
    Uses the Anypytool function to load a model variable from an .anydata.h5 file by using the anybody variable path in the study with . instead of /

    WARNING : Only non constants values are stored in h5. Values that don't change during a simulation are not stored in h5 files

    Creates a Dictionary containing : The variable description that will be used for graphs
                                       If the variable is multidinensional : The variable total is calculated and the components are output
                                       If the variable is a vector : The variable is put in "Total" and no components are created

    VariablePath : The path of the variable (IN THE OUTPUT DIRECTORY)
                 : Ex: The variable Main.HumanModel.BodyModel.Right.ShoulderArm.Seg.Scapula.ghProth.Axes
                       will be stored after the simulation in the directory Main.Study.Output.Model.Right.ShoulderArm.Seg.Scapula.ghProth.Axes

                 VariablePath = Output.Model.Right.ShoulderArm.Seg.Scapula.ghProth.Axes

                 If a shortcut to the Seg directory was created (AnyFolder &Seg= Main.HumanModel.BodyModel.Right.ShoulderArm.Seg;)
                 VariablePath = Output.Seg.Scapula.ghProth.Axes


    FilePath is the path of the file to load (it must begin with "Output." since it's the output data that is stored in an h5 file)
    VariableDescription : Description of the variable that is going to be used in the axis label on the graph
    SequenceComposantes : Indicates which colomns corresponds to which component (x,y or z)
                          The sequence is xyz by defaukt. So the first column will be x, then y then z
    MultiplyFactor : If multiplying every value by a factor is needed (to convert m to mm for example)

    OutputDictionary : Can be put on False to be able to have Variable in an array and not in a Dictionary that seperates it's components
                      : False must be used if you want to make calculations or if Variable is not in 2D but more (examples rotation matrices)



    rotation_matrix_path : str
                 : can be used if the variable is a vector and needs to be rotated by a certain rotation matrix for each timestep in anybody
                 : string of the path of the rotation matrix over time dim(nstep,3,3) that will be used to rotate the vector

    inverse_rotation : bool (False by default)
                     : specifies if the transformation needs to be an inverse transform or not

    select_matrix_line : int : the index of the line from a matrix to select if the variable is a matrix
                             : the index of the first line is 0

    select_matrix_column : int : the index of the line from a matrix to select if the variable is a matrix
                               : the index of the first line is 0


    AJOUTER DANS LA DOC
    Only use for a muscle :

    select_muscle_RefFrame_output : str : selects the muscle output in the "insertion" or "origin" reference frame of the muscle

    Under every muscle, there is a folder RefFrameOutput containing the output information of the muscle (Force, moment force, jacobian)
    And each of theses variables is a matrix (a n x 3 matrix) that contain the output of the muscle at different reference frame.
    Each reference frame are the ones used to define the muscle, there are in this order if they exist :(insertion, via point, origin, wrapping segment, other refframes).

    For a simple muscle, the matrix is 2x3 where the first line is the insertion and the second is the origin.
    But for a muscle that has a via point, the order is : insertion, via_point, origin

    So if we want the force at the origin, it is sometimes in the second line, sometimes later depending on the number of viapoints.

    In this same folder, there is an array of pointer that tell which line corresponds to which reference frame.

    So if we want output of the origin of the muscle : select_muscle_RefFrame_output = "origin"
    So if we want output of the insertion of the muscle : select_muscle_RefFrame_output = "insertion"

    """

    # Extrait les informations nécessaires du dictionnaire h5File créé avec la fonction Loadh5FileData
    FilePath = h5File["FilePath"]
    Failed = h5File["Failed"]
    h5Data = h5File["h5Data"]

    # Ne cherche la variable que si elle existe dans le fichier
    if VariablePath in h5Data:
        Output = np.array(h5Data[VariablePath])

        # Selects a particular line of the variable to create a vector
        if select_matrix_line is not False:
            if Output.ndim == 3:
                Output = Output[:, select_matrix_line, :]
            else:
                raise ValueError(f"The variable : {VariablePath} \nisn't a matrix, the option select_matrix_line cannot be used")

        # Selects a particular column of the variable to create a vector
        if select_matrix_column is not False:
            if Output.ndim == 3:
                Output = Output[:, :, select_matrix_column]
            else:
                raise ValueError(f"The variable : {VariablePath} \nisn't a matrix, the option select_matrix_column cannot be used")

        # Selects a certain line of a variable in a muscle RefFrameOutput
        if select_muscle_RefFrame_output:

            # checks that the variable to charge is a muscle variable
            if not MusclePath:
                raise ValueError(f"The variable : {VariablePath} \nmust be a muscle variable to activate the loading option : 'select_muscle_RefFrame_output'")

            # The variable must be a matrix in the RefFrameOutput
            if not Output.ndim == 3 or "RefFrameOutput" not in VariablePath:
                raise ValueError(f"The variable : {VariablePath} must be a matrix in the RefFrameOutput folder of a muscle to use the loading option 'select_muscle_RefFrame_output'")

            # The origin is always the first member of RefFrameArray
            if select_muscle_RefFrame_output == "origin":

                # Selects the first ref frame output line
                Output = Output[:, 0, :]
            # for the insertion the position depends on the number of via points

            elif select_muscle_RefFrame_output == "insertion":
                # Counts the number of via points in this directory
                # tests if there is only one via point
                if f"{MusclePath}.Via" in h5Data:
                    # there is one via point so the insertion is at position 2 (3rd line)
                    RefFrameOutput_position = 2

                # increase the number after Via (Via1, Via2) and counts the max number of via points
                else:
                    via_points_counter = 0
                    while f"{MusclePath}.Via{via_points_counter + 1}" in h5Data:
                        via_points_counter += 1

                    # if no via points, the position is the second row (1)
                    if via_points_counter == 0:
                        RefFrameOutput_position = 1
                    else:
                        RefFrameOutput_position = 1 + via_points_counter

                # Selects the wanted ref frame output line
                Output = Output[:, RefFrameOutput_position, :]
            else:
                raise ValueError(f"for the Muscle variable {VariablePath}\n'select_muscle_RefFrame_output' '{select_muscle_RefFrame_output}' not suported\nOnly'insertion' and 'origin' are supported")

    # Si la variable n'existe pas, ne la cherche pas, met un message d'erreur et remplit la variable avec des 0
    else:
        print(f"La variable : {VariablePath} \nn'existe pas dans le fichier h5 : {FilePath}")
        Output = np.zeros(len(h5Data["Output.Abscissa.t"]))

    # If Failed is the number of the first step that failed, deletes these failed steps
    if Failed is not False:

        CleanOutput = CleanFailedSimulationSteps(Output, Failed)

    # When none of the steps failed
    else:
        CleanOutput = Output

    # rotates the vector if rotation_matrix_path has been declared
    if rotation_matrix_path:
        if CleanOutput.ndim == 2:
            # get the rotation matrix if it exists in the h5Data
            if rotation_matrix_path in h5Data:
                rotation_matrix = np.array(h5Data[rotation_matrix_path])

                CleanOutput = transform_vector(CleanOutput, rotation_matrix, inverse_transform=inverse_rotation)

            else:
                print(f"The rotation matrix : {rotation_matrix_path} n'existe pas dans le fichier h5 : {FilePath}")
        # if the array isn't a vector, it cannot be transformed
        else:
            raise ValueError(f"The variable : '{VariablePath}' isn't a vector, so it cannot be rotated by using the argument 'rotation_matrix_path'")

    # Mise en forme du dictionnaire output si activé
    if OutputDictionary:

        # If the output is a vector (ndim=1) or has only one column
        # vect_dir cannot be activated
        if CleanOutput.ndim == 1 or CleanOutput.shape[1] == 1:
            if kwargs.get("vect_dir", False):
                raise ValueError(f"The variable '{VariablePath}' is a 1D value not a vector, so the argument 'vect_dir' cannot be used to calculate the director vector")

        # Converts the array to a Dictionary
        VariableOutput = array_to_dictionary(CleanOutput, **kwargs)

    # Variable output est un array si OutputDictionary est False
    else:
        MultiplyFactor = kwargs.get("MultiplyFactor", 1)
        VariableOutput = CleanOutput * MultiplyFactor

    return VariableOutput


def LoadMuscle(h5File, AnybodyMuscleName, MuscleName, PartString, AnybodyPartNumbers, MuscleVariableDictionary, PartNumbers_in_Dictionary=None, FileType="h5", ):
    """
    Load the variables of a muscle from a .anydata.h5 file or from a AnyFileOut file (when the type is h5 but another extension)
    Failed : removes the 0 in the results in case the simulation failed after a certain time
    It can handle muscles that are seperated in multiple parts (deltoideys lateral has 5 muscles in anybody)

    Also calculates the total of each variable along the multiple parts.
    For the activity or corrected activity, the Total is the mean activity of all parts at each timestep
    If the variable is an activity or a force, also calculates its maximum at each timestep and names this component "Max"

    [AnybodyPartNumbers] : List to select the parts to load
                          : To load multiple parts [AnybodyPartNumbersList] = [FirstPart, LastPart]
                              : FirstPart = the number of the first part to load (generally 1)
                              : LastPart = the number of the last part to load
                              Example : [AnybodyPartNumbers] = [1,3] will load the part 1,2 and 3 of the muscle

                          : To select muscle with only one part without a number [AnybodyPartNumbers] = []
                          : To select only the 3rd part of a muscle [AnybodyPartNumbers] = [3]


     Ex: to load the deltoideus lateral (Called deltoideus_lateral_part_n in anybody) that has 4 parts and give it the name Deltoideus Lateral
         the supraspinatus (Called supraspinatus_n) that has 6 parts and give it the name Supraspinatus
         the biceps brachii longum (named biceps_brachii_caput_longum) that has only 1 part
         the middle trapezius (part 1 to 3 of the trapezius_scapular_n)


    PartString : String to add after the name to have the name of the muscle part in anybody
    Ex : deltoideus goes from deltoideus_lateral_part_1 to deltoideus_lateral_part_5
         AnybodyPartNumbers = [1, 5]
         PartString = "_part_"

    Ex: Select only the first part of the deltoideus_lateral
        AnybodyPartNumbers = [1]

    NumberOfParts = Nombre de parties dans la sélection
                Calculated in LoadOutput.DefineVariablesToLoad
                Si le muscle comporte une seule partie, stocke 1
    """

    """
    FAIRE LE LOAD DEPUIS AnyFileOutFile
    """

    MuscleOutput = {}

    # Names to give in the Dictionary
    VariableNames = list(MuscleVariableDictionary.keys())

    # If the muscle is in one part
    if AnybodyPartNumbers == []:

        # Puts every musclepart values in a different Dictionary key
        MuscleOutput[MuscleName] = {}

        # Loads the muscle variables from a h5 file
        if FileType == "h5":

            # Parcours les variables à load et les met dans le dictionnaire du muscle
            for VariableName in VariableNames:

                # Gets the path of the folder that contains the variable
                MuscleFolderPath = MuscleVariableDictionary[VariableName]["MuscleFolderPath"]

                # Builds the Muscle variable path
                MusclePath = MuscleFolderPath + "." + AnybodyMuscleName

                # if an AnybodyVariableName is entered, adds it to the path
                # If AnybodyVariableName=="", it means the variable charged is named MuscleFolderPath.AnybodyMuscleName
                if MuscleVariableDictionary[VariableName]["AnybodyVariableName"]:
                    MuscleVariablePath = MusclePath + "." + MuscleVariableDictionary[VariableName]["AnybodyVariableName"]

                # Gets the loading options of the variable other than its path
                variable_loading_options = MuscleVariableDictionary[VariableName].copy()
                del variable_loading_options["AnybodyVariableName"]

                # Loads the muscle variable and stores it in a Dictionary
                MuscleOutput[MuscleName][VariableName] = LoadAnyVariable(h5File, MuscleVariablePath, MusclePath=MusclePath, **variable_loading_options)

        # # Loads the muscle variables from a AnyFileOut file
        # else:
        #     for VariableName in VariableNames:
        #         MuscleVariablePath = MuscleFolderPath + "." + AnybodyMuscleName + "." + AnybodyVariableNames[index]
        #         MuscleOutput[VariableName] = LoadAnyFileOutVariable(
        #             h5File, FileType, MuscleVariablePath)

    # if only one muscle part is loaded
    # Will load it and name it as the MuscleName without a number
    elif len(AnybodyPartNumbers) == 1:

        # Gets the number of the muscle part
        Part = AnybodyPartNumbers[0]

        # Creates the name of the muscle part in anybody
        AnybodyMusclePart = AnybodyMuscleName + PartString + str(Part)

        # Creates the name of the muscle part given in the Output Dictionary without a number
        MusclePart = MuscleName

        # Puts every musclepart values in a different Dictionary key
        MuscleOutput[MusclePart] = {}

        # Loads the muscle variables from a h5 file
        if FileType == "h5":

            # For every variable, loads the data of every part of the muscle
            for VariableName in VariableNames:

                # Gets the path of the folder that contains the variable
                MuscleFolderPath = MuscleVariableDictionary[VariableName]["MuscleFolderPath"]

                MusclePath = MuscleFolderPath + "." + AnybodyMusclePart

                # if an AnybodyVariableName is entered, adds it to the path
                # If AnybodyVariableName=="", it means the variable charged is named MuscleFolderPath.AnybodyMuscleName
                if MuscleVariableDictionary[VariableName]["AnybodyVariableName"]:
                    MuscleVariablePath = MusclePath + "." + MuscleVariableDictionary[VariableName]["AnybodyVariableName"]

                # Gets the loading options of the variable other than its path
                variable_loading_options = MuscleVariableDictionary[VariableName].copy()
                del variable_loading_options["AnybodyVariableName"]

                # Loads the muscle part variables
                MuscleOutput[MusclePart][VariableName] = LoadAnyVariable(h5File, MuscleVariablePath, MusclePath=MusclePath, **variable_loading_options)

        # # Loads the muscle variables from a AnyFileOut file
        # else:
        #     # For every variable, loads the data of every part of the muscle
        #     for VariableName in VariableNames:

        #         MuscleVariablePath = MuscleFolderPath + "." + AnybodyMusclePart + "." + AnybodyVariableNames[index]

        #         # Loads the muscle part variables
        #         MuscleOutput[MusclePart][VariableName] = LoadAnyFileOutVariable(
        #             h5File, FileType, MuscleVariablePath)

    # if multiple muscle parts are selected
    elif len(AnybodyPartNumbers) == 2:

        # informations sur les parties
        FirstPart = AnybodyPartNumbers[0]
        LastPart = AnybodyPartNumbers[1]

        # Parts number to load
        PartsNumbersToLoad = list(range(FirstPart, LastPart + 1))

        # Nombres qui seront données dans le dictionnaires après leur nom (valeurs contenues dans PartNumbers_in_Dictionary)
        # Pour les muscles qui sont composés de quelques parties (exemple: middle trapezius = [4, 6] mais devra être nommé 1-2-3 et pas 4-5-6)
        PartNumbersDictionary = list(range(PartNumbers_in_Dictionary[0], PartNumbers_in_Dictionary[1] + 1))

        # Parcours les numéros des muscleparts de la première sélectionnée à la dernière AnybodyPartNumbers = [1, 3] chargera les parties 1, 2 et 3
        # crée un indice pour pouvoir sélectionner le numéro de partie qui sera donné dans le dictionnaire (sélectionné dans PartNumbersDictionary)
        for Partindex, Part in enumerate(PartsNumbersToLoad):

            # Creates the name of the muscle part in anybody
            AnybodyMusclePart = AnybodyMuscleName + PartString + str(Part)

            # Creates the name of the muscle part given in the Output Dictionary
            # uses the numbers in PartNumbersDictionary (that go from 1 to NumberOfParts)
            MusclePart = MuscleName + " " + str(PartNumbersDictionary[Partindex])

            # Puts every musclepart values in a different Dictionary key
            MuscleOutput[MusclePart] = {}

            # Loads the muscle variables from a h5 file
            if FileType == "h5":

                # For every variable, loads the data of every part of the muscle
                for VariableName in VariableNames:

                    # Gets the path of the folder that contains the variable
                    MuscleFolderPath = MuscleVariableDictionary[VariableName]["MuscleFolderPath"]

                    MusclePath = MuscleFolderPath + "." + AnybodyMusclePart

                    # if an AnybodyVariableName is entered, adds it to the path
                    # If AnybodyVariableName=="", it means the variable charged is named MuscleFolderPath.AnybodyMuscleName
                    if MuscleVariableDictionary[VariableName]["AnybodyVariableName"]:
                        MuscleVariablePath = MusclePath + "." + MuscleVariableDictionary[VariableName]["AnybodyVariableName"]

                    # Gets the loading options of the variable other than its path
                    variable_loading_options = MuscleVariableDictionary[VariableName].copy()
                    del variable_loading_options["AnybodyVariableName"]

                    # Loads the muscle part variables
                    MuscleOutput[MusclePart][VariableName] = LoadAnyVariable(h5File, MuscleVariablePath, MusclePath=MusclePath, **variable_loading_options)

            # # Loads the muscle variables from a AnyFileOut file
            # else:
            #     # For every variable, loads the data of every part of the muscle
            #     for VariableName in VariableNames:

            #         MuscleVariablePath = MuscleFolderPath + "." + AnybodyMusclePart + "." + AnybodyVariableNames[index]

            #         # Loads the muscle part variables
            #         MuscleOutput[MusclePart][VariableName] = LoadAnyFileOutVariable(
            #             h5File, FileType, MuscleVariablePath)

    return MuscleOutput


def LoadMuscleDictionary(h5File, MuscleDictionary, MuscleVariableDictionary, FileType="h5"):
    """
    Loads the muscles variables stored in a .anydata.h5 file or a AnyFileOut file

    FileType : h5 or AnyFileOut

    The muscles to load, the number of parts to load, the part string and the name of the muscle are stored in the Dictionary:

    MuscleDictionary =  {"MuscleName1": ['MuscleFolderPath', 'AnybodyMuscleName', 'PartString', [PartNumber]],
                         "MuscleName2": ['MuscleFolderPath', 'AnybodyMuscleName', 'PartString', [PartNumbers]]}

                         To make a more complex selection, a muscle loaded can be a combination of multiple muscles in anybody with different names
                         The muscle informations are combined in a list [MuscleInformations_1, MuscleInformations_2]
                             MuscleDictionary = {"MuscleName": [[, 'AnybodyMuscleName_1', 'PartString_1', [PartNumbers_1]],
                                                                [, 'AnybodyMuscleName_2', 'PartString_2', [PartNumbers_2]]
                                                                ]}

                                     VariablePath = Output.Model.Right.ShoulderArm.Mus.Supraspinatus_1

                                     If a shortcut to the Seg directory was created (AnyFolder &Mus = Main.HumanModel.BodyModel.Right.ShoulderArm.Mus;)
                                     VariablePath = Output.Mus.Supraspinatus_1

                        PartString : the string that seperate the muscle name from the musclepart number
                                   : Exemple : supraspinatus_2 PartString = "_"

                       [PartNumbers] : List to select the parts to load
                                             : To load multiple parts [PartNumbersList] = [FirstPart, LastPart]
                                                 : FirstPart = the number of the first part to load (generally 1)
                                                 : LastPart = the number of the last part to load
                                                 Example : [PartNumbers] = [1,3] will load the part 1,2 and 3 of the muscle

                                             : To load muscle with only one part without a number [PartNumbers] = []
                                             : To select only the 3rd part of a muscle [PartNumbers] = [3]

                        Ex: to load the deltoideus lateral (Called deltoideus_lateral_part_n in anybody) that has 4 parts and give it the name Deltoideus Lateral
                            the supraspinatus (Called supraspinatus_n) that has 6 parts and give it the name Supraspinatus
                            the biceps brachii longum (named biceps_brachii_caput_longum) that has only 1 part
                            the middle trapezius (part 1 to 3 of the trapezius_scapular_n)


                            MuscleDictionary = {"deltoideus lateral":["deltoideus_lateral","_part_", [1, 4]],
                                                "Supraspinatus": ["supraspinatus","_", [1, 6]],
                                                "middle trapezius": ["trapezius_scapular","_", [1, 3]],
                                                "biceps brachii longum": ["biceps_brachii_caput_longum", "", []]
                                                }

                        Ex : To load the pectoralis major clavicular and sternal as the pectoralis major :
                           MuscleDictionary = {"Pectoralis major": [["pectoralis_major_thoracic", "_part_", [1, 10]],
                                                                    ["pectoralis_major_clavicular", "_part_", [1, 5]]
                                                                    ]}

    MuscleVariableDictionary : Dictionary that has the same structure as VariableDictionary and with AnybodyVariableName (the name of the variable in the AnybodyMusclePath in MuscleDictionary) (CHANGE IN FUTURE)
                             : Can add an entry (list) "combine_muscle_part_operation" to controlthe way of combining the muscle with several muscle parts

                             : MuscleFolderPath : The path of the directory where the muscle variables are stored (FROM THE OUTPUT DIRECTORY)
                                              : Ex1: The muscle variable Ft is in the folder   : Main.HumanModel.BodyModel.Right.ShoulderArm.Mus.MuscleName.Ft
                                              : so MuscleFolderPath = Output.HumanModel.BodyModel.Right.ShoulderArm.Mus

                                              : Ex2 : A custom muscle variable (MomentArm) is calculated and results are stored on a folder : Main.

                            : "combine_muscle_part_operations" : ["combining_operation_1", "combining_operation_2"...]
                                                                 : Muscle parts are combined in a variable named : MuscleName, that combines every variable of every muscle parts.
                                                                 : the combining operations are :
                                                                     : (Default) : "total" sums the variable between all muscle parts
                                                                     : "max" finds the maximum of the variable between all muscle parts
                                                                     : "min" finds the minimum of the variable between all muscle parts
                                                                     : "mean" calculates the average of the variable between all muscle parts

                              : Exemple : Load Fm in newton, with a combined muscle with the total and the mean of every muscle parts
                                          Load Ft in newton, with a combined muscle with only the total (done by default) of every muscle parts
                                          Load CorrectedActivity in % and rename it Activity and calculates the maximal activity between every muscle part

                                          MuscleVariableDictionary = {"Fm": {"MuscleFolderPath": "Output.Model.HumanModel.Right.ShoulderArm.Mus",
                                                                             "AnybodyVariableName": "Fm", "VariableDescription": "Force musculaire [Newton]", "combine_muscle_part_operations" : ["max", "mean"]},

                                                                      "Ft": {"MuscleFolderPath": "Output.Model.HumanModel.Right.ShoulderArm.Mus",
                                                                             "AnybodyVariableName": "Ft", "VariableDescription": "Force dans le tendon [Newton]"},

                                                                      "Activity": {"MuscleFolderPath": "Output.Model.HumanModel.Right.ShoulderArm.Mus",
                                                                                   "AnybodyVariableName": "CorrectedActivity","VariableDescription": "Activité Musculaire [%]", "MultiplyFactor": 100, "combine_muscle_part_operations" : ["max"]}
                                                                      }

    """

    Muscles = {}
    # Parcours le dictionnaire
    for MuscleName, MuscleInformations in MuscleDictionary.items():

        # Si on veut sélectionner plusieurs muscles ayant des noms différents dans anybody ou faire une sélection de parties plus complexe
        if isinstance(MuscleInformations[0], list):

            # initialise le dictionnaire de sortie
            Muscles[MuscleName] = {}

            # Initialise le nombre de parties de la liste précédemment chargée à 0
            previous_last_part_number = 0

            # Parcours chaque liste contenant les informations de muscle
            for LineNumber in range(0, len(MuscleInformations)):

                # Sélectionne le nom complet du dossier dans lequel le muscle est situé sur Anybody
                AnybodyMuscleName = MuscleInformations[LineNumber][0]

                # Sélectionne la chaine de charactère avant le numéro de partie
                PartString = MuscleInformations[LineNumber][1]

                # Sélectionne les nombre des parties sélectionnées dans Anybody
                AnybodyPartNumbers = MuscleInformations[LineNumber][2]

                # Sélectionne le nombre de parties sélectionnées dans ce muscle
                NumberOfParts = MuscleInformations[LineNumber][3]

                # List that contains the first part number and the last part number to give in the dictionary
                # increments the number of parts to give by previous_last_part_number so the muscleparts of the next list have the right numbers
                # if the last list stopped at muscle_part_number = 5, previous_last_part_number=5 so the current list will start at 5+1=6
                PartNumbers_in_Dictionary = [1 + previous_last_part_number, NumberOfParts + previous_last_part_number]

                # Stocke le numéro de la dernière partie chargée pour incrémenter les numéros de partie de la prochaine liste
                # previous_last_part_number = NumberOfParts
                previous_last_part_number = PartNumbers_in_Dictionary[1]

                # Met le muscle dans un dossier en fonction du nom choisi et des numéros de partie choisis
                Current_Muscle_selection = LoadMuscle(h5File, AnybodyMuscleName, MuscleName, PartString, AnybodyPartNumbers, MuscleVariableDictionary, PartNumbers_in_Dictionary, FileType)

                # Adds the current muscle selection to the muscle dictionary containing the previously loaded muscle parts
                Muscles[MuscleName] = {**Muscles[MuscleName], **Current_Muscle_selection}

        # Dans le cas où on ne sélectionne que des muscles ayant le même nom dans Anybody, donc pas une liste de plusieurs muscles
        else:

            # Sélectionne le nom complet du dossier dans lequel le muscle est situé sur Anybody
            AnybodyMuscleName = MuscleInformations[0]

            # Sélectionne la chaine de charactère avant le numéro de partie
            PartString = MuscleInformations[1]

            # Sélectionne les nombre des parties sélectionnées dans Anybody
            AnybodyPartNumbers = MuscleInformations[2]

            # Sélectionne le nombre de parties sélectionnées dans ce muscle
            NumberOfParts = MuscleInformations[3]

            # List that contains the first part number and the last part number to give in the dictionary
            PartNumbers_in_Dictionary = [1, NumberOfParts]

            # Met le muscle dans un dossier en fonction du nom choisi et des numéros de partie choisis
            Muscles[MuscleName] = LoadMuscle(
                h5File, AnybodyMuscleName, MuscleName, PartString, AnybodyPartNumbers, MuscleVariableDictionary, PartNumbers_in_Dictionary, FileType)

        # Creates an entry in the dictionary that will store all the muscle parts variables combined
        Muscles[MuscleName] = combine_muscle_parts(Muscles[MuscleName], MuscleName, MuscleVariableDictionary)

    return Muscles


def combine_muscle_parts(MuscleOutput, MuscleName, MuscleVariableDictionary):
    """
    function that combines all muscle fibers variables of a muscle and adds an entry to the dictionary that will store these combined variables

    MuscleOutput : dict : Contains every variable of every part of the current muscle

    MuscleName : str : Name to give to the muscle in the output dictionary

    MuscleVariableDictionary : dict : Contains the informations of the muscle variable to load

    the way of combining the variable depends on the values on the list in the entry of MuscleVariableDictionary named : "combine_muscle_part_operations"
    "combine_muscle_part_operations" : ["combining_operation_1", "combining_operation_2"...]
                                     : the combining operations are
                                         : (Default) : "total" sums the variable between all muscle parts
                                         : "max" finds the maximum of the variable between all muscle parts
                                         : "min" finds the minimum of the variable between all muscle parts
                                         : "mean" calculates the average of the variable between all muscle parts

    ----------------------------------------
    return
    MuscleOutput = MuscleOutput in entry with an added entry with all combined muscle part variables

    For a muscle with only one part, it will overwrite the muscle variables as if the muscle was combined to match the structure of other muscles with multiple parts

    Combined variables component naming :
        All the component of each variables are combined.
        When the total of a muscle part is combined, its name is the name of the operation (operation="max" --> combined_total_component_name="Max")

        For the other component, if there is only one combining operation, its name stays the same ("x" --> "x")
        If there are multiple component and multiple combining operations, combined components are named "operation" + "_component_name"

        Ex : "combine_muscle_part_operations": ["total", "mean"] for a variable with components ["x", "y"]
           : the combined muscle will have the component sequence ["Total", "Mean", "Total_x", "Total_y", "Mean_x", "Mean_y"]
    """

    combined_MuscleOutput = {}

    # number of parts
    number_of_parts = len(MuscleOutput)

    first_muscle_part_name = list(MuscleOutput.keys())[0]

    nstep = len(MuscleOutput[first_muscle_part_name][list(MuscleVariableDictionary.keys())[0]]["Total"])

    for Variable_Name in MuscleVariableDictionary:

        # only variable dictionary can be combined, not matrices. So by default this variable won't get combined
        if isinstance(MuscleOutput[first_muscle_part_name][Variable_Name], np.ndarray):
            if "combine_muscle_part_operations" in MuscleVariableDictionary[Variable_Name]:
                raise ValueError(f"The muscle variable '{Variable_Name}' is a matrix, it cannot be combined. 'combine_muscle_part_operations' shouldn't be used for this variable.")
            else:
                continue

        # Gets the seqence and the descriptions of the variable
        Sequence_Composantes = MuscleOutput[first_muscle_part_name][Variable_Name]["SequenceComposantes"]
        Variable_description = MuscleOutput[first_muscle_part_name][Variable_Name]["Description"]

        # Gets the combine muscle operation list
        operations = MuscleVariableDictionary[Variable_Name].get("combine_muscle_part_operations", ["total"])

        # Stores the description of the combined muscle variable
        combined_MuscleOutput[Variable_Name] = {"SequenceComposantes": [], "Description": Variable_description}

        for Composante in Sequence_Composantes:
            composantes_value = np.zeros([nstep, len(MuscleOutput)])

            # gets every value of this variable between every muscle_part
            for muscle_part_index, muscle_part in enumerate(MuscleOutput):
                composantes_value[:, muscle_part_index] = MuscleOutput[muscle_part][Variable_Name][Composante]

            # if len(operations) == 1:
            for operation in operations:

                # if the Total is the component that is combined, the name of the combined component is the name of the operation
                if Composante == "Total":
                    combined_composante_name = operation.capitalize()
                # For only one operation, the names of the combined_component are the same then the original
                elif len(operations) == 1:
                    combined_composante_name = Composante
                # for several operations, the name of the combined components names (that are not the total) will be "Operation_"+"Composante_Name"
                else:
                    combined_composante_name = operation.capitalize() + "_" + Composante

                # Adds the combined_composante_name to the sequence of components
                combined_MuscleOutput[Variable_Name]["SequenceComposantes"].append(combined_composante_name)

                # For muscles with multiple parts
                if number_of_parts > 1:
                    if "total" == operation:
                        combined_MuscleOutput[Variable_Name][combined_composante_name] = np.sum(composantes_value, axis=1)

                    elif "max" == operation:
                        combined_MuscleOutput[Variable_Name][combined_composante_name] = np.max(composantes_value, axis=1)

                    elif "min" == operation:
                        combined_MuscleOutput[Variable_Name][combined_composante_name] = np.min(composantes_value, axis=1)

                    elif "mean" == operation:
                        combined_MuscleOutput[Variable_Name][combined_composante_name] = np.mean(composantes_value, axis=1)

                # for muscles with only one muscle part, doesn't do the calculations but copies the variables components to create the same structure as a muscle with multiple parts that was combined
                else:
                    combined_MuscleOutput[Variable_Name][combined_composante_name] = composantes_value.flatten()

    MuscleOutput[MuscleName] = combined_MuscleOutput

    return MuscleOutput


def LoadGHReactions(GHReactionsShape, h5File, MuscleVariableDictionary, RotG, PosG):
    """
    Loads the GHReactions informations :
    The EdgeMuscles and the CavityNodes Position in the local glenoid implant reference frame

    GHReactionsShape can be "Circle" or "Edge" depending on the type of the GHReactions used
    """
    CavityEdgeNode = {}

    # Initialise Muscle array as a (0,5) np.array
    EdgeMuscleDictionary = np.empty((0, 5), int)

    if GHReactionsShape == "Circle":
        EdgeMuscleDir = 'Output.Seg.Scapula.GlenImplantPos.GHReactionCenterNode.CavityEdgeNode'
    elif GHReactionsShape == "Edge":
        EdgeMuscleDir = 'Output.Seg.Scapula.GlenImplantPos.CavityEdgeNode'

    # Total number of edge muscles
    NumberEdgeMuscles = 8
    # Creates a MuscleArray to load the EdgeMuscles
    EdgeMuscleDictionary = {"Edge Muscle": ["Output.Jnt.MyGHReactions", "EdgeMuscle", "", NumberEdgeMuscles]}

    # Goes through every edgemuscles number
    for EdgeMuscleNumber in range(1, NumberEdgeMuscles + 1):

        # Gets the position of the EdgeMuscles insertion points in mm
        Pos = LoadAnyVariable(h5File, f"{EdgeMuscleDir}{EdgeMuscleNumber}.r", MultiplyFactor=1000, OutputDictionary=False)

        # Transforms the global position to the local coordinate system of the glenoid implant
        # Only takes the position at the first step because it is a constant

        PosLocal = transform_vector(Pos, RotG, PosG, inverse_transform=True)

        # Stores the local coordinate of this EdgeNode
        CavityEdgeNode["Cavity EdgeNode " + str(EdgeMuscleNumber)] = PosLocal[0, :]

    # Loads the Muscles variables
    GHReactions = LoadMuscleDictionary(h5File, EdgeMuscleDictionary, MuscleVariableDictionary, FileType="h5")

    # Adds the shape of the GHReaction
    GHReactions["Shape"] = GHReactionsShape

    # Adds the position of the edgenodes
    GHReactions["Cavity Nodes Position"] = CavityEdgeNode

    # Loads the strength of the muscle
    GHReactionsConstants = {"AnybodyFileOutPath": "Main.Study.FileOut",
                            "Informations": ["EdgeMuscleStrength"]
                            }
    # Loads the constants
    GHReactionsConstants = LoadAnyFileOut(h5File["FilePath"], GHReactionsConstants, LoadConstantsOnly=True)

    GHReactions = {**GHReactions, **GHReactionsConstants}

    return GHReactions
