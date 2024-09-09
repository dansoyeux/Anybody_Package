import numpy as np

from Anybody_Package.Anybody_LoadOutput.Tools import array_to_dictionary
from Anybody_Package.Anybody_LoadOutput.Tools import transform_vector
from Anybody_Package.Anybody_LoadOutput.Tools import combine_variable

import Anybody_Package.Anybody_LoadOutput.LoadAnybodyData as LoadAnybodyData


def load_simulation(FileDirectory, FileName, VariablesToLoad, Failed=False):
    """
    Reads variables from an anydata.h5 file and put it in a result dictionary

    Charge plusieurs h5 et les mets dans le même dictionnaire sous forme de cas de simulation

    FileDirectory : Type : string
                  : Chemin d'accès au dossier où le fichiers h5 sont situés

    FileName : Nom des fichiers à charger
             : Noms des fichier (sans extension)
             : pour un fichier Resultats.anydata.h5 FileName = "Resultats"

    VariablesToLoad : à Définir avec la fonction define_variables_to_load (Contient les variables à charger, les muscles et leur variables à charger et les constantes à charger)
                    : Si une des catégories (Variable, Constante ou muscle) n'est pas dans le dictionnaire, sa valeur sera None pour ne pas essayer de charger ces informations

    Sums the total variable for a muscle in multiple parts
    AddConstants : adds the constants that are not stored in the h5 file by reading them in the FileOut file
    Failed : removes the failed steps in case the simulation failed after a certain step

    Template to load a variable :
        Results["nom_de_variable"] = LoadAnybodyData.LoadAnyVariable(h5File, "Output.Chemin.accès.variable.dans.Anybody.depuis.l'output",MultiplyFactor = , VariableDescription = "Description de la variable [unité de la variable]")
    """

    # Chemin d'accès au fichier (sans l'extension)
    if FileDirectory == "":
        FilePath = FileName
    else:
        FilePath = FileDirectory + "/" + FileName

    h5File = LoadAnybodyData.Loadh5File(FilePath, Failed)

    Results = {}

    # Charge les variables seulement s'il y en a
    if "Variables" in VariablesToLoad:

        # Fais une copie pour que VariablesToLoad ne change pas si on effectue des changements sur ces dictionnaires dans les fonctions
        VariableDictionary = VariablesToLoad["Variables"].copy()

        for Variable in VariableDictionary:

            Results[Variable] = LoadAnybodyData.LoadAnyVariable(h5File, **VariableDictionary[Variable])

    # Section des muscles si les muscles sont dans VariablesToLoad
    if "Muscles" in VariablesToLoad:

        # Fais une copie pour que VariablesToLoad ne change pas si on effectue des changements sur ces dictionnaires dans les fonctions
        MuscleDictionary = VariablesToLoad["Muscles"].copy()
        MuscleVariableDictionary = VariablesToLoad["MuscleVariables"].copy()

        # Gets the muscles variables
        Muscles = LoadAnybodyData.LoadMuscleDictionary(h5File, MuscleDictionary, MuscleVariableDictionary, FileType="h5")

        # Met les informations des muscles dans les résultats
        Results["Muscles"] = Muscles

    # Charge les constantes s'il y en a dans VariablesToLoad
    if "Constantes" in VariablesToLoad:

        # Fais une copie pour que VariablesToLoad ne change pas si on effectue des changements sur ces dictionnaires dans les fonctions
        ConstantsDictionary = VariablesToLoad["Constantes"].copy()

        Constants = LoadAnybodyData.LoadAnyFileOut(FilePath, ConstantsDictionary, LoadConstantsOnly=True)

        Results["Model informations"] = Constants

    # calculates the number of steps from the length of the time
    VariablesToLoad["nStep"] = len(h5File["h5Data"]["Output/Abscissa/t"])

    # Ferme le fichier h5
    h5File["h5Data"].close()

    VariablesToLoad["Data Source"] = "Anybody"
    Results["Loaded Variables"] = VariablesToLoad

    return Results


def load_simulation_cases(FileDirectory, CasesFileNamesList, SimulationCasesNames, VariablesToLoad, Failed=False):
    """
    Charge plusieurs h5 et les mets dans le même dictionnaire sous forme de cas de simulation

    FileDirectory : Type : string
                  : Chemin d'accès au dossier où les fichiers h5 sont situés
                  : Les fichiers doivent être dans le même dossier

    CasesFileNamesList : Liste des noms des fichiers à charger
             : Noms des fichier (sans extension)
             : pour un fichier Resultats.anydata.h5 FileName = "Resultats"


    SimulationCasesNames : type liste
                         : Nom des cas de simulation à mettre comme clé de dictionnaire

    Failed : List containing the first Failed step of each simulation, or False if the simulation didn't failed

    Exemple 1 : FileNamesList = ['nom__cas1','nom_cas2','nom__cas3']
              : SimulationCasesNames = ['Cas 1','Cas 2','Cas 3']

    Exemple 2 : Les cas de simulations vont de 1 à 3 mais cette simulation n'a que les cas 1 et 2
              : On met donc le second nom de fichier à '' pour signifier que cette simulation n'a pas de cas 2
              : CasesFileNamesList = ['nom__cas1','','nom__cas3']
              : SimulationCasesNames = ['Cas 1','Cas 2','Cas 3']

    Exemple 3 : On remplit les cas de simulation au fur et à mesure. Il y a 3 cas mais le cas 3 n'a pas encore été simulé :
              : SimulationCasesNames = ['Cas 1','Cas 2','Cas 3']
              : CasesFileNamesList = ['nom__cas1','nom__cas2',''] ou ['nom__cas1','nom__cas2']
    """

    # Si aucun cas n'a fail, crée une liste remplie de Failed = False
    if Failed is False:
        Failed = [False] * len(CasesFileNamesList)

    Results = {}
    # Parcours le nom des cas de simulation et charge les fichiers h5 correspondants
    for index in range(len(CasesFileNamesList)):

        # Crée un cas de simulation seulement si un fichier existe pour ce cas
        if not CasesFileNamesList[index] == '':
            Results[SimulationCasesNames[index]] = load_simulation(FileDirectory, CasesFileNamesList[index], VariablesToLoad, Failed[index])

    return Results


def create_compared_simulations(simulation_names, *args):
    """
    Assemble plusieurs simulations avec cas de simulation en comparaison de simulations

    simulation_names : list : liste des noms des simulations

    ensuite, mettre les dictionnaires de résultats avec cas de simulation à fusionner

    """
    # check that the number of the simulations names and the number of dictionary to compare match
    if not len(simulation_names) == len(args):
        raise ValueError(f"The number of the simulations names ({len(simulation_names)}) and the number of dictionary to compare ({len(args)}) don't match")
        return

    compared_simulations = {}

    for index, simulation_data in enumerate(args):

        # checks that the current result dictionary has simulation cases
        if "Loaded Variables" in simulation_data:
            raise ValueError(f"The simulation number {index + 1} must have simulation cases to be assembled as a compared simulations dictionary")
            return

        compared_simulations[simulation_names[index]] = simulation_data

    return compared_simulations


def define_variables_to_load(VariableDictionary=None, MuscleDictionary=None, MuscleVariableDictionary=None, ConstantsDictionary=None):
    """
    Function to build the Dictionary that will store which variable to load from an h5File

    VariableDictionary = {
                           "VARIABLE_1_NAME": {"VariablePath": "VARIABLE_1_PATH", "VariableDescription": "VARIABLE_1_DESCRIPTION", "MultiplyFactor": VARIABLE_1_MULTIPLY_FACTOR, "SequenceComposantes": ["VARIABLE_1_SEQUENCE_COMPOSANTE"]},
                           "VARIABLE_2_NAME": {"VariablePath": "VARIABLE_2_PATH", "VariableDescription": "VARIABLE_2_DESCRIPTION"}
                                    }
                            VariablePath : The path of the variable (IN THE OUTPUT DIRECTORY)
                                         : Ex: The variable Main.HumanModel.BodyModel.Right.Model.ShoulderArm.Seg.Scapula.ghProth.Axes
                                               will be stored after the simulation in the directory Main.Study.Output.Model.Right.ShoulderArm.Seg.Scapula.ghProth.Axes

                                         VariablePath = Output.Right.ShoulderArm.Seg.Scapula.ghProth.Axes

                                         If a shortcut to the Seg directory was created (AnyFolder &Seg= Main.HumanModel.BodyModel.Right.ShoulderArm.Seg;)
                                         VariablePath = Output.Seg.Scapula.ghProth.Axes

                            FilePath is the path of the file to load (it must begin with "Output." since it's the output data that is stored in an h5 file)
                            VariableDescription : Description of the variable that is going to be used in the axis label on the graph
                            SequenceComposantes : Indicates which colomns corresponds to which component (x,y or z)
                                                  The sequence is xyz by defaukt. So the first column will be x, then y then z
                            MultiplyFactor : If multiplying every value by a factor is needed (to convert m to mm for example)


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


    ConstantsDictionary : Stores the constants categories and also the Anybody path of the AnyFileOut in the Anybody tree
                           ConstantsDictionary = {"AnybodyFileOutPath": "ANYBODYFILEOUTPATH",
                                                  "CONSTANTS_CATEGORY": [LIST_OF_CATEGORY_CONSTANTS]}
    Ex : to have a mannequin variables category and a simulation parameters category
         When the AnyFileOutPath is Main.Study.FileOut

    ConstantDictionary =     {"AnybodyFileOutPath": "Main.Study.FileOut",
                              "Simulation_Parameters": ["Movement", "Case", "GHReactions", "nstep"],
                               "Mannequin": ["GlenohumeralFlexion", "GlenohumeralAbduction", "GlenohumeralExternalRotation"]}


    VariablesToLoad = {"Variables":VariableDictionary,
                       "Muscles": {"MuscleDictionary":MuscleDictionary, "MuscleVariables"}



    -------------------------------------------
    return

    VariablesToLoad = {"Variables": VariableDictionary,
                       "Muscles": MuscleDictionary,
                       "Constantes": ConstantsDictionary,
                       "MusclesVariables": MusclesVariableDictionary
                       }

    Dans MuscleDictionarry[MuscleName][4], on calcul le nombre de parties sélectionné
    Nombre_de_parties
    Ex: si on charge les parties [4, 6], 3 muscles sont chargés
        si on charge la parties [4], 1 muscle est chargé

    """

    if VariableDictionary is None:
        raise ValueError("At least one non-muscle variable must be charged in order to be able to produce graphs")

    VariablesToLoad = {"Variables": VariableDictionary}

    if MuscleDictionary:
        # Vérifie que les variables musculaires ont bien été définies
        if not MuscleVariableDictionary:
            raise ValueError("Muscles variables to load were not defined")
            return

        VariablesToLoad["Muscles"] = {}

        # Calcul et ajoute le nombre de parties de muscles qui seront créés plus tard
        # Parcours chaque muscles
        for MuscleName in MuscleDictionary:

            # Charge la liste qui contient les informations du muscle
            MuscleInformations = MuscleDictionary[MuscleName].copy()

            # if the muscle information is a with only one name in anybody
            if isinstance(MuscleInformations[0], str):

                # Liste contenant le premier et le dernier numéro de partie
                PartNumbers = MuscleInformations[2]

                # Si le muscle n'a pas de partie ou si une seule partie est sélectionnée stocke 1
                if PartNumbers == [] or len(PartNumbers) == 1:
                    MuscleInformations.append(1)

                # Sinon calcul le nombre de parties sélectionnées pour ce muscle
                else:
                    # informations sur les parties
                    FirstPart = PartNumbers[0]
                    LastPart = PartNumbers[1]

                    # Nombre de parties dans la sélection
                    NumberOfParts = LastPart - FirstPart + 1

                    MuscleInformations.append(NumberOfParts)

            # if the muscle is a selection of muscles with multiple names in anybody
            elif isinstance(MuscleInformations[0], list):

                # Parcours chaque liste contenant les informations de muscle
                for LineNumber in range(0, len(MuscleInformations)):

                    # Liste contenant le premier et le dernier numéro de partie
                    PartNumbers = MuscleInformations[LineNumber][2]

                    # Si le muscle n'a pas de partie ou si une seule partie est sélectionnée stocke 1
                    if PartNumbers == [] or len(PartNumbers) == 1:
                        MuscleInformations[LineNumber].append(1)

                    # Sinon calcul le nombre de parties sélectionnées pour ce muscle
                    else:
                        # informations sur les parties
                        FirstPart = PartNumbers[0]
                        LastPart = PartNumbers[1]

                        # Nombre de parties dans la sélection
                        NumberOfParts = LastPart - FirstPart + 1

                        MuscleInformations[LineNumber].append(NumberOfParts)

            VariablesToLoad["Muscles"][MuscleName] = MuscleInformations.copy()

        VariablesToLoad["MuscleVariables"] = MuscleVariableDictionary

    if ConstantsDictionary:
        # Vérifie que le chemin d'accès de l'objet AnyFileOut est bien spécifié
        if "AnybodyFileOutPath" not in ConstantsDictionary:
            raise ValueError("Définir le chemin anybody de l'objet AnyFileOut")
            return

        VariablesToLoad["Constantes"] = ConstantsDictionary

    return VariablesToLoad


# %% tools to modify previously charged result dictionaries


def combine_simulation_cases(result_dictionary, combine_cases, operation="mean"):
    """
    function that combines multiple simulation cases into one result dictionary
    Makes the average of each variable between the cases

    result_dictionary : dictionary : Result dictionary with simulation cases

    combine_cases : dict : combine_cases = {"combined_case_name_1": [list_of_cases_to_combine_1]
                                            "combined_case_name_2": [list_of_cases_to_combine_2]
                                            "combined_case_name_3": [list_of_cases_to_combine_3]
                                            }

    operation : str : name of the operation done
              : list of combination posible : ["mean", "total", "min" "max"]

    return

    combined_results : dict : dictionary with simulation the simulation cases : new_case_name_1, new_case_name_2, new_case_name_3
                        Example : for a result dictionary with simulation cases : case_1, case_2, case_3, case_4, case_5, case_6 we want the average of case_1 and case_3 to be put in a case named combined_1
                                : and case_2 and case_4 to be averaged a and put in combined_2, and keep case_5 as it was
                                : combine_cases = {"combined_1": ["case_1", "case_3"],
                                                   "combined_2": ["case_2", "case_4"],
                                                   "case_5": ["case_5"]
                                                   }

    """

    # Trouve la liste des variables normales, des variables musculaires et les noms des muscles
    loaded_variables = result_dictionary[list(result_dictionary.keys())[0]]["Loaded Variables"]

    Variables = loaded_variables["Variables"]
    MuscleVariables = loaded_variables["MuscleVariables"]
    Muscles = loaded_variables["Muscles"]

    combined_results = {}

    # Goes through each case group to combine
    for combined_case_name, cases_to_combine in combine_cases.items():

        # Does the combine operations only if there are multiple cases to combine
        if len(cases_to_combine) == 1:
            try:
                combined_results[combined_case_name] = result_dictionary[cases_to_combine[0]]

            except KeyError as exc:
                raise ValueError(f"The case '{exc}' in the category '{combined_case_name}' doesn't exist in the result dictionary")

        else:
            combined_results[combined_case_name] = {"Loaded Variables": loaded_variables, "Muscles": {}}

            # goes through each normal variable
            for variable_name in Variables:

                try:
                    # builds the list of the variable dictionaries to combine
                    variable_dictionaries_list = [result_dictionary[case][variable_name] for case in cases_to_combine]

                except KeyError as exc:
                    raise ValueError(f"The case '{exc}' in the category '{combined_case_name}' doesn't exist in the result dictionary")

                # combines the current variable
                combined_results[combined_case_name][variable_name] = combine_variable(variable_dictionaries_list, operation)

            # goes through each muscle
            for muscle_name in Muscles:
                combined_results[combined_case_name]["Muscles"][muscle_name] = {}

                # goes through each muscle part
                for muscle_part in result_dictionary[cases_to_combine[0]]["Muscles"][muscle_name]:
                    combined_results[combined_case_name]["Muscles"][muscle_name][muscle_part] = {}

                    # goes through each variable name
                    for muscle_variable_name in MuscleVariables:
                        # builds the list of the variable dictionaries to combine
                        muscle_part_variable_dictionaries_list = [result_dictionary[case]["Muscles"][muscle_name][muscle_part][muscle_variable_name] for case in cases_to_combine]

                        # combines the current muscle variable
                        combined_results[combined_case_name]["Muscles"][muscle_name][muscle_part][muscle_variable_name] = combine_variable(muscle_part_variable_dictionaries_list, operation)

    return combined_results


def sum_result_variables(data, summed_variable_name, summed_variable_sequence, summed_variable_description, variables_to_add=[], muscle_variables_to_add=[], muscles_to_add=[]):
    """
    function that sums multiple variables for all the data in the result dictionary
    It works for all data structures

    Each variable are summed depending on the order of the component names entered for each variables
    ALL VARIABLES MUST HAVE THE SAME NUMBER OF COMPONENTS THAN THE summed_variable_sequence

    data : data dictionary
    summed_variable_name : (str) the name of the variable that will store the summed variables
    combined_variable_sequence : (list) sequence of the of the variable that will store the summed variables

    variables_to_add : (dict) dictionary containing a list containing the names of the variables to sum and the order of the component to sum
                      variables_to_add = {"variable_1": [component_1, component_2, component_3],
                                          "variable_2": [component_1, component_2, component_3]}

    muscle_variables_to_add : (list) dictionary containing the names of the muscle variables to sum, the order of the component to sum and the name of the muscles to sum
                         for the first variable, only a list of muscles are summed, and vor the variable 2 all existing muscles are summed
                         muscle_variables_to_add = {"muscle_variable_1": [component_1, component_2, component_3]
                                                    "muscle_variable_2": [component_1, component_2, component_3]}

    muscles_to_add : (list) list of muscles to add

    -------------------------------------
    return

    data : data dictionary with the new variable combined_variable_name added
    """

    from Anybody_Package.Anybody_Graph.GraphFunctions import get_result_dictionary_data_structure

    # gets the data structure
    data_structure_counter, data_source = get_result_dictionary_data_structure(data)

    def sum_variables(data, summed_variable_name, summed_variable_sequence, summed_variable_description, variables_to_add=[], muscle_variables_to_add=[], muscles_to_add=[], case_name_error=""):
        """
        function that sums multiple variables for a single variable dictionary (only one simulation case)
        It works for all data structures

        we can select a list of variable to combine, a list of muscle variables to combine and muscle to include in this combination


        Each variable are summed depending on the order of the component names entered for each variables
        ALL VARIABLES MUST HAVE THE SAME NUMBER OF COMPONENTS THAN THE summed_variable_sequence

        data : data dictionary
        summed_variable_name : (str) the name of the variable that will store the summed variables
        combined_variable_sequence : (list) sequence of the of the variable that will store the summed variables

        variables_to_add : (dict) dictionary containing a list containing the names of the variables to sum and the order of the component to sum
                          variables_to_add = {"variable_1": [component_1, component_2, component_3],
                                              "variable_2": [component_1, component_2, component_3]}

        muscle_variables_to_add : (list) dictionary containing the names of the muscle variables to sum, the order of the component to sum and the name of the muscles to sum
                             for the first variable, only a list of muscles are summed, and vor the variable 2 all existing muscles are summed
                             muscle_variables_to_add = {"muscle_variable_1": [component_1, component_2, component_3]
                                                        "muscle_variable_2": [component_1, component_2, component_3]}
        muscles_to_add : (list) list of muscles to add

        case_name_error : str string to add the name of the simulation case in the error message

        -------------------------------------
        return

        data : data dictionary with the new variable combined_variable_name added
        """

        # Gets the number of simulation steps of this data
        nStep = data["Loaded Variables"]["nStep"]

        # initializes an array for each component
        summed_variable_data = {component: np.zeros(nStep) for component in summed_variable_sequence}

        # Stores the component sequence and the summed variable description
        data[summed_variable_name] = {**summed_variable_data, "SequenceComposantes": summed_variable_sequence, "Description": summed_variable_description}

        # gets the number of output variables
        number_component = len(summed_variable_sequence)

        # Sums the normal variables
        if variables_to_add:
            # goes through each variable
            for variable_name, component_variable in variables_to_add.items():

                # Checks that the number of components to sum match between the variable and the output variable
                if not len(component_variable) == number_component:
                    raise ValueError(f"The number of components to add for the variable '{variable_name}' is not the same than the summed_variable_sequence\n ({len(component_variable)} > {number_component})")
                # Checks that the variable exists
                if variable_name not in data:
                    raise ValueError(f"The variable '{variable_name}' doesn't exist in the entered result dictionary{case_name_error}")

                # goes through every component to add
                for component_index, component_name in enumerate(component_variable):
                    # Checks that the component exists
                    if component_name not in data[variable_name]:
                        raise ValueError(f"The component '{component_name}' of the variable '{variable_name}' doesn't exist in the entered result dictionary{case_name_error}")

                    # name of the output component
                    summed_component_name = summed_variable_sequence[component_index]

                    # Adds the current component to the data stored in the summed variable data
                    summed_variable_data[summed_component_name] += data[variable_name][component_name]

        # Sums the muscle variable
        if muscle_variables_to_add:

            # Goes through each variable
            for variable_name, component_variable in muscle_variables_to_add.items():

                # # Gets the informations of the current summed variable
                # component_variable = variable_to_add_informations["component_variable"]

                # Checks that the number of components to sum match between the variable and the output variable
                if not len(component_variable) == number_component:
                    raise ValueError(f"The number of components to add for the variable '{variable_name}' is not the same than the summed_variable_sequence\n ({len(component_variable)} > {number_component})")

                # Select all muscles to add if "all" was entered
                if muscles_to_add == "all":
                    muscles_to_add = list(data["Muscles"].keys())

                for muscle_name in muscles_to_add:

                    # Checks that the muscle exists
                    if muscle_name not in data["Muscles"]:
                        raise ValueError(f"The muscle '{muscle_name} doesn't exist in the entered result dictionary{case_name_error}")

                    # checks that the variable exists for this muscle
                    if variable_name not in data["Muscles"][muscle_name][muscle_name]:
                        raise ValueError(f"For the muscle '{muscle_name}', the muscle variable '{variable_name}' doesn't exist in the entered result dictionary{case_name_error}")

                    # goes through every component to add
                    for component_index, component_name in enumerate(component_variable):
                        # Checks that the component exists
                        if component_name not in data["Muscles"][muscle_name][muscle_name][variable_name]:
                            raise ValueError(f"For the muscle '{muscle_name}', the component '{component_name}' of the variable '{variable_name}' doesn't exist in the entered result dictionary{case_name_error}")

                        # name of the output component
                        summed_component_name = summed_variable_sequence[component_index]

                        # Adds the current component to the data stored in the summed variable data
                        summed_variable_data[summed_component_name] += data["Muscles"][muscle_name][muscle_name][variable_name][component_name]

        return data

    # for a ResultDictionary without any simulation case
    if data_structure_counter == 0:
        data = sum_variables(data, summed_variable_name, summed_variable_sequence, summed_variable_description, variables_to_add, muscle_variables_to_add, muscles_to_add)

    # For a result dictionary with simulation cases
    elif data_structure_counter == 1:
        # sums the cases variables one by one
        for case_name, case_data in data.items():
            data[case_name] = sum_variables(data[case_name], summed_variable_name, summed_variable_sequence, summed_variable_description, variables_to_add, muscle_variables_to_add, muscles_to_add, case_name_error=f" '{case_name}'")

    # for compared simulation cases
    else:
        # sums the cases simulation by simulation
        for simulation_name, simulation_data in data.items():

            for case_name, case_data in simulation_data.items():
                data[simulation_name][case_name] = sum_variables(data[simulation_name][case_name], summed_variable_name, summed_variable_sequence, summed_variable_description, variables_to_add, muscle_variables_to_add, muscles_to_add, case_name_error=f" '{simulation_name}/{case_name}'")

    return data
