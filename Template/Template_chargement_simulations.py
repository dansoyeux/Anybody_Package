# import Anybody_LoadOutput.LoadOutput as LoadOutput
# import Anybody_Tools as LoadOutputTools

from Anybody_Package.Anybody_LoadOutput.LoadOutput import DefineVariablesToLoad
from Anybody_Package.Anybody_LoadOutput.LoadOutput import DefineVariablesToLoad
from Anybody_Package.Anybody_LoadOutput.LoadOutput import LoadSimulationCases
from Anybody_Package.Anybody_LoadOutput.LoadOutput import LoadResultsh5
from Anybody_Package.Anybody_LoadOutput.LoadOutput import create_compared_simulations

from Anybody_Package.Anybody_LoadOutput.Tools import SaveVariableToFile
from Anybody_Package.Anybody_LoadOutput.Tools import ArrayToDictionary

import numpy as np

import pandas as pd

# %% Setup des variables à charger

# Muscles
MuscleDictionary = {"Deltoideus lateral": ["deltoideus_lateral", "_part_", [1, 4]],
                    "Deltoideus posterior": ["deltoideus_posterior", "_part_", [1, 4]],
                    "Deltoideus anterior": ["deltoideus_anterior", "_part_", [1, 4]],
                    "Supraspinatus": ["supraspinatus", "_", [1, 6]],
                    "Infraspinatus": ["infraspinatus", "_", [1, 6]],
                    "Serratus anterior": ["serratus_anterior", "_", [1, 6]],
                    "Lower trapezius": ["trapezius_scapular", "_part_", [1, 3]],
                    "Middle trapezius": ["trapezius_scapular", "_part_", [4, 6]],
                    "Upper trapezius": ["trapezius_clavicular", "_part_", [1, 6]],
                    "Biceps brachii long head": ["biceps_brachii_caput_longum", "", []],
                    "Biceps brachii short head": ["biceps_brachii_caput_breve", "", []],
                    "Pectoralis major clavicular": ["pectoralis_major_clavicular", "_part_", [1, 5]],
                    "Pectoralis major sternal": ["pectoralis_major_thoracic", "_part_", [1, 10]],

                    "Pectoralis major": [["pectoralis_major_thoracic", "_part_", [1, 10]],
                                         ["pectoralis_major_clavicular", "_part_", [1, 5]]
                                         ],

                    "Pectoralis minor": ["pectoralis_minor", "_", [1, 4]],
                    "Latissimus dorsi": ["latissimus_dorsi", "_", [1, 11]],
                    "Triceps long head": ["Triceps_LH", "_", [1, 2]],
                    "Upper Subscapularis": ["subscapularis", "_", [1, 2]],
                    "Downward Subscapularis": ["subscapularis", "_", [3, 6]],
                    "Subscapularis": ["subscapularis", "_", [1, 6]],
                    "Teres minor": ["teres_minor", "_", [1, 6]],
                    "Teres major": ["teres_major", "_", [1, 6]],
                    "Rhomboideus": ["rhomboideus", "_", [1, 3]],
                    "Levator scapulae": ["levator_scapulae", "_", [1, 4]],
                    "Sternocleidomastoid clavicular": ["Sternocleidomastoid_caput_clavicular", "", []],
                    "Sternocleidomastoid sternum": ["Sternocleidomastoid_caput_Sternum", "", []],
                    "Coracobrachialis": ["coracobrachialis", "_", [1, 6]]
                    }

MuscleVariableDictionary = {"Fm": {"MuscleFolderPath": "Output.Mus", "AnybodyVariableName": "Fm", "VariableDescription": "Force musculaire [Newton]"},
                            "Ft": {"MuscleFolderPath": "Output.Mus", "AnybodyVariableName": "Ft", "VariableDescription": "Force musculaire totale [Newton]"},
                            "Activity": {"MuscleFolderPath": "Output.Mus", "AnybodyVariableName": "CorrectedActivity", "VariableDescription": "Activité Musculaire [%]", "MultiplyFactor": 100, "combine_muscle_part_operations": ["max", "mean"]},

                            "F origin": {"MuscleFolderPath": "Output.Mus", "AnybodyVariableName": "RefFrameOutput.F", "VariableDescription": "Force Musculaire à l'origine du muscle [N]", "select_matrix_line": 0,
                                         "rotation_matrix_path": "Output.Seg.Scapula.AnatomicalFrame.ISB_Coord.Axes", "inverse_rotation": True, "SequenceComposantes": ["AP", "IS", "ML"],
                                         "combine_muscle_part_operations": ["total", "mean"]},

                            "F insertion": {"MuscleFolderPath": "Output.Mus", "AnybodyVariableName": "RefFrameOutput.F", "VariableDescription": "Force Musculaire à l'insertion du muscle [N]", "select_matrix_line": 1,
                                            "rotation_matrix_path": "Output.Seg.Scapula.AnatomicalFrame.ISB_Coord.Axes", "inverse_rotation": True, "SequenceComposantes": ["AP", "IS", "ML"],
                                            "combine_muscle_part_operations": ["total", "mean"]},

                            # "MomentArm": {"MuscleFolderPath": "Output.Mus", "AnybodyVariableName": "MomentArm", "VariableDescription": "Bras de levier du muscle [mm]",
                            #                  "combine_muscle_part_operations": ["mean"], "MultiplyFactor": 1000}
                            }

# Variables
VariableDictionary = {"Abduction": {"VariablePath": "Output.rotD", "VariableDescription": "Angle d'abduction [°]"},
                      "GHLin": {"VariablePath": "Output.Jnt.GHLin.Pos", "VariableDescription": "Déplacement Linéaire de l'humérus [mm]", "MultiplyFactor": 1000, "SequenceComposantes": ["AP", "IS", "ML"]},
                      }


# Constantes (si un fichier AnyFileOut contenant des constantes est créé en même temps que le fichier h5)
# Constantes
ConstantsDictionary = {"AnybodyFileOutPath": "Main.Study.FileOut",  # CHEMIN D'ACCÈS ANYBODY DE L'OBJET AnyFileOut
                       "Paramètres de simulation": ["Case", "MuscleRecruitment", "nStep", "tEnd", "GHReactions", "Movement"],
                       "Mannequin": ["GlenohumeralFlexion", "GlenohumeralAbduction", "GlenohumeralExternalRotation"]
                       }

BallAndSocket_ConstantsDictionary = {"AnybodyFileOutPath": "Main.Study.FileOut",
                                     "Paramètres de simulation": ["Case", "MuscleRecruitment", "nstep", "GHReactions", "Movement"],
                                     "Mannequin": ["GlenohumeralFlexion", "GlenohumeralAbduction", "GlenohumeralExternalRotation"],
                                     }


Variables = DefineVariablesToLoad(VariableDictionary, MuscleDictionary, MuscleVariableDictionary, ConstantsDictionary)


# %% Chargement des fichiers .h5

"""Chemin d'accès au dossier où sont sauvegardes les fichiers h5"""
SaveDataDir = r"Fichiers h5/"

# Chemin d'accès au dossier dans lequel les fichiers doivent être sauvegardés
SaveSimulationsDirectory = "Saved Simulations"

# Nom des fichiers .h5 (sans l'extension anydata.h5)
Files = ["NOM_FICHIER_1",
         "NOM_FICHIER_2"
         ]

# Noms des simulations
CaseNames = ["NOM_DU_CAS_1",
             "NOM_DU_CAS_2"
             ]

Results = LoadSimulationCases(SaveDataDir, Files, CaseNames, Variables)

# Sauvegarde des résultats dans des fichiers .pkl
SaveVariableToFile(Results, SaveSimulationsDirectory, "Results")

# %% Sauvegarde des dictionnaires de variables

# # Chemin d'accès au dossier dans lequel les fichiers doivent être sauvegardés
SaveVariablesDirectory = "Saved VariablesDictionary"

SaveVariableToFile(Variables, SaveVariablesDirectory, "Variables")

# %% Sauvegarde data bergmann


def LoadBergmannData():
    BodyWeight = 75 * 9.81

    abduction = np.array([15, 30, 45, 75]).T

    # ForceContact = np.array([21, 35, 51, 85]).T / 100 * BodyWeight

    ForceContact_x = np.array([7.5, 13, 21, 34]).T
    ForceContact_y = np.array([-19, -31, -44, -74]).T
    ForceContact_z = np.array([4.5, 8.5, 16, 25]).T

    ForceContact = np.array([ForceContact_x, ForceContact_y, ForceContact_z]).T
    ForceContact = ForceContact / 100 * BodyWeight

    dataBergmann = {}

    dataBergmann["Abduction"] = ArrayToDictionary(abduction, VariableDescription="Angle d'abduction [°]")
    dataBergmann["ForceContact"] = ArrayToDictionary(ForceContact, VariableDescription="Force de contact [Newton]", SequenceComposantes=["AP", "IS", "ML"])

    return dataBergmann

# dataBergmann_2007 = LoadBergmannData()
# SaveVariableToFile(dataBergmann_2007, SaveSimulationsDirectory, "dataBergmann_2007")

# %% Data muscles Wickham


def load_data_wickham(SheetName, MinAngle, MaxAngle):

    from scipy.interpolate import UnivariateSpline

    File = pd.ExcelFile('Data Wikham et al/dataWickham.xlsx')

    dataWickham_raw = pd.read_excel(File, SheetName)

    Variables = pd.Series.to_numpy(dataWickham_raw.iloc[0, :])
    data = dataWickham_raw
    data = data.drop(index=[0])

    # Get muscles names
    Muscles = dataWickham_raw.columns.tolist()

    dataWickham = {}

    # Creates angles so that all the variables have the same angles
    # Will interpolate the datas with this angles
    Interpolated_Angle = np.linspace(MinAngle, MaxAngle, 100)

    # Convert these datas to dictionnary
    AngleDictionary = ArrayToDictionary(Interpolated_Angle, VariableDescription="Angle d'abduction [°]")

    for index in range(0, len(Variables)):

        MuscleName = Muscles[index].replace(".1", "")

        Array = pd.Series(data.iloc[:, index]).dropna().to_numpy()
        Array = Array.astype('float')

        Variable = Variables[index]
        if Variable == "Abduction":
            Angle = Array

        # Once we reach activity, interpolates the data to have the activity for the Angles specified in Interpolated_Angle
        elif Variable == "Activity":
            Activity = Array

            # Interpolates the angle so that every data have the same angles
            Interpolation_Function = UnivariateSpline(Angle, Activity, s=5)

            Interpolated_Activity = Interpolation_Function(Interpolated_Angle)

            ActivityDictionary = ArrayToDictionary(Interpolated_Activity, VariableDescription="Activité Musculaire [%]", MaximumOn=True)

            dataWickham[MuscleName] = {MuscleName: {"Activity": ActivityDictionary}}

    File.close()

    Results = {"Muscles": dataWickham}
    Results["Abduction"] = AngleDictionary

    return Results


# # Abduction
# dataWickham_abduction = load_data_wickham("Abduction", 15, 120)
# SaveVariableToFile(dataWickham_abduction, SaveSimulationsDirectory, "dataWickham_abduction")

# # Abduction Long Range
# dataWickham_abduction = load_data_wickham("Abduction", 1, 165.5)
# SaveVariableToFile(dataWickham_abduction, SaveSimulationsDirectory, "dataWickham_abduction_FullRange")

# # ADduction
# dataWickham_adduction = load_data_wickham("Adduction", 15, 120)
# SaveVariableToFile(dataWickham_adduction, SaveSimulationsDirectory, "dataWickham_adduction")

# %% data translation Dal Maso


def load_variable_from_excel(file_name, current_sheet_name, variables_description, SequenceComposantes_y, multiply_factors, min_x, max_x, n_points=100):
    """
    Toujours trier la variable x du plus petit au plus grand

    AJOUTER LES NOMS DE VARIABLES EN SORTIE,
    CONDENSER DANS UN DICTIONNAIRE
    """
    from scipy import interpolate

    Results = {}

    # nombre de variables pour chaque mesures, par défaut
    n_variables = 2

    # Number of lines on top that are not informations to loadheader line in excel
    n_header_line = 2

    File = pd.ExcelFile(file_name)

    # Définit les noms de variables comme la ligne de header entrée
    # Enlève les lignes avant cette ligne de variables
    data = pd.read_excel(File, current_sheet_name, header=n_header_line - 1, )

    File.close()

    # Liste des variables
    Variables = data.columns.to_numpy()[:n_variables]

    # numéro de la colomne à interpoler
    n_col_interpolate = 1
    # Nom de la variable qui sert d'interpolation (par défaut la première variable)
    var_interpolate = Variables[n_col_interpolate - 1]

    # Creates an empty array with the number of interpolated variables
    # array_interpolated_variable_y = np.empty((n_interpolate_points, len(Variables )))
    array_interpolated_variable_y = np.zeros((n_points, len(SequenceComposantes_y)))
    # array_interpolated_variable_y = np.array([[]])

    # Will interpolate the datas with these points
    interpolated_x = np.linspace(min_x, max_x, n_points)
    variable_y_index = 0

    for column_name in data.columns.to_list():

        # Dans le cas où on a la variable d'interpolation
        if var_interpolate in column_name:
            variable_x = data.loc[:, column_name].dropna().to_numpy()

        # quand on n'a pas la variable d'interpolation, interpole cette variable avec la variable en x
        else:
            # Gets the variable, enlève les NAN
            variable_y = data.loc[:, column_name].dropna().to_numpy()

            # calculates the spline interpolate function
            # interpolation_function = InterpolatedUnivariateSpline(variable_x, variable_y, s=5)

            # interpolated_variable_y = interpolation_function(interpolated_x)

            # calculates the spline interpolate function
            interpolation_function = interpolate.CubicSpline(variable_x, variable_y)

            interpolated_variable_y = interpolation_function(interpolated_x)

            # Adds the interpolated y variables to an array
            array_interpolated_variable_y[:, variable_y_index] = interpolated_variable_y
            variable_y_index += 1

    # Stocke le résultat de la variable x
    Results[var_interpolate] = ArrayToDictionary(interpolated_x, variables_description[0], MultiplyFactor=multiply_factors[0])

    # Stocke la variable en y interpollée
    Results[Variables[1]] = ArrayToDictionary(array_interpolated_variable_y, variables_description[1], MultiplyFactor=multiply_factors[1], SequenceComposantes=SequenceComposantes_y)

    return Results


multiply_factors = [1, 1]
variables_description = ["Angle d'abduction [°]", "Translation de la tête humérale [mm]"]
SequenceComposantes_y = ["ML", "AP", "IS"]
file_name = "Translations Dal Maso.xlsx"

# data_Dal_Maso_sup = load_variable_from_excel(file_name, "Dal Maso Supérieur", variables_description, SequenceComposantes_y, multiply_factors, 0, 90, n_points=100)
# data_Dal_Maso_inf = load_variable_from_excel(file_name, "Dal Maso Inférieur", variables_description, SequenceComposantes_y, multiply_factors, 0, 90, n_points=100)


# SaveVariableToFile(data_Dal_Maso_sup, SaveSimulationsDirectory, "data_Dal_Maso_sup")
# SaveVariableToFile(data_Dal_Maso_inf, SaveSimulationsDirectory, "data_Dal_Maso_inf")
