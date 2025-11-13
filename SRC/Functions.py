# dependencies
import pandas as pd
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import seaborn as sns

#***************************************
def display_dataframe_summary_cf(dataframe):
    """
    Display a summary of the key information

    Return: dataframe
    
    """

    print("--- Shape of the dataframe ---")
    print(f"{dataframe.shape[0]} lignes pour {dataframe.shape[1]} colonnes")
    
    
    print("\n--- Column information ---")
    dataframe.info()

    print("\n--- Check missing values ---")
    print(dataframe.isnull().sum())

    return(dataframe.head())


#***************************************
def check_missing_values(df):
    """
    Détecte les valeurs manquantes (incluant NaN, None, chaînes vides et espaces)
    dans un DataFrame et retourne un tableau avec le nombre et le pourcentage
    de valeurs manquantes par colonne.

    Paramètres:
    df (pd.DataFrame): Le DataFrame à analyser.

    Retourne:
    pd.DataFrame: Un DataFrame récapitulatif des valeurs manquantes.
    """
    # Remplacement des chaînes de caractères vides ou avec seulement des espaces par NaN
    df_cleaned = df.replace(r'^\s*$', np.nan, regex=True)
    
    # Calcul du nombre total de valeurs manquantes par colonne
    missing_count = df_cleaned.isnull().sum()
    
    # Calcul du pourcentage de valeurs manquantes par colonne
    total_rows = len(df)
    missing_percentage = (missing_count / total_rows) * 100
    
    # Création du DataFrame récapitulatif
    missing_data = pd.DataFrame({
        'valeurs_manquantes_count': missing_count,
        'valeurs_manquantes_%': missing_percentage
    })
    missing_data['valeurs_manquantes_%'] = missing_data['valeurs_manquantes_%'].round(2).astype(str) + '%'
    
    # Filtrer les colonnes qui ont des valeurs manquantes
    missing_data = missing_data[missing_data['valeurs_manquantes_count'] > 0]
    
    # Trier le DataFrame par le nombre de valeurs manquantes en ordre décroissant
    missing_data = missing_data.sort_values(by='valeurs_manquantes_count', ascending=False)
    
    return missing_data

#*************************************************************************
def count_outliers_zscore(df, column, threshold=3):
    """
    Compte les valeurs aberrantes dans une colonne d'un DataFrame
    en utilisant la méthode du Z-score.

    Args:
        df (pd.DataFrame): Le DataFrame contenant les données.
        column (str): Le nom de la colonne à analyser.
        threshold (int or float): Le seuil de Z-score. Par défaut, 3.

    Returns:
        int: Le nombre de valeurs aberrantes détectées.
    """
    if column not in df.columns:
        raise ValueError(f"La colonne '{column}' n'existe pas dans le DataFrame.")
    
    mean = df[column].mean()
    std = df[column].std()
    
    z_scores = (df[column] - mean) / std
    
    # Sélectionne les valeurs aberrantes (True/False)
    is_outlier = np.abs(z_scores) > threshold

    print(f"{is_outlier.sum()} outliers for '{column}' variable")


#************************************************************************
import pandas as pd
import numpy as np

def handle_outliers_zscore(df, columns, action='detect', threshold=3):
    """
    Détecte ou supprime les valeurs aberrantes dans une ou plusieurs
    colonnes d'un DataFrame en utilisant la méthode du Z-score.

    Args:
        df (pd.DataFrame): Le DataFrame d'entrée.
        columns (list): Une liste des noms des colonnes à analyser.
        action (str): L'action à effectuer.
                      - 'detect': retourne un dictionnaire avec le nombre d'aberrantes par colonne.
                      - 'remove': retourne le DataFrame sans les lignes contenant des valeurs aberrantes.
        threshold (int or float): Le seuil de Z-score pour la détection.

    Returns:
        dict ou pd.DataFrame: Dépend de la valeur de 'action'.
    """
    if action not in ['detect', 'remove']:
        raise ValueError("L'action doit être 'detect' ou 'remove'.")

    df_zscore = df.copy()
    outlier_counts = {}

    for col in columns:
        if col not in df_zscore.columns:
            print(f"Attention : La colonne '{col}' n'existe pas dans le DataFrame.")
            continue
        
        # Calcul des Z-scores
        mean = df_zscore[col].mean()
        std = df_zscore[col].std()
        z_scores = (df_zscore[col] - mean) / std
        
        # Identification des valeurs aberrantes
        is_outlier = np.abs(z_scores) > threshold
        
        if action == 'detect':
            outlier_counts[col] = is_outlier.sum()
            print(f"{is_outlier.sum()} outliers for {col} variable")
        
        elif action == 'remove':
            df_zscore = df_zscore[~is_outlier]
            print(f"The outliers for the variable {col} have been successfully handled")

    if action == 'remove':
        return df_zscore
    
    
#***********************************************************************************
import pandas as pd
import numpy as np

def handle_outliers_iqr(df, columns, action='detect', threshold=1.5):
    """
    Détecte ou supprime les valeurs aberrantes dans une ou plusieurs
    colonnes d'un DataFrame en utilisant la méthode de l'IQR.

    Args:
        df (pd.DataFrame): Le DataFrame d'entrée.
        columns (list): Une liste des noms des colonnes à analyser.
        action (str): L'action à effectuer.
                      - 'detect': retourne un dictionnaire avec le nombre d'aberrantes par colonne.
                      - 'remove': retourne le DataFrame sans les lignes contenant des valeurs aberrantes.
        threshold (int or float): Le seuil de l'IQR (par défaut 1.5).

    Returns:
        dict ou pd.DataFrame: Dépend de la valeur de 'action'.
    """
    if action not in ['detect', 'remove']:
        raise ValueError("L'action doit être 'detect' ou 'remove'.")

    df_iqr = df.copy()
    outlier_counts = {}

    for col in columns:
        if col not in df_iqr.columns:
            print(f"Attention : La colonne '{col}' n'existe pas dans le DataFrame.")
            continue
        
        # Calcul de l'IQR et des seuils
        Q1 = df_iqr[col].quantile(0.25)
        Q3 = df_iqr[col].quantile(0.75)
        IQR = Q3 - Q1
        
        lower_bound = Q1 - (threshold * IQR)
        upper_bound = Q3 + (threshold * IQR)
        
        # Identification des valeurs aberrantes
        is_outlier = (df_iqr[col] < lower_bound) | (df_iqr[col] > upper_bound)
        
        if action == 'detect':
            outlier_counts[col] = is_outlier.sum()
            print(f"{is_outlier.sum()} outliers for {col} variable")
        
        elif action == 'remove':
            df_iqr = df_iqr[~is_outlier]
            print(f"The outliers for the variable {col} have been successfully handled")

    if action == 'remove':
        return df_iqr


#************************************************************************************
import pandas as pd
import numpy as np

def handle_outliers_modified_zscore(df, columns, action='detect', threshold=3.5):
    """
    Detects or removes outliers in one or more DataFrame columns using the
    Modified Z-score method.

    Args:
        df (pd.DataFrame): The input DataFrame.
        columns (list): A list of column names to analyze.
        action (str): The action to perform.
                      - 'detect': returns a dictionary with the number of outliers per column.
                      - 'remove': returns the DataFrame without the rows containing outliers.
        threshold (float): The Modified Z-score threshold (default is 3.5).

    Returns:
        dict or pd.DataFrame: Depends on the value of 'action'.
    """
    if action not in ['detect', 'remove']:
        raise ValueError("The 'action' must be 'detect' or 'remove'.")

    df_copy = df.copy()
    outlier_counts = {}

    for col in columns:
        if col not in df_copy.columns:
            print(f"Warning: Column '{col}' not found in the DataFrame. Skipping.")
            continue
        
        # Calculate the Median and MAD (Median Absolute Deviation)
        median = df_copy[col].median()
        mad = np.median(np.abs(df_copy[col] - median))
        
        # Check to avoid division by zero
        if mad == 0:
            is_outlier = pd.Series([False] * len(df_copy), index=df_copy.index)
        else:
            # Calculate the Modified Z-scores
            modified_z_scores = 0.6745 * (df_copy[col] - median) / mad
            # Identify outliers
            is_outlier = np.abs(modified_z_scores) > threshold

        if action == 'detect':
            outlier_counts[col] = is_outlier.sum()
            print(f"{is_outlier.sum()} outliers for {col} variable")
        
        elif action == 'remove':
            df_copy = df_copy[~is_outlier]
            print(f"The outliers for the variable {col} have been successfully handled")

    if action == 'detect':
        return outlier_counts
    else:  # action == 'remove'
        return df_copy







    