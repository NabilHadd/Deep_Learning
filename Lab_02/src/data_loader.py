import pandas as pd

def sav_to_csv(sav_file_path, csv_file_path):
    """
    Convierte un archivo SAV (SPSS) a formato CSV.
    
    Args:
        sav_file_path (str): Ruta del archivo SAV de entrada
        csv_file_path (str): Ruta del archivo CSV de salida
    
    Returns:
        pd.DataFrame: DataFrame con los datos cargados
    """
    try:
        # Leer archivo SAV
        df = pd.read_spss(sav_file_path)
        
        # Guardar como CSV
        df.to_csv(csv_file_path, index=False, encoding='utf-8', sep=';')
        
        return df
    
    except FileNotFoundError:
        print(f"Error: No se encontró el archivo {sav_file_path}")
    except Exception as e:
        print(f"Error al convertir el archivo: {e}")





def load_csv(csv_file_path):
    """
    Carga un archivo CSV en un DataFrame de pandas.
    
    Args:
        csv_file_path (str): Ruta del archivo CSV a cargar
    
    Returns:
        pd.DataFrame: DataFrame con los datos cargados
    """
    try:
        df = pd.read_csv(csv_file_path, encoding='utf-8', sep=';')
        return df
    
    except FileNotFoundError:
        print(f"Error: No se encontró el archivo {csv_file_path}")
    except Exception as e:
        print(f"Error al cargar el archivo: {e}")




def make_csv(df, csv_file_path):
    """
    Guarda un DataFrame de pandas en formato CSV.
    
    Args:
        df (pd.DataFrame): DataFrame a guardar
        csv_file_path (str): Ruta del archivo CSV de salida
    
    Returns:
        None
    """

    try:
        df.to_csv(csv_file_path, index=False, encoding='utf-8', sep=';')

    except Exception as e:
        print(f"Error al guardar el archivo: {e}")