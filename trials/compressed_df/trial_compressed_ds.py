import zipfile
import pandas as pd
import random
import pathlib


# Thanks to https://realpython.com/python-zipfile/#compressing-files-and-directories

# generate random dataframe
def generate_random_df():
    # Paramètres
    nombre_dataframes = 5  # Nombre de DataFrames à générer
    nombre_lignes = 10     # Nombre de lignes par DataFrame
    colonnes = ['A', 'B', 'C']  # Noms des colonnes

    # Liste pour stocker les DataFrames
    dataframes = []

    # Génération des DataFrames
    for _ in range(nombre_dataframes):
        # Crée un DataFrame avec des valeurs aléatoires
        data = {
            'A': [random.randint(0, 100) for _ in range(nombre_lignes)],
            'B': [random.uniform(0, 1) for _ in range(nombre_lignes)],
            'C': [random.choice(['apple', 'banana', 'cherry']) for _ in range(nombre_lignes)]
        }
        df = pd.DataFrame(data)
        dataframes.append(df)

    # Sauvegarder les DataFrames
    for i, df in enumerate(dataframes):
        df.to_feather(f'data/raw/trial_{i}.feather')

def compress_raw_data():
    directory = pathlib.Path("data/raw")
    # Compresser les fichiers feather
    with zipfile.ZipFile("data/compressed/dataframes.zip", mode="w") as archive:
        for file_path in directory.iterdir():
            archive.write(file_path, arcname=file_path.name)

def read_compressed_data():
    with zipfile.ZipFile("data/compressed/dataframes.zip", mode="r") as archive:
        print(archive.namelist())
        with archive.open("trial_2.feather") as file:
            df_from_zip = pd.read_feather(file)
    df = pd.read_feather('data/raw/trial_2.feather')
    print('df identiques : ', df.equals(df_from_zip))


def main():
    generate_random_df()
    compress_raw_data()
    read_compressed_data()


if __name__ == '__main__':
    main()