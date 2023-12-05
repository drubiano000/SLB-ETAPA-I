# -*- coding: utf-8 -*-
"""
Created on Sun Nov 26 15:23:14 2023

@author: jf.perez33, drubiano
"""
#Para más memoria RAM
print("... Inicio")
print(">>>>>>>>>>>>>>>> RAM <<<<<<<<<<<")
from psutil import virtual_memory
ram_gb = virtual_memory().total / 1e9
print('Your runtime has {:.1f} gigabytes of available RAM\n'.format(ram_gb))

if ram_gb < 20:
  print('Not using a high-RAM runtime')
else:
  print('You are using a high-RAM runtime!')
print("--------------------------------------------------------------------------------")
#Para uso de GPU
print(">>>>>>>>>>>>>>>> GPU <<<<<<<<<<<")
import tensorflow as tf
gpu_info = !nvidia-smi
gpu_info = '\n'.join(gpu_info)
if gpu_info.find('failed') >= 0:
  print('Not connected to a GPU')
else:
  print(gpu_info)
print("GPU disponible: ", tf.config.list_physical_devices('GPU'))
print("--------------------------------------------------------------------------------")
#Para el acceso al directorio en drive
import os


from tensorflow import keras
import numpy as np
import pandas as pd
import random
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from google.colab import drive
drive.mount('/content/drive')
# Cambiar al directorio deseado
os.chdir('/content/drive/MyDrive/SLB/')


def concatenar_dataframes(loc_files: str)-> pd.DataFrame:
    """
    Carga y concatena DataFrames desde archivos CSV en la ubicación dada.

    Parámetros:
    - loc_files (str): Ruta a la ubicación de archivos.

    Retorna:
    - pd.DataFrame: DataFrame resultante de la concatenación.
    """
    df = pd.DataFrame()
    for f in os.listdir(loc_files):
        f = "{}/{}".format(loc_files,f)
        #print(f)
        dftemp = pd.read_csv(f)
        #print(dftemp["tipo_causa"].unique())
        if df.shape[0] == 0:
            df = dftemp
            print(f)
            print(dftemp["tipo_causa"].unique())
        else:
            #Solo para los archivos que tienen al menos 4 tipos de causa:
            if len(dftemp["tipo_causa"].unique()) < 5:
                print(f)
                print(dftemp["tipo_causa"].unique())
                df = pd.concat([df,dftemp])
        print(df.shape)
    return df


def mostrar_valores_columna(df: pd.DataFrame, nombre_columna: str)->list:
    """
    Muestra los valores únicos de una columna específica en un DataFrame.

    Parameters:
        - df (pd.DataFrame): DataFrame que contiene los datos.
        - nombre_columna (str): Nombre de la columna para la cual se desean los valores únicos.

    Returns:
        - list: Lista de valores únicos en la columna especificada.
    """
    lista_valores = []
    lista_valores = df[nombre_columna].unique().tolist()
    print(lista_valores)

def visualizacion_nulos_por_col(df: pd.DataFrame())-> None:
    """
    Visualiza los porcentajes de nulos por columna
    Parameters:
        - df (pd.DataFrame): DataFrame que contiene los datos.

    Returns
        lista (list): lista con los nombres de las columnas que son de interes porque son las
                      que tienen un porcentaje de nulos, por debajo del valor de porc_selec.
                      Adicional son las columnas que tienen en su nombre el segmento de texto
                      contenido en seg.
    """
    lista = []
    for c in df.columns:
        porcentaje = df[c].isnull().sum()/df.shape[0]*100
        print("% null in {}: {:2.1f} %".format(c, df[c].isnull().sum()/df.shape[0]*100))

def visualizacion_nulos_por_col_lista(df: pd.DataFrame(), porc_selec: int, seg: str)-> list:
    """
    Visualiza los porcentajes de nulos por columna
    Parameters:
        - df (pd.DataFrame): DataFrame que contiene los datos.

    Returns
        lista (list): lista con los nombres de las columnas que son de interes porque son las
                      que tienen un porcentaje de nulos, por debajo del valor de porc_selec.
                      Adicional son las columnas que tienen en su nombre el segmento de texto
                      contenido en seg.
    """
    lista = []
    for c in df.columns:
        porcentaje = df[c].isnull().sum()/df.shape[0]*100
        print("% null in {}: {:2.1f} %".format(c, df[c].isnull().sum()/df.shape[0]*100))
        if (porcentaje <= porc_selec) and seg in c:
          lista.append(c)
    return lista

def graficar_por_pozo(df: pd.DataFrame, cual_pozo: str, cual_columna: str):
    """
    Grafica una columna específica para un pozo dado en un DataFrame.

    Parameters:
    - df (pd.DataFrame): DataFrame que contiene los datos.
    - cual_pozo (str): Nombre del pozo para el cual se desea realizar la gráfica.
    - cual_columna (str): Nombre de la columna que se desea graficar.

    Returns:
    - None
    """
    #-----------------------------------------------
    # Obtener los valores únicos de la columna "Well"
    valores_unicos_well = df["Pozo"].unique()

    # Mostrar los valores únicos
    #print(valores_unicos_well)
    #-----------------------------------------------

    df_pozo = df[df["Pozo"] == cual_pozo]

    # Mostrar solamente las columnas "Pozo" y "Av_Amp_pro"
    resultados_av_amp = df_pozo[["Pozo", "Av_Amp_pro"]]

    #-----------------------------------------------
    # Mostrar el resultado
    #print(resultados_av_amp)
    #-----------------------------------------------

    df_pozo[cual_columna].plot()
    plt.title(f'Gráfico para el pozo {cual_pozo}')
    plt.xlabel('Índice')
    plt.ylabel(cual_columna)
    plt.show()


def graficar_todos_pozos(df: pd.DataFrame, lst_cols: list):
    nombres_pozos = df['Pozo'].unique().tolist()
    for pozo in nombres_pozos:
      for col in lst_cols:
          graficar_por_pozo(df, pozo, col)


def conteos(df: pd.DataFrame, cual_columna: str, columnas: list):
    """
    Realiza un conteo de ocurrencias para una columna específica en un DataFrame.

    Parameters:
    - df (pd.DataFrame): DataFrame que contiene los datos.
    - cual_columna (str): Nombre de la columna para la cual se desea realizar el conteo.
    - columnas (list): Las columnas sobre las que se hace el conteo

    Returns:
    - None
    """
    df_nuevo = df.copy()
    df_nuevo = df_nuevo[columnas + [cual_columna]]
    print("En la columna" + cual_columna + ": ")
    print(df_nuevo.groupby([cual_columna]).count())


def elimininar_columnas(df: pd.DataFrame(), columnas: list) -> pd.DataFrame():
    """
    Eliminar las columnas de un DataFrame utilizando la biblioteca pandas.

    Parameters:
    - df (pd.DataFrame): DataFrame que contiene los datos.

    Returns:
    - pd.DataFrame: DataFrame con las columnas numéricas reescaladas.
    """
    # Lista de columnas que no se deben reescalar
    cols_not_scaled = columnas

    # Elimina las columnas que no se deben reescalar del DataFrame
    df_scaled = df.drop(cols_not_scaled, axis=1)

    # Retorna el DataFrame con las columnas numéricas reescaladas
    return df_scaled

def imprimir_columnas(df: pd.DataFrame()):
    """
    Imprime los nombres de las columnas de un DataFrame.

    Parameters:
    - df (pd.DataFrame): DataFrame del cual se imprimirán los nombres de las columnas.
    """
    # Imprime los nombres de las columnas del DataFrame
    print(">>> Columnas a reescalar: ")
    print(df.columns)


def reescalar(df: pd.DataFrame()) -> pd.DataFrame():
    """
    Reescala un DataFrame utilizando StandardScaler de scikit-learn.

    Parameters:
    - df (pd.DataFrame): DataFrame que se va a reescalar.

    Returns:
    - pd.DataFrame: DataFrame reescalado.
    """
    # Nota: Esta función utiliza StandardScaler de scikit-learn para reescalar un
    # DataFrame. La función primero crea un objeto StandardScaler, luego ajusta y
    # transforma el DataFrame original, y finalmente devuelve un nuevo DataFrame
    # con los datos reescalados. Nota: el parámetro copy=False en StandardScaler
    # modifica el DataFrame original en lugar de crear una copia.

    # Crea un objeto StandardScaler
    scaler = StandardScaler(copy=False)

    # Aplica la transformación de escala al DataFrame
    scaler.fit_transform(df)

    # Retorna el DataFrame con los datos reescalados
    return pd.DataFrame(df)


def reconstruir_df(df_orig: pd.DataFrame, df_reescalado: pd.DataFrame, columnas: list) -> pd.DataFrame:
    """
    Reconstruye un DataFrame reescalado con las columnas originales.

    Parameters:
    - df_orig (pd.DataFrame): DataFrame original.
    - df_reescalado (pd.DataFrame): DataFrame reescalado.
    - columnas (list): Lista de columnas a reconstruir.

    Returns:
    - pd.DataFrame: DataFrame reescalado reconstruido con las columnas originales.
    """
    #Asigna las columnas seleccionadas del DataFrame original (df_orig) a las mismas
    #columnas en el DataFrame reescalado (df_reescalado).
    #Esto se hace para reconstruir el DataFrame reescalado con las columnas originales.
    df_reescalado[columnas] = df_orig[columnas]
    return df_reescalado


def procesar_indices(df_p: pd.DataFrame, ventana: int):
    """
    Esta función procesa los índices de un DataFrame, ubicando los índice de las observaciones con FAI y con HIB.
    A partir de estos, por cada uno de estos, hacia atrás arma ventanas con índices, quitando los que correspon a FAI y a HIB

    Parámetros:
    - df_p: DataFrame de Pandas, el DataFrame a procesar.
    - ventana: int, el tamaño de la ventana.

    Retorna:
    - list: Una lista de índices filtrados.
    """

    #Corresponde a la ventana que hacia atrás, será observaciones sin falla:
    win_size = ventana
    df = df_p

    dft = df.loc[(df["tipo_causa"] == "FAI") | (df["tipo_causa"] == "HIB")]

    print(dft.shape)
    print(dft.index)

    # Filtrar el DataFrame
    fai_index = dft.index.to_list()
    print(fai_index)
    print(type(fai_index))
    nfai = len(fai_index)
    print("Número de fallas: {}".format(nfai))

    # Quita las fallas repetidas
    fai_index_temp = fai_index
    for i in fai_index:
        win = np.arange(i + 1, i + win_size + 1).tolist()
        for j in win:
            if j in fai_index:
                fai_index_temp.remove(j)
    print(fai_index_temp)
    print(len(fai_index_temp))
    fai_index = fai_index_temp
    nfai = len(fai_index)
    print(" >>> Número de fallas: {}".format(nfai))

    # Retorna la lista de índices filtrados
    return (fai_index, nfai)


def procesar_indices_horizonte(df_p: pd.DataFrame, ventana: int, horizonte_planeacion: int, horizonte_observacion: int):
    """
    Encuentra los índices que cumplen dos condiciones:
    1) En el horizonte_observacion, hay al menos una fila con "FAI" o "HIB" en la columna "tipo_causa".
    2) En el horizonte_planeacion, no hay "FAI" ni "HIB" en la columna "tipo_causa".

    Parameters:
    - df_p (pd.DataFrame): DataFrame de entrada con la columna "tipo_causa".
    - ventana (int): Número de días en la ventana.
    - horizonte_planeacion (int): Número de días en el horizonte de planeación.
    - horizonte_observacion (int): Número de días en el horizonte de observación.

    Returns:
    - indices_resultado (list): Lista de índices que cumplen ambas condiciones.
    - longitud de indices_resultado (int): longitud de la lista de índices que cumplen ambas condiciones
    """
    indices_resultado = []

    for i in range(len(df_p) - horizonte_observacion + 1):
        # Obtener las sub-ventanas de días según los parámetros dados
        ventana_actual = df_p.iloc[i:i + ventana]
        planeacion_actual = df_p.iloc[i + ventana:i + ventana + horizonte_planeacion]
        observacion_actual = df_p.iloc[i + ventana + horizonte_planeacion:i + ventana + horizonte_planeacion + horizonte_observacion]

        # Condición 1: Existe al menos una fila en horizonte_observacion con "FAI" o "HIB"
        if any(observacion_actual["tipo_causa"].isin(["FAI", "HIB"])):
            # Condición 2: En horizonte_planeacion no hay "FAI" ni "HIB"
            # Nota: no mira si hay o no fallas en la ventana
            if not any(planeacion_actual["tipo_causa"].isin(["FAI", "HIB"])):
                indices_resultado.append(i + ventana + horizonte_planeacion)

    print(" >>> Número de fallas: {}".format(len(indices_resultado)))
    return indices_resultado, len(indices_resultado)


def procesar_indices_aleatorios(df: pd.DataFrame, fai_index: list, nfai: int, win_size: int):
    """
    Procesa índices aleatorios para casos sin fallas, generando un conjunto que no incluye
    ventanas con falla y selecciona aleatoriamente el doble del número de observaciones con falla.

    Parámetros:
    - df (pd.DataFrame): DataFrame original.
    - fai_index (list): Lista de índices de observaciones con falla.
    - nfai (int): Número de observaciones con falla.
    - win_size (int): Tamaño de la ventana.

    Retorna:
    - Tuple[list, int]: Tupla con la lista de índices sin falla seleccionados aleatoriamente y su cantidad.
    """

    # Copia la lista de índices con falla
    fai_index_plus = fai_index.copy()

    # Genera un conjunto de índices extendidos según el tamaño de la ventana
    for i in range(1, win_size + 1):
        temp_list = [x + i for x in fai_index]
        print(temp_list)
        #Agrega al final de fai_index_plus, el contenido de temp_list
        fai_index_plus.extend(temp_list)

    # Elimina duplicados y convierte a lista
    fai_index_plus = np.unique(np.array(fai_index_plus)).tolist()

    # Genera una lista de todos los índices sin falla
    nofai_index = np.arange(0, df.shape[0]).tolist()

    # Filtra los índices que podrían incluir una ventana con falla
    nofai_index = [ele for ele in nofai_index if ele not in fai_index_plus]

    # Selecciona aleatoriamente el doble del número de observaciones sin falla
    nofai_index = random.sample(nofai_index, 2 * nfai)

    # Obtiene la cantidad de índices sin falla seleccionados
    nnfai = len(nofai_index)

    print(" >>> Número de casos sin fallas considerados: {}".format(nnfai))

    return (nofai_index, nnfai)


def armar_dfs(df: pd.DataFrame(), lista_variables: list, long_lista: int, fai_index: list, nfai: int, nofai_index: list, nnfai: int, win_size: int):
    """
    Arma DataFrames con series temporales de variables para casos con y sin fallas.

    Parámetros:
    - df (pd.DataFrame): DataFrame original.
    - lista_variables (list): Lista de variables a considerar.
    - long_lista (int): Longitud de la lista de variables.
    - fai_index (list): Lista de índices de observaciones con falla.
    - nofai_index (list): Lista de índices de observaciones sin falla.
    - win_size (int): Tamaño de la ventana.

    Retorna:
    - pd.DataFrames: Tupla con dos elementos, el primero es un array 3D con las series temporales de variables,
      y el segundo es un array 1D con las etiquetas de falla (1 para caso con falla, 0 para caso sin falla).
    """
    print(">>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>")
    nvars = long_lista
    datax = np.zeros(shape=(nfai + nnfai, win_size, nvars))
    print(">>>>>>>> fai_index:")
    print(fai_index)
    #print(datax)
    print("--------------------------")
    print("... En armado de dfs")

    lista_eliminados_nfai = []
    lista_eliminados_nnfai = []
    # Series temporales para casos con fallas
    for idx, i in enumerate(fai_index):
        if i - win_size >= 0:  # Para asegurarse de que hay datos suficientes para la ventana
        #Si por el ejemplo el índice es 2, no puede tomar los 20 anteriores de la ventana.
          print(">>>>> fai: ")
          print("i: " + str(i))
          print("datax[idx, ]: ")
          print(datax[idx, ] )
          print("df.iloc[i - win_size: i][lista_variables]: ")
          print(df.iloc[i - win_size: i][lista_variables])
          datax[[idx, ]] = df.iloc[i - win_size: i][lista_variables]
        else:
          # Eliminar la fila correspondiente a idx
          lista_eliminados_nfai.append(idx)

    # Series temporales para casos sin fallas
    for idx, i in enumerate(nofai_index):
        if i - win_size >= 0:  # Para asegurarse de que hay datos suficientes para la ventana
          #Observaciones para cada variable por la ventana de tiempo:
          print(">>>>> nofai: ")
          print("datax[[nfai + idx, ]]:")
          print(datax[[nfai + idx, ]])
          print("df.iloc[i - win_size:i][lista_variables]: ")
          print(df.iloc[i - win_size:i][lista_variables])
          datax[[nfai + idx, ]] = df.iloc[i - win_size:i][lista_variables]
        else:
          # Eliminar la fila correspondiente a nfai + idx
          lista_eliminados_nnfai.append(nfai + idx)

    #Se eliminan los que no tienen ventana de datax
    for i in lista_eliminados_nfai:
      datax = np.delete(datax, i, axis=0)
    n_nfai = nfai - len(lista_eliminados_nfai)

    for i in lista_eliminados_nnfai:
      datax = np.delete(datax, i, axis=0)
    n_nnfai = nnfai -len(lista_eliminados_nnfai)

    print(datax.shape)

    datay = np.ones(shape=(n_nfai))
    datay = np.append(datay, np.zeros(shape=(n_nnfai)))

    print(datay)
    print(datay.shape)

    return datax, datay, n_nfai, n_nnfai

def armar_dfs_horizonte(df: pd.DataFrame(), lista_variables: list, long_lista: int, fai_index: list, nfai: int, nofai_index: list, nnfai: int, win_size: int, horizonte_planeacion: int):
    """
    Arma DataFrames con series temporales de variables para casos con y sin fallas.

    Parámetros:
    - df (pd.DataFrame): DataFrame original.
    - lista_variables (list): Lista de variables a considerar.
    - long_lista (int): Longitud de la lista de variables.
    - fai_index (list): Lista de índices de observaciones con falla.
    - nofai_index (list): Lista de índices de observaciones sin falla.
    - win_size (int): Tamaño de la ventana.
    - horizonte_planeacion (int): Número de días que corresponden a la plenación del negocio

    Retorna:
    - pd.DataFrames: Tupla con dos elementos, el primero es un array 3D con las series temporales de variables,
      y el segundo es un array 1D con las etiquetas de falla (1 para caso con falla, 0 para caso sin falla).
    """
    print(">>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>")
    nvars = long_lista
    datax = np.zeros(shape=(nfai + nnfai, win_size, nvars))
    print(">>>>>>>> fai_index:")
    print(fai_index)
    #print(datax)
    print("--------------------------")
    print("... En armado de dfs")

    lista_eliminados_nfai = []
    lista_eliminados_nnfai = []
    # Series temporales para casos con fallas
    for idx, i in enumerate(fai_index):
        if i - win_size - horizonte_planeacion >= 0:  # Para asegurarse de que hay datos suficientes para la ventana
        #Si por el ejemplo el índice es 2, no puede tomar los 20 anteriores de la ventana.
          print(">>>>> fai: ")
          print("i: " + str(i))
          print("datax[idx, ]: ")
          print(datax[idx, ] )
          print("df.iloc[i - win_size - horizonte_planeacion: i - horizonte_planeacion][lista_variables]: ")
          print(df.iloc[i - win_size - horizonte_planeacion: i - horizonte_planeacion][lista_variables])
          #Toma desde i - win_size - horizonte_planeacion, pero solo lo referente a win_size
          datax[[idx, ]] = df.iloc[i - win_size - horizonte_planeacion: i - horizonte_planeacion][lista_variables]
        else:
          # Eliminar la fila correspondiente a idx
          lista_eliminados_nfai.append(idx)

    # Series temporales para casos sin fallas
    for idx, i in enumerate(nofai_index):
        if i - win_size - horizonte_planeacion >= 0:  # Para asegurarse de que hay datos suficientes para la ventana
          #Observaciones para cada variable por la ventana de tiempo:
          print(">>>>> nofai: ")
          print("datax[[nfai + idx, ]]:")
          print(datax[[nfai + idx, ]])
          print("df.iloc[i - win_size - horizonte_planeacion:i - horizonte_planeacion][lista_variables]: ")
          print(df.iloc[i - win_size - horizonte_planeacion:i - horizonte_planeacion][lista_variables])
          datax[[nfai + idx, ]] = df.iloc[i - win_size - horizonte_planeacion:i - horizonte_planeacion][lista_variables]
        else:
          # Eliminar la fila correspondiente a nfai + idx
          lista_eliminados_nnfai.append(nfai + idx)

    #Se eliminan los que no tienen ventana de datax
    for i in lista_eliminados_nfai:
      datax = np.delete(datax, i, axis=0)
    n_nfai = nfai - len(lista_eliminados_nfai)

    for i in lista_eliminados_nnfai:
      datax = np.delete(datax, i, axis=0)
    n_nnfai = nnfai -len(lista_eliminados_nnfai)

    print(datax.shape)

    #Pone en uno todos los que implican falla:
    datay = np.ones(shape=(n_nfai))
    #Pone en cero todos los que no implican falla:
    datay = np.append(datay, np.zeros(shape=(n_nnfai)))

    print(datay)
    print(datay.shape)

    return datax, datay, n_nfai, n_nnfai


def graficar_dfs(datax: pd.DataFrame(), datay: pd.DataFrame()):
    """
    Grafica las series temporales de las variables para diferentes clases.

    Parámetros:
    - datax (pd.DataFrame): DataFrame con las series temporales de variables.
    - datay (pd.DataFrame): DataFrame con las etiquetas de clase.

    No retorna nada, solo genera y muestra la gráfica.
    """

    classes = np.unique(datay, axis=0)
    plt.figure()

    for c in classes:
        c_x_train = datax[datay == c]
        print("c_x_train: " + str(c_x_train.shape))
        print("longitud: " + str(c_x_train.shape[0]))
        plt.plot(c_x_train.shape[0], label="class " + str(c))

    plt.legend(loc="best")
    plt.show()
    plt.close()

def imprimir_características_df(df: pd.DataFrame()):
    #Al imprimir, aparece: número de observaciones x tamaño de la ventana x número de variables consideradas:
    print("Número de observaciones x tamaño de la ventana x número de variables consideradas: ")
    print(df.shape)


# La función define una arquitectura de red neuronal convolucional 1D utilizando la API funcional de Keras.
def estructurar_red(input_shape, num_classes: int) -> keras.models.Model:
    """
    Genera una arquitectura de red neuronal convolucional 1D.

    Parámetros:
    - input_shape (Tuple[int]): Forma de los datos de entrada.

    Retorna:
    - keras.models.Model: Modelo de red neuronal convolucional 1D.
    """

    # Capa de entrada
    input_layer = keras.layers.Input(input_shape)

    # Capas convolucionales
    # Crear la primera capa convolucional (conv1)
    conv1 = keras.layers.Conv1D(filters=64, kernel_size=3, padding="same")(input_layer)
    # Aplicar normalización por lotes a conv1
    conv1 = keras.layers.BatchNormalization()(conv1)
    # Aplicar la función de activación ReLU a conv1
    conv1 = keras.layers.ReLU()(conv1)

    # Crear la segunda capa convolucional (conv2)
    conv2 = keras.layers.Conv1D(filters=64, kernel_size=3, padding="same")(conv1)
    # Aplicar normalización por lotes a conv2
    conv2 = keras.layers.BatchNormalization()(conv2)
    # Aplicar la función de activación ReLU a conv2
    conv2 = keras.layers.ReLU()(conv2)

    # Crear la tercera capa convolucional (conv3)
    conv3 = keras.layers.Conv1D(filters=64, kernel_size=3, padding="same")(conv2)
    # Aplicar normalización por lotes a conv3
    conv3 = keras.layers.BatchNormalization()(conv3)
    # Aplicar la función de activación ReLU a conv3
    conv3 = keras.layers.ReLU()(conv3)


    # Capa de reducción global
    gap = keras.layers.GlobalAveragePooling1D()(conv3)

    # Capa de salida
    output_layer = keras.layers.Dense(num_classes, activation="softmax")(gap)

    # Modelo
    return keras.models.Model(inputs=input_layer, outputs=output_layer)

def entrenar_modelo(modelo, num_epochs_p: int, batch_size_p: int, datax, datay):
    """
    Entrena un modelo de red neuronal con los datos proporcionados.
    Esta función toma un modelo de red neuronal, un número de épocas,
    un tamaño de lote, datos de entrada (datax), y etiquetas correspondientes
    (datay) como entrada. Luego, compila y entrena el modelo, guardando el
    mejor modelo durante el entrenamiento y deteniéndolo temprano si la
    pérdida en la validación no mejora. Finalmente, devuelve el historial
    del entrenamiento del modelo.

    Parámetros:
    - modelo: El modelo de red neuronal a entrenar.
    - num_epochs_p (int): Número de épocas de entrenamiento.
    - batch_size_p (int): Tamaño del lote para el entrenamiento.
    - datax: Datos de entrada para el entrenamiento.
    - datay: Etiquetas correspondientes a los datos de entrada.

    Retorna:
    - history: El historial del entrenamiento del modelo.
    """
    epochs = num_epochs_p
    batch_size = batch_size_p

    # Callbacks para guardar el mejor modelo y detener el entrenamiento temprano
    callbacks = [
        keras.callbacks.ModelCheckpoint("best_model.h5", save_best_only=True, monitor="val_loss"),
        keras.callbacks.EarlyStopping(monitor="val_loss", patience=50, verbose=1),
    ]

    # Compilar el modelo
    modelo.compile(
        optimizer="adam",
        loss="sparse_categorical_crossentropy",
        metrics=["sparse_categorical_accuracy"],
    )

    # Entrenar el modelo
    history = modelo.fit(
        datax,
        datay,
        batch_size=batch_size,
        epochs=epochs,
        callbacks=callbacks,
        validation_split=0.2,
        verbose=1,
    )

    return history

def graficar_historial(historial):
    """
    Grafica el historial de entrenamiento de una métrica específica.

    Parámetros:
    - historial: Historial de entrenamiento del modelo.

    Retorna:
    - None
    """
    metric = "sparse_categorical_accuracy"

    # Crear una nueva figura
    plt.figure()

    # Graficar la métrica de entrenamiento y validación
    plt.plot(historial.history[metric])
    plt.plot(historial.history["val_" + metric])

    # Configurar el título y etiquetas del gráfico
    plt.title("Modelo " + metric)
    plt.ylabel(metric, fontsize="large")
    plt.xlabel("Época", fontsize="large")

    # Agregar leyenda
    plt.legend(["Entrenamiento", "Validación"], loc="best")

    # Mostrar el gráfico
    plt.show()

    # Cerrar la figura
    plt.close()

def quitar_nulos(df: pd.DataFrame, resetea_indices: bool)-> pd.DataFrame:
  """
    Quita todas las filas que al menos tienen un nulo, en el DataFrame.

    Parámetros:
    - df (DataFrame): dataframe al que se le quiere eliminar las filas que tienen al menos un nul
      en alguna de sus columnas.
    - resetea_indices (bool): indica si se quiere o no resetear los índices del dataframe.

    Retorna:
    - df (DataFrame): dataframe sin las filas que tienen al menos una columna nan
    """

  # Eliminar filas con al menos un NaN
  print("Eliminación de nulos: ")
  print("Número de filas antes de eliminar nulos: " + str(len(df)))
  df = df.dropna()

  # Si el parámetro es True, resetear los índices después de eliminar las filas
  if (resetea_indices == True):
    df.reset_index(drop=True, inplace=True)
  print("Número de filas después de eliminar nulos: " + str(len(df)))

  return df

def rearmar_solo_col_interes(df: pd.DataFrame, lista_cols: list)-> pd.DataFrame:
  """
    Arma un nuevo DataFrame solo con las columnas de la lista lista_cols.

    Parámetros:
    - df (DataFrame): dataframe que contiene todas las columnas.
    - lista_cols (list): lista con las columnas requeridas.

    Retorna:
    - df_simplificado (DataFrame): dataframe con las columnas requeridas
    """
  df_simplificado = df[lista_cols].copy()
  return df_simplificado

loc_archivos = "2023 11 30 V-1d/"
df_concat = concatenar_dataframes(loc_archivos)
mostrar_valores_columna(df_concat, "tipo_causa")

visualizacion_nulos_por_col(df_concat)
print(">>>>>>>>>>>>>> Generación de las columnas de interés")
porcentaje = int(input("Escriba el valor máximo de porcenaje de nulos para definir columnas de interes: "))
segmento = input("Escriba el tipo de columna que le interesa: pro para promedio, min para mínimo y max para el máximo: ")
lista_mejores_cols = visualizacion_nulos_por_col_lista(df_concat, porcentaje, segmento)
print("Las columnas de interes son: ")
print(lista_mejores_cols)
seguir = input("Mejores columnas identificadas. Enter para continuar...")

graficar = input("Quiere graficar para todos los pozos? 1 -> Si, 0 -> No")
if (graficar == 1):
  graficar_todos_pozos(df_concat, lista_mejores_cols)

conteos(df_concat, "tipo_causa", lista_mejores_cols)
#------- Normalización:
lista_cols = ["Time_Inic", "Pozo", "tipo_causa"]
df_a_reescalar = elimininar_columnas(df_concat, lista_cols)
imprimir_columnas(df_a_reescalar)
df_reescalado = reescalar(df_a_reescalar)
df_reescalado = reconstruir_df(df_concat, df_reescalado, lista_cols)

print("Revisión del df reescalado: ")
print(df_reescalado)

#El manejo de nulos, no puede ser luego del manejo de índices, porque es sobre df_reescalado que se arman
#los índices, y al quitar de df_reescalado los nulos, este queda desincronizado contra los índices de
#fai_index y fai_index y nofai_index
df_reescalado = rearmar_solo_col_interes(df_reescalado, lista_cols + lista_mejores_cols)
df_reescalado = quitar_nulos(df_reescalado, True)
seguir = input("Nulos eliminados. Enter para continuar...")

"""
#Esta es la manera de generar datos para entrenamiento para un horizonte de predicción de una sola unidad de tiempo
win_size = int(input("Escriba el valor de la ventana: "))
(fai_index, nfai) = procesar_indices(df_reescalado, win_size)
#En la siguiente línea se usa contra df_concat_cols_int
(nofai_index, nnfai)= procesar_indices_aleatorios(df_reescalado, fai_index, nfai, win_size)
vars = lista_mejores_cols #Con las variables seleccionadas
nvars = len(vars)
"""

#Esta es la manera de generar datos para entrenamiento para un horizonte de predicción que puede ser de varias unidades de tiempo
win_size = int(input("Escriba el valor de la ventana: "))
horizonte_planeacion = int(input("Escriba el valor del horizonte de planeación: "))
horizonte_observacion = int(input("Escriba el valor del horizonte de observación: "))
(fai_index, nfai) = procesar_indices_horizonte(df_reescalado, win_size, horizonte_planeacion, horizonte_observacion)
#En la siguiente línea se usa contra df_concat_cols_int
(nofai_index, nnfai)= procesar_indices_aleatorios(df_reescalado, fai_index, nfai, win_size)
vars = lista_mejores_cols #Con las variables seleccionadas
nvars = len(vars)

"""
#Esta es la manera de generar datos para entrenamiento para un horizonte de predicción de una sola unidad de tiempo
(datax, datay, nfai, nnfai) = armar_dfs(df_reescalado, vars, nvars, fai_index, nfai, nofai_index, nnfai, win_size)
"""

#Esta es la manera de generar datos para entrenamiento para un horizonte de predicción que puede ser de varias unidades de tiempo
(datax, datay, nfai, nnfai) = armar_dfs_horizonte(df_reescalado, vars, nvars, fai_index, nfai, nofai_index, nnfai, win_size, horizonte_planeacion)
graficar_dfs(datax, datay)
imprimir_características_df(datax)
num_classes = len(np.unique(datay))
print("Número de clases:")
print(num_classes)
# Genera una permutación aleatoria de los índices
idx = np.random.permutation(len(datax))
# Reorganiza las filas en datax utilizando los índices permutados
datax = datax[idx]
# Reorganiza las etiquetas en datay utilizando los índices permutados
datay = datay[idx]
#------------ Entrenamiento:
#Armar la red:
modelo = estructurar_red(input_shape = datax.shape[1:], num_classes = num_classes)
# keras.utils.plot_model: Una función de utilidad de Keras para visualizar modelos.
# modelo: El modelo de red neuronal que se quiere visualizar.
# show_shapes = True: Un parámetro opcional que indica si se deben mostrar las formas de las capas en el gráfico.
# Generación y presentación de un gráfico que representa el modelo de red neuronal.
keras.utils.plot_model(modelo, show_shapes = True)
print("-----------------------------------------")
print("datax sin nan: ")
print(datax)
print("-----------------------------------------")
print("datay sin nan: ")
print(datay)
print(">>>>>>>>>>> Se han eliminado los nan")
siga = 0
#pausa = input("Enter para continuar...")
while (siga == 0):
  epocas = int(input("Escriba el número de epocas: "))
  tamanio = int(input("Tamaño del lote para entrenamiento: "))
  historial = entrenar_modelo(modelo, epocas, tamanio, datax, datay)
  graficar_historial(historial)
  siga = int(input("Quiere salir? 1-> Si, 0-> No "))
