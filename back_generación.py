# -*- coding: utf-8 -*-
"""
Created on Mon Sep 11 09:17:32 2023

@author: drubiano
"""
import pandas as pd
from datetime import datetime, timedelta

#Archivo relacionado con los datos de SD y Fallas:
path = "E:/Nuevo 2/SLB/Fuentes/"
archivo_sd_fallas = "DOWNTIMES_ANDES.csv"
ruta_archivo_sd_fallas = path + archivo_sd_fallas

#Lista con los nombres de las causas de fallas de las bombas:
causas_filtradas = ['Shutdowns', 'Mechanical Failure', 'Electrical Failure']

#Archivo relacionado con los datos de los canales:
archivo_canales = "ESP_RAW_DATA_ANDES.csv"
ruta_archivo_canales = path + archivo_canales

#Alias de cada canal:
dic_alias_canal = {}
dic_alias_canal["Average Amps"] = "Av_Amp_"
dic_alias_canal["Discharge Pressure"] = "Ds_Pre_"	
dic_alias_canal["Drive Frequency"] = "Dr_Fre_"	
dic_alias_canal["Intake Pressure"] = "Int_Pre_"
dic_alias_canal["Intake Temperature"] = "Int_Tmp_"
dic_alias_canal["Motor Temperature"] = "Mo_Tmp_"
dic_alias_canal["Vibration"] = "Vib_"
dic_alias_canal["Passive Current Leakage"] = "Pas_Cur_"		
dic_alias_canal["Active Current Leakage"	] = "Act_Cur_"
dic_alias_canal["Motor Load"] = "Mot_Ld_"
dic_alias_canal["Zero Current"] = "Zer_Cur_"
dic_alias_canal["Input Voltage"] = "Inp_Volt_"
dic_alias_canal["Output voltage"] = "Out_vol_"	
dic_alias_canal["VSD Output current"] = "VSD_"

#Lista de canales:
lista_canales = ["Average Amps", 
                 "Discharge Pressure", 
                 "Drive Frequency",
                 "Intake Pressure",
                 "Intake Temperature",
                 "Motor Temperature",
                 "Vibration",
                 "Passive Current Leakage",
                 "Active Current Leakage",
                 "Motor Load",
                 "Zero Current",
                 "Input Voltage",
                 "Output voltage",
                 "VSD Output current"]

#--------------------------------------------------------------------------------------------------------------------------------------
#                                          MANEJO DE LOS DATOS RELACIONADOS CON LAS CAUSAS DE FALLAS

# Función para cargar el archivo CSV y retornar un DataFrame
def cargar_csv(ruta_archivo: str, causas_filtro: []) -> pd.DataFrame:
    """
    Carga el archivo .csv en un DataFrame.

    Parámetros:
       La ruta del archivo .csv ()incluido el nombre del archivo). 
       La lista con los nombres de las causas de falla

    Returns:
       DataFrame: dataframe con los datos extraídos del archivo .csv.
    """
    try:
        df = pd.read_csv(ruta_archivo, sep=';')
        
        # Actualizar los valores en la columna "cause_3"
        df['cause_3'] = df['cause_3'].replace({'Electrical Failures': 'Electrical Failure'})
        df['cause_3'] = df['cause_3'].replace({'Electrical Failure Operations': 'Electrical Failure'})
        df_filtrado = df[df['cause_3'].isin(causas_filtro)]
        # Asegurarse de que la columna "date" esté en formato de fecha
        df_filtrado['date'] = pd.to_datetime(df_filtrado['date'], format='%d-%b-%y')
        # Ordenar el DataFrame por las columnas "well", "cause_3" y "date" en orden ascendente
        df_filtrado = df_filtrado.sort_values(by=['well', 'cause_3', 'date'], ascending=[True, True, True])

        return df_filtrado
    except FileNotFoundError:
        print("El archivo no se encuentra en la ubicación especificada.")
        return None

# Función para procesar el DataFrame y obtener la lista de diccionarios de Fallas y SD, a cada una le pone un nombre
# y la fecha de inicio y la fecha de finalización:
def obtener_diccionarios_eventos(df: pd.DataFrame) -> list:
    """
    Genera una lista de diccionarios, donde cada diccionario contiene la información relacionada con cada causa identificada:
    - 'identif_causa'
    - 'pozo'
    - 'fecha_inicial'
    - 'fecha_final'
    - 'causa'   

    Parámetros:
    df: DataFrame con los datos de las cuasas de las fallas de las bombas

    Returns:
       lis: lista de diccionarios.
    """
    grupos = []
    diccionarios = []
    
    # Contadores:
    contadores = {
                    "Shutdowns": 0,
                    "Mechanical Failure": 0, 
                    "Electrical Failure": 0
                 }
    
    for _, row in df.iterrows():
        row_date = row['date'] 
        if grupos and (row['well'] != grupos[-1]['well'] or row['cause_3'] != grupos[-1]['cause_3'] or
                      (row_date - grupos[-1]['date']).days > 1):
            # Llegó al final del grupo:    
            diccionarios.append({
                'identif_causa': row['cause_3'] + "_" + str(contadores[grupos[-1]['cause_3']]),
                'pozo': grupos[0]['well'],
                'fecha_inicial': grupos[0]['date'].replace(hour=0, minute=0, second=0).strftime('%Y-%m-%d %H:%M:%S+00:00'),
                'fecha_final': grupos[-1]['date'].replace(hour=23, minute=59, second=59).strftime('%Y-%m-%d %H:%M:%S+00:00'),
                'causa': grupos[0]['cause_3']
            })
            contadores[row['cause_3']] = contadores[grupos[-1]['cause_3']] + 1         
   
            #Se limpia grupos, para empezar a armar un nuevo grupo
            grupos = []

        #La línea leída del df se va agregando a grupos (durante la creación del grupo):
        grupos.append({
            'identif_causa': "",
            'well': row['well'],
            'date': row_date,
            'cause_3': row['cause_3']
        })
            
    if grupos:
        diccionarios.append({
            'identif_causa': row['cause_3'] + "_" + str(contadores[grupos[-1]['cause_3']]),
            'pozo': grupos[0]['well'],
            'fecha_inicial': grupos[0]['date'].replace(hour=0, minute=0, second=0).strftime('%Y-%m-%d %H:%M:%S+00:00'),
            'fecha_final': grupos[-1]['date'].replace(hour=23, minute=59, second=59).strftime('%Y-%m-%d %H:%M:%S+00:00'),
            'causa': grupos[0]['cause_3']
        })

    return diccionarios

#--------------------------------------------------------------------------------------------------------------------------------------
#                CRUCE ENTRE LOS DATOS RELACIONADOS CON LAS CAUSAS DE FALLAS Y/O SD CON LOS DATOS RELACIONADOS CON LOS CANALES

# Función para cargar y filtrar el archivo CSV por pozo
def cargar_filtrar_por_pozo(nombre_archivo: str, nombre_pozo: str) -> pd.DataFrame:
    """
    Carga el archivo .csv en un DataFrame, con los datos de los canales para un pozo.

    Parámetros:
       nombre_archivo: la ruta y el nombre del archivo 
       nomre_pozo: nombre del pozo para el cual se filtran los datos de las conales.

    Returns:
       DataFrame: dataframe con los datos extraídos del archivo .csv.
    """
    # Cargar el archivo CSV en un DataFrame
    #df = pd.read_csv(nombre_archivo, parse_dates=['Time'],  nrows=1000000)
    #dayfirst=True para indicar que en las fechas, el formato día-mes-año (como '31/12/2019')
    df = pd.read_csv(nombre_archivo, parse_dates=['Time'], dayfirst=True)
    
    # Filtrar el DataFrame por el nombre del pozo
    df_pozo = df[df['well'] == nombre_pozo]
    df_pozo = df_pozo[df_pozo['Time'] != 'Time']    

    try:
        # Time es de tipo Object
        # El formato de Time es el del siguiente ejemplo: 31/12/2019 19:19.
        # Crear una copia de la columna 'Time' antes de la conversión. 
        #La copia se usa en los casos en los que ya no aplique el formato mencionada
        df_pozo['Time_copy'] = df_pozo['Time']
        df_pozo['Time'] = pd.to_datetime(df_pozo['Time'], format='%d/%m/%Y %H:%M')
    except ValueError:
        # El formato de Time es el del siguiente ejemplo: 2020-01-01T00:19:00Z
        # Es un formato que no se pudo tratar en el Try, por eso genera la excepción.
        df_pozo['Time'] = pd.to_datetime(df_pozo['Time_copy'], format='%Y-%m-%dT%H:%M:%SZ')

    # Eliminar la columna de copia, ya que no son necesarias
    df_pozo.drop(columns=['Time_copy'], inplace=True)
    
    # Verificar que todos los valores de la columna 'Time' sean de tipo datetime
    if pd.api.types.is_datetime64_dtype(df_pozo['Time']):
        # Formatear la columna Time en el formato AAAA-MM-DD 00:00:00+00:00
        df_pozo['Time'] = df_pozo['Time'].dt.strftime('%Y-%m-%d %H:%M:%S+00:00')
    
    # Ordenar el DataFrame por la columna Time
    df_pozo = df_pozo.sort_values(by='Time', ascending=True)
  
    return df_pozo


#Función que genera agregaciones, hacia atrás, a paritr de la fecha de inicio las causas de falla de bomba o de SD que son de un tipo dado.
def calcular_agregaciones_desde_evento(lista, df, causa_filtro, periodo, canal_filtro, cuantas_agregaciones, pozo_filtro) -> pd.DataFrame:
    """
    Genera un dataframe con las agregaciones, de tal manera que cada agregación se construye en función de una ventana de tiempo
    dada por el parámetro periodo. La agregación corresponde a las estadísticas: min, max y prom sobre los datos un canal, dado por canal_filtro, 
    tomados en el periodo de tiempo defindo por el parámetro perido. Esta función genera las agregraciones, hacia atrás, a partir del inicio
    de una causa de falla o de UN sd. Se generan tantas agregaciones como lo indique el parámetro cuantas_aregaciones.

    Parámetros:
       lista: lista de diccionarois con los datos de las cusas.
       df: dataframe con los datos de los canales.
       causa_filtro: tipo de causa que define las causas, a partir de las cuales, se generan agregaciones hacia atrás.
       peirodo: tiempo que define la ventana, sobre la cual se establece un agregación en términos de los valores estadísticos,
                generados sobre los datos de un canal en especial, definido por canal_filtro.
       canal_filtro: canal por el cual se requiere filtar los datos.
       cuantas_agregaciones: indica cuantas agregaciones se generan.
       pozo_filtro: pozo sobre el cual se filtran los diferentes datos.
       
    Returns:
       DataFrame: dataframe con las agrgaciones generadas.
    """
    diccionarios_salida = []
    #Genera un df solo con las columnas requeridas: "Time" y la de la canal objeto de agregación:
    df_canal = df[["Time", canal_filtro]]
    df_canal = df_canal[df_canal[canal_filtro] != "NaN"]
    df_canal = df_canal[df_canal[canal_filtro] != "nan"]

    i = 0
    for diccionario_leido in lista:
        if diccionario_leido['causa'] == causa_filtro and diccionario_leido['pozo'] == pozo_filtro:
            # Se lee la fecha inicial:
            fecha_inicial = datetime.strptime(diccionario_leido['fecha_inicial'], '%Y-%m-%d %H:%M:%S+00:00')
            # Convierte fecha_inicial a datetime64[ns, UTC]
            fecha_inicial = pd.to_datetime(fecha_inicial, utc=True)

            # Filtrar registros anteriores a la fecha inicial de diccionario_leido
            df_canal['Time'] = pd.to_datetime(df_canal['Time'])
            df_filtrado = df_canal[df_canal['Time'] < fecha_inicial]

            # Obtenemos el DataFrame invertido
            df_invertido = df_filtrado.iloc[::-1]
                
            # Generación de las agregaciones
            tiempo_final = None
            contador = 0
            for _, row in df_invertido.iterrows():
                if tiempo_final is None:
                    # La primera línea del dataframe
                    tiempo_final = pd.to_datetime(row['Time'])  # Convierte row['Time'] a Timestamp
                elif (tiempo_final - pd.to_datetime(row['Time'])) > timedelta(days=periodo):
                    # Final del período, luego hay que construir la estadísticas con la información del período.
                    df_grupo = df_filtrado[(df_filtrado['Time'] >= pd.to_datetime(row['Time'])) & (df_filtrado['Time'] < tiempo_final)]
                    #print("tiempo_final: " + str(tiempo_final))
                    #print("pd.to_datetime(row['Time']:" + str(pd.to_datetime(row['Time'])))
                    #print("------------------ ANTES ---------------------")
                    #print(df_grupo[['Time', 'Average Amps']])
                    #enter = input("Enter...")
                    estadisticas = {
                        'Time_Inic': tiempo_final,
                        'Pozo': pozo_filtro,
                        'identif_causa': diccionario_leido['identif_causa'],
                        'minimo': df_grupo[canal_filtro].min(),
                        'maximo': df_grupo[canal_filtro].max(),
                        'promedio': df_grupo[canal_filtro].mean()
                    }
                    #- print(estadisticas)
                    diccionarios_salida.append(estadisticas)
                    # Marca el nuevo tiempo final para la siguiente agregación:
                    tiempo_final = pd.to_datetime(row['Time'])
                    if contador == cuantas_agregaciones:
                        # Si se completa el número de agregaciones esperado, se interrumpe el ciclo
                        break
                    contador += 1
        i = i + 1
    # Crear un DataFrame a partir de la lista de diccionarios resultante
    df_resultado = pd.DataFrame(diccionarios_salida)
    print("***********************************************")
    print(diccionarios_salida)
    return df_resultado



#Función que genera agregaciones, para un tipo de canal, filtrando los datos para un pozo:
def calcular_agregaciones_por_canal(lista, df, duracion, canal_filtro, pozo_filtro) -> pd.DataFrame:
    """
    Genera un dataframe con las agregaciones, de tal manera que cada agregación se construye en función de una ventana de tiempo
    dada por el parámetro periodo. La agregación corresponde a las estadísticas: min, max y prom sobre los datos de un canal, dado por canal_filtro, 
    tomados en el periodo de tiempo defindo por el parámetro duracion. Acompañando a cada agregación, se identifica si en la ventana de tiempo
    de la misma hubo alguna causa de falla: SD, FAILURE o Las dos o Ninguna.

    Parámetros:
       lista: lista de diccionarois con los datos de las cusas.
       df: dataframe con los datos de los canales.
       duracion: tiempo que define la ventana, sobre la cual se establece un agregación en términos de los valores estadísticos,
                generados sobre los datos de un canal en especial, definido por canal_filtro.
       canal_filtro: canal por el cual se requiere filtar los datos.
       pozo_filtro: pozo sobre el cual se filtran los diferentes datos.
       
    Returns:
       DataFrame: dataframe con las agrgaciones generadas.
    """
    diccionarios_salida = []

    # Asegurarse de que 'Time' sea de tipo datetime y tenga zona horaria UTC
    df['Time'] = pd.to_datetime(df['Time'], utc=True)
    df.sort_values(by='Time', inplace=True)
    
    # Filtro sobre la lista para diccionarios con el pozo correcto
    lista_leida = [d for d in lista if d['pozo'] == pozo_filtro]

    if lista_leida:
        # Genera un df solo con las columnas requeridas: "Time" y la columna de canal objeto de agregación:
        df_canal = df[["Time", canal_filtro]]
        df_canal = df_canal[~df_canal[canal_filtro].isin(["NaN", "nan"])]

        # Inicializa el tiempo_inicial con el tiempo del primer registro
        tiempo_inicial = df_canal.iloc[0]['Time']
        
        for _, row in df_canal.iterrows():
            # Filtrar registros dentro del grupo de tiempo
            tiempo_actual = row['Time']
            
            if (tiempo_actual - tiempo_inicial) >= timedelta(days=duracion):
                # Se identifica la generación de la ventana de tiempo, por lo que se genera la respectiva agregación
                # a partir de los diccionarios de lista_leida
                eventos = []
                for diccionario_leido in lista_leida:
                    fecha_inicial = pd.to_datetime(diccionario_leido['fecha_inicial'], format='%Y-%m-%d %H:%M:%S+00:00', utc=True)
                    fecha_final = pd.to_datetime(diccionario_leido['fecha_final'], format='%Y-%m-%d %H:%M:%S+00:00', utc=True)

                    if (tiempo_inicial <= fecha_inicial < tiempo_actual) or (tiempo_inicial <= fecha_final < tiempo_actual):
                        causa_evento = diccionario_leido['causa']
                        if causa_evento == "Shutdowns":
                            eventos.append("SD")
                        elif causa_evento in ["Mechanical Failure", "Electrical Failure"]:
                            eventos.append("FAI")
                        
                grupo_causa = ""
                if "SD" in eventos and "FAI" in eventos:
                    grupo_causa = "HIB"
                elif "SD" in eventos:
                    grupo_causa = "SD"
                elif "FAI" in eventos:
                    grupo_causa = "FAI"
                else:
                    grupo_causa = "NIN"

                df_grupo = df_canal[(df_canal['Time'] >= tiempo_inicial) & (df_canal['Time'] < tiempo_actual)]

                estadisticas = {
                    'Time_Inic': tiempo_inicial,
                    'Pozo': pozo_filtro,
                    'tipo_causa': grupo_causa,
                    dic_alias_canal[canal_filtro] + 'min': df_grupo[canal_filtro].min(),
                    dic_alias_canal[canal_filtro] + 'max': df_grupo[canal_filtro].max(),
                    dic_alias_canal[canal_filtro] + 'pro': df_grupo[canal_filtro].mean()
                }
                diccionarios_salida.append(estadisticas)
                
                # Actualiza el tiempo_inicial para el próximo grupo
                tiempo_inicial = tiempo_actual

    # Crear un DataFrame a partir de la lista de diccionarios resultante
    df_resultado = pd.DataFrame(diccionarios_salida)
    print("***********************************************")
    print(diccionarios_salida)
    return df_resultado

#Función que genera agregaciones, para un tipo de canal, filtrando los datos para un pozo:
def calcular_agregaciones(lista, df, duracion, pozo_filtro) -> pd.DataFrame:
    """
    Genera un dataframe con las agregaciones, de tal manera que cada agregación se construye en función de una ventana de tiempo
    dada por el parámetro periodo. La agregación corresponde a las estadísticas: min, max y prom de todos los canales, 
    tomados en el periodo de tiempo defindo por el parámetro duracion. Acompañando a cada agregación, se identifica si en la ventana de tiempo
    de la misma hubo alguna causa de falla: SD, FAILURE o Las dos o Ninguna.

    Parámetros:
       lista: lista de diccionarois con los datos de las cusas.
       df: dataframe con los datos de los canales.
       duracion: tiempo que define la ventana, sobre la cual se establece un agregación en términos de los valores estadísticos,
                generados sobre los datos de un canal en especial, definido por canal_filtro.
       pozo_filtro: pozo sobre el cual se filtran los diferentes datos.
       
    Returns:
       DataFrame: dataframe con las agrgaciones generadas.
    """    
    diccionarios_salida = []

    # Asegurarse de que 'Time' sea de tipo datetime y tenga zona horaria UTC
    df['Time'] = pd.to_datetime(df['Time'], utc=True)
    df.sort_values(by='Time', inplace=True)
    
    # Filtro sobre la lista para diccionarios con el pozo correcto
    lista_leida = [d for d in lista if d['pozo'] == pozo_filtro]

    if lista_leida:
        # Copia en df_canal (es para manterner el nombre a lo largo del algoritmo):
        df_canal = df
        
        # Inicializa el tiempo_inicial con el tiempo del primer registro
        tiempo_inicial = df_canal.iloc[0]['Time']
        
        for _, row in df_canal.iterrows():
            # Filtrar registros dentro del grupo de tiempo
            tiempo_actual = row['Time']
            
            if (tiempo_actual - tiempo_inicial) >= timedelta(days=duracion):
                # Se identifica la generación de la ventana de tiempo, por lo que se genera la respectiva agregación
                # a partir de los diccionarios de lista_leida
                eventos = []
                for diccionario_leido in lista_leida:
                    fecha_inicial = pd.to_datetime(diccionario_leido['fecha_inicial'], format='%Y-%m-%d %H:%M:%S+00:00', utc=True)
                    fecha_final = pd.to_datetime(diccionario_leido['fecha_final'], format='%Y-%m-%d %H:%M:%S+00:00', utc=True)

                    if (tiempo_inicial <= fecha_inicial < tiempo_actual) or (tiempo_inicial <= fecha_final < tiempo_actual):
                        causa_evento = diccionario_leido['causa']
                        if causa_evento == "Shutdowns":
                            eventos.append("SD")
                        elif causa_evento in ["Mechanical Failure", "Electrical Failure"]:
                            eventos.append("FAI")
                        
                grupo_causa = ""
                if "SD" in eventos and "FAI" in eventos:
                    grupo_causa = "HIB"
                elif "SD" in eventos:
                    grupo_causa = "SD"
                elif "FAI" in eventos:
                    grupo_causa = "FAI"
                else:
                    grupo_causa = "NIN"
                    
                    
                estadisticas = {
                    'Time_Inic': tiempo_inicial,
                    'Pozo': pozo_filtro,
                    'tipo_causa': grupo_causa,
                }

                #Por cada canal, cuyo nombre está en la lista lista_canales, se generan las respectivas estadísticas:
                for canal_filtro in lista_canales: 
                    #Se arma el df para el canal dado por canal_filtro:
                    df_canal_g = df_canal[["Time", canal_filtro]]
                    #Se eliminan las filas que no tienen valor:
                    df_canal_g = df_canal_g[~df_canal_g[canal_filtro].isin(["NaN", "nan"])]
                    #Se filtra solo para los registros de ese canal que estan dentro de los tiempos definidos por la ventana de tiempo
                    #(tiempo_inicial y tiempo_actual):
                    df_grupo = df_canal_g[(df_canal_g['Time'] >= tiempo_inicial) & (df_canal_g['Time'] < tiempo_actual)]

                    estadisticas[dic_alias_canal[canal_filtro] + 'min'] = df_grupo[canal_filtro].min()
                    estadisticas[dic_alias_canal[canal_filtro] + 'max'] = df_grupo[canal_filtro].max()
                    estadisticas[dic_alias_canal[canal_filtro] + 'pro']  = df_grupo[canal_filtro].mean()
                
                diccionarios_salida.append(estadisticas)
                
                # Actualiza el tiempo_inicial para el próximo grupo
                tiempo_inicial = tiempo_actual

    # Crear un DataFrame a partir de la lista de diccionarios resultante
    df_resultado = pd.DataFrame(diccionarios_salida)
    print("***********************************************")
    print(diccionarios_salida)
    return df_resultado

#---------------------------------------------------------------------------------------------------
#Función que muesta los diccionarios con los datos de una falla o SD, que estan en la lista que corresponde al nombre de pozo dado.
def mostrar_diccionarios_por_pozo(lista_diccionarios: list, nombre_pozo: str):
    """
    Muestra en la consola los diccionarios en la lista que corresponden al nombre de pozo dado.

    Parámetros:
        lista_diccionarios (list): Lista de diccionarios generada por la función obtener_diccionarios.
        nombre_pozo (str): Nombre del pozo a buscar.

    Returns:
        None
    """
    print(f"Diccionarios para el pozo: {nombre_pozo}")
    for diccionario in lista_diccionarios:
        if diccionario['pozo'] == nombre_pozo:
            print(diccionario)
        
#Genera una lista de nombres de pozos únicos a partir de un DataFrame
def generar_lista_pozos(df: pd.DataFrame())-> list:
    """
    Genera una lista de nombres de pozos únicos a partir de un DataFrame.

    Parametros:
       df (pd.DataFrame): El DataFrame que contiene los datos de los pozos.

    Returns:
       list: Una lista de nombres de pozos únicos presentes en el DataFrame.
    """
    nombres_pozos_unicos = df['pozo'].unique().tolist()
    return nombres_pozos_unicos

#Guarda el contenido de un DataFrame en un archivo . csv    
def guardar_df_to_csv(df: pd.DataFrame, nombre_archivo: str):
    """
    Guarda el contenido de un pdf en un archiv .csv.

    Parametros:
       df (pd.DataFrame): El DataFrame que contiene las estadísticas por cada canal.
    """
    # Utiliza el método `to_csv` para guardar el DataFrame en el archivo CSV
    df.to_csv(nombre_archivo, index=False)

#---------------------------------------------------------------------------------------------------
#                                     PARA TODOS LOS POZOS
#Función que carga el dataframe con la información de canales de todos los pozos
def cargar_filtrar_todos_pozos(nombre_archivo: str) -> pd.DataFrame:
    """
    Carga el archivo .csv en un DataFrame, con los datos de los canales de todos los pozos.

    Parámetros:
       nombre_archivo: la ruta y el nombre del archivo 

    Returns:
       DataFrame: dataframe con los datos extraídos del archivo .csv.
    """
    # Cargar el archivo CSV en un DataFrame
    #df = pd.read_csv(nombre_archivo, parse_dates=['Time'],  nrows=1000000)
    #dayfirst=True para indicar que en las fechas, el formato día-mes-año (como '31/12/2019')
    df = pd.read_csv(nombre_archivo, parse_dates=['Time'], dayfirst=True)
    
    # Filtrar el DataFrame quitanto todos los campos "Time" que son iguales al valor "Time":
    df_pozo = df[df['Time'] != 'Time']    

    try:
        # Time es de tipo Object
        # El formato de Time es el del siguiente ejemplo: 31/12/2019 19:19.
        # Crear una copia de la columna 'Time' antes de la conversión. 
        #La copia se usa en los casos en los que ya no aplique el formato mencionada
        df_pozo['Time_copy'] = df_pozo['Time']
        df_pozo['Time'] = pd.to_datetime(df_pozo['Time'], format='%d/%m/%Y %H:%M')
    except ValueError:
        # El formato de Time es el del siguiente ejemplo: 2020-01-01T00:19:00Z
        # Es un formato que no se pudo tratar en el Try, por eso genera la excepción.
        df_pozo['Time'] = pd.to_datetime(df_pozo['Time_copy'], format='%Y-%m-%dT%H:%M:%SZ')

    # Eliminar la columna de copia, ya que no son necesarias
    df_pozo.drop(columns=['Time_copy'], inplace=True)
    
    # Verificar que todos los valores de la columna 'Time' sean de tipo datetime
    if pd.api.types.is_datetime64_dtype(df_pozo['Time']):
        # Formatear la columna Time en el formato AAAA-MM-DD 00:00:00+00:00
        df_pozo['Time'] = df_pozo['Time'].dt.strftime('%Y-%m-%d %H:%M:%S+00:00')
    
    # Ordenar el DataFrame por la columna Time
    df_pozo = df_pozo.sort_values(by='Time', ascending=True)
  
    return df_pozo

#Función que genera todas las agregaciones sobre todos los pozos:
def calcular_agregaciones_todos_pozos(lista, duracion):
    """
    Genera todas las agragaciones de todos los pozos
    
    Parámetros:
        lista con los diccionarios con los datos de SD y Fallas 

    """    
    df = cargar_filtrar_todos_pozos(ruta_archivo_canales)
    lista_pozos = generar_lista_pozos()
    for nombre_pozo in lista_pozos:
        print("Pozo: " + pozo)
        df_pozo = df[df['well'] == nombre_pozo]
        df_canales = calcular_agregaciones(lista, df_pozo, duracion, nombre_pozo)
        archivo = "estadísticas_t.csv"
        ruta_archivo = path + nombre_pozo + "_" + archivo
        guardar_df_to_csv(df_canales, ruta_archivo)

#---------------------------------------------------------------------------------------------------
#                                             PRUEBAS

# Uso:
# Consolidación de Fallas y SD:
df = cargar_csv(ruta_archivo_sd_fallas, causas_filtradas)
resultados = obtener_diccionarios_eventos(df)
# Mostrar los eventos para un pozo en especial:
pozo = 'Well-52'
print(">>> Las fallas y los SD del pozo " + pozo)
mostrar_diccionarios_por_pozo(resultados, pozo)
#------------------------------------------------------------------
# Los posibles canales para canal_filtro son:
'''
-	Average Amps	
-	Discharge Pressure	
-	Drive Frequency		
-	Intake Pressure 
-	Intake Temperature	
-	Motor Temperature	
-	Vibration	
-	Passive Current Leakage		
-	Active Current Leakage	
-	Motor Load	
-	Zero Current
-	Input Voltage	
-	Output voltage	
-	VSD Output current		
'''

print(">>> Cargando canales...")
df_ag = cargar_filtrar_por_pozo(ruta_archivo_canales, pozo)
#Para agregaciones cada 50 días
print(">>> Generando agregaciones...")

# Generación de las agregaciones para el pozo 52, solo para el evento 'Mechanical Failure':
#df_r = calcular_agregaciones_desde_evento(resultados, df_ag, 'Mechanical Failure', 50, "Average Amps", 10, pozo)

# Generación de las agregaciones para el canal "Average Amps":
#df_r = calcular_agregaciones_por_canal(resultados,df_ag,50,"Average Amps",pozo)

# Generación de las agregaciones:
df_r = calcular_agregaciones(resultados,df_ag,50,pozo)

# Envío del contenido del datafrme a un archivo:
archivo = "estadísticas.csv"
ruta_archivo = path + pozo + "_" + archivo
guardar_df_to_csv(df_r, ruta_archivo)    
    
print(">>> ... Proceso finalizado")

#Mezcla de 2 datafarmes por una misma columna: nuevo_dataframe = dataframe1.merge(dataframe2, on='time')