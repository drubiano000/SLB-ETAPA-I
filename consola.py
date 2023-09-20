# -*- coding: utf-8 -*-
"""
Created on Wed Sep 20 00:30:24 2023

@author: drubiano
"""

import back_generación as bg  # Importa el módulo back_generación.py
    
# Función para cargar el archivo CSV con datos de SD y Fallas
def cargar_datos_sd_fallas():
    global df_sd_fallas
    df_sd_fallas = bg.cargar_csv(bg.ruta_archivo_sd_fallas, bg.causas_filtradas)
    print("Archivo CSV de SD y Fallas cargado con éxito.")

# Función para cargar el archivo CSV con datos de canales
def cargar_datos_canales():
    global df_canales
    global pozo_actual
    pozo_actual = input("Escriba el nombre del pozo")
    df_canales = bg.cargar_filtrar_por_pozo(bg.ruta_archivo_canales, pozo_actual)
    print("Archivo CSV de Canales cargado con éxito.")

# Función para generar la lista de diccionarios con la caracterización de SD y Fallas
def generar_lista_diccionarios_sd_fallas():
    global lista_diccionarios_sd_fallas
    lista_diccionarios_sd_fallas = bg.obtener_diccionarios_eventos(df_sd_fallas)
    print("Lista de diccionarios de SD y Fallas generada con éxito.")

# Función para calcular las agregaciones desde el inicio de un evento hacia atrás
def calcular_agregaciones_desde_evento():
    print("para el pozo cargado: " + pozo_actual)
    causa_filtro = input("Ingrese el tipo de causa (Shutdowns, Mechanical Failure, Electrical Failure): ")
    duracion_ventana = float(input("Ingrese la duración de la ventana de tiempo en días: "))
    canal_filtro = input("Ingrese el canal a filtrar: ")
    num_agregaciones = int(input("Ingrese el número de agregaciones a calcular: "))
    df_agregaciones = bg.calcular_agregaciones_desde_evento(lista_diccionarios_sd_fallas, df_canales, causa_filtro, duracion_ventana, canal_filtro, num_agregaciones, pozo_actual)
    nombre_archivo = input("Ingrese el nombre del archivo CSV para guardar las agregaciones: ")
    ruta_archivo = bg.path + pozo_actual + "_" + nombre_archivo
    bg.guardar_df_to_csv(df_agregaciones, ruta_archivo)
    print("Agregaciones calculadas y guardadas con éxito.")

# Función para calcular las agregaciones para toda la línea de tiempo por canal
def calcular_agregaciones_toda_linea_tiempo():
    duracion_ventana = float(input("Ingrese la duración de la ventana de tiempo en días: "))
    canal_filtro = input("Ingrese el canal a filtrar: ")
    df_agregaciones = bg.calcular_agregaciones_por_canal(lista_diccionarios_sd_fallas, df_canales, duracion_ventana, canal_filtro, pozo_actual)
    nombre_archivo = input("Ingrese el nombre del archivo CSV para guardar las agregaciones: ")
    ruta_archivo = bg.path + pozo_actual + "_" + nombre_archivo
    bg.guardar_df_to_csv(df_agregaciones, ruta_archivo)
    print("Agregaciones calculadas y guardadas con éxito.")

#Función para calcular las agregaciones para toda la línea de tiempo todos los canales
def calcular_agregaciones_toda_linea_tiempo_todos_canales():
    duracion_ventana = float(input("Ingrese la duración de la ventana de tiempo en días: "))
    df_agregaciones = bg.calcular_agregaciones(lista_diccionarios_sd_fallas, df_canales, duracion_ventana, pozo_actual)
    nombre_archivo = "estadísticas.csv"
    ruta_archivo = bg.path + pozo_actual + "_" + nombre_archivo
    bg.guardar_df_to_csv(df_agregaciones, ruta_archivo)
    print("Agregaciones calculadas y guardadas con éxito.")

#Función para calcular las agregaciones para toda la línea de tiempo todos los pozos, todos los canales
def calcular_agregaciones_toda_linea_tiempo_todos_pozos():
    print("Puede demorar tiempo, se ejecutará sobre todos los pozos...")
    duracion_ventana = float(input("Ingrese la duración de la ventana de tiempo en días: "))
    df_canales_p = bg.cargar_filtrar_todos_pozos(bg.ruta_archivo_canales)
    bg.calcular_agregaciones(lista_diccionarios_sd_fallas, df_canales_p, duracion_ventana)
                
# Menú principal
while True:
    print("\nMENU DE USUARIO")
    print("1. (*) Cargar archivo CSV con datos de SD y Fallas.")
    print("2. (*) Cargar el archivo CSV con datos de canales de un pozo.")
    print("3. Generar la lista de diccionarios con la caracterización de SD y Fallas.")
    print("4. Calcular las agregaciones desde el inicio de un evento hacia atrás, de un pozo, dada un canal.")
    print("5. Calcular las agregaciones para toda la línea de tiempo, de un pozo, dado un canal.")
    print("6. (*) Calcular las agregaciones para toda la línea de tiempo, de un pozo, para todos los canales.")
    print("7. Calcular las agregaciones para toda la línea de tiempo, para todos los pozos, para todos los canales.")
    print("8. (*) Salir")
    
    opcion = input("Ingrese el número de la opción deseada: ")
    
    if opcion == '1':
        cargar_datos_sd_fallas()
    elif opcion == '2':
        if 'df_sd_fallas' not in globals():
            print("Por favor, primero cargue el archivo CSV de sd y fallas (opción 1).")
        else:
            cargar_datos_canales()
    elif opcion == '3':
        if 'df_sd_fallas' not in globals():
            print("Por favor, primero genere la lista de diccionarios de SD y Fallas (opción 3).")
        else:
            generar_lista_diccionarios_sd_fallas()
    elif opcion == '4':
        if 'lista_diccionarios_sd_fallas' not in globals() and 'df_canales' not in globals() and 'df_sd_fallas' not in globals():
            print("Por favor, primero cargue los datos de SD y Fallas (opción 1) y los datos de Canales (opción 2) y genere la lista de diccionarios (opción 3)")
        else:
            calcular_agregaciones_desde_evento()
    elif opcion == '5':
        if 'lista_diccionarios_sd_fallas' not in globals() and 'df_canales' not in globals() and 'df_sd_fallas' not in globals():
            print("Por favor, primero cargue los datos de SD y Fallas (opción 1) y los datos de Canales (opción 2) y genere la lista de diccionarios (opción 3)")
        else:
            calcular_agregaciones_toda_linea_tiempo()
    elif opcion == '6':
        if 'lista_diccionarios_sd_fallas' not in globals() and 'df_canales' not in globals() and 'df_sd_fallas' not in globals():
            print("Por favor, primero cargue los datos de SD y Fallas (opción 1) y los datos de Canales (opción 2) y genere la lista de diccionarios (opción 3)")
        else:
            calcular_agregaciones_toda_linea_tiempo_todos_canales()  
    elif opcion == '7':
        if 'lista_diccionarios_sd_fallas' not in globals() and 'df_sd_fallas' not in globals():
            print("Por favor, primero cargue los datos de SD y Fallas (opción 1) y genere la lista de diccionarios (opción 3)")
        else:
            calcular_agregaciones_toda_linea_tiempo_todos_pozos() 
    elif opcion == '8':
        print("Saliendo del programa. ¡Hasta luego!")
        break
    else:
        print("Opción no válida. Por favor, seleccione una opción válida.")
