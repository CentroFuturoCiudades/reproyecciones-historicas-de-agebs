import os
import numpy as np
import pandas as pd
import geopandas as gpd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.colors import Normalize
from matplotlib.colors import LinearSegmentedColormap
from shapely.geometry import Point
import contextily as ctx

def load_mpios_metropoli(mpios_metropoli_path):
    mpios_en_metropoli = gpd.read_file(mpios_metropoli_path)
    mpios_en_metropoli['CVE_ENT'] = mpios_en_metropoli['CVEGEO'].str[:2]
    mpios_en_metropoli['CVE_MUN'] = mpios_en_metropoli['CVEGEO'].str[2:5]
    return mpios_en_metropoli

def load_agebs_1990(agebs_1990_path):
    # Leer las AGEBs
    agebs_1990  = gpd.read_file(agebs_1990_path)

    # Procesamiento de AGEBs
    agebs_1990["CVEGEO"] = agebs_1990["CVE_ENT"] + agebs_1990["CVE_MUN"] + agebs_1990["CVE_LOC"] + agebs_1990["CVE_AGEB"]
    return agebs_1990

def load_censo_1990(censo_1990_path):
    # Cargar el censo 19990
    census_1990 = pd.read_csv(censo_1990_path)

    # Elimina valores faltantes y modifica el formato de MUN, LOC para obtener el CVEGEO
    census_1990 = census_1990.dropna()
    census_1990['MUN'] = census_1990['MUN'].astype(int)
    census_1990['LOC'] = census_1990['LOC'].astype(int)
    census_1990['MUN']     = census_1990['MUN'].apply(lambda x: f'{x:03d}')
    census_1990['LOC']     = census_1990['LOC'].apply(lambda x: f'{x:04d}')
    census_1990['CVEGEO']  = census_1990['ENT'] + census_1990['MUN'] + census_1990['LOC'] + census_1990['AGEB']
    census_1990.rename(columns={'Población total':'POBTOT'}, inplace = True)
    return census_1990

def filter_censo_1990(agebs_1990, censo_1990_cleaned, zona_metropolitana, cols):
    agebs_1990_filtered = pd.merge(zona_metropolitana, agebs_1990, on=['CVE_ENT', 'CVE_MUN'], how='inner')
    agebs_1990_filtered = gpd.GeoDataFrame(agebs_1990_filtered, geometry='geometry')
    agebs_1990_filtered.crs = agebs_1990.crs
    
    censo_1990_filtered = agebs_1990_filtered.merge(censo_1990_cleaned, on='CVEGEO', how='left')
    censo_1990_filtered = censo_1990_filtered[cols]
    return censo_1990_filtered
    
def load_censo_2000(agebs_2000_path):
    # Carga del censo 2000 (agebs y censo)
    agebs_2000  = gpd.read_file(agebs_2000_path)

    # Procesamiento 
    agebs_2000['CVE_MUN'] = agebs_2000['CVE_MUN'].str[2:]
    agebs_2000['CVE_AGEB'] = agebs_2000['CVE_AGEB'].str.replace('-', '')
    agebs_2000.rename(columns = {"CVE_EDO":"CVE_ENT","CVE_AGEB": "CVEGEO", "Z1": "POBTOT"}, inplace = True)
    return agebs_2000

def filter_censo_2000(agebs_2000, zona_metropolitana, cols):
    # Obtener datos del lugar
    #NOM_MUN = PLACE["NOM_MUN"]
    #CVE_ENT = PLACE["CVE_ENT"]
    #CVE_MUN = PLACE["CVE_MUN"]

    # Filtrar únicamente del lugar
    #censo_2000_filtered = agebs_2000.loc[(agebs_2000["CVE_ENT"]== PLACE["CVE_ENT"]) & (agebs_2000["CVE_MUN"]== PLACE["CVE_MUN"])]

    censo_2000_filtered = pd.merge(zona_metropolitana, agebs_2000, on=['CVE_ENT', 'CVE_MUN'], how='inner')
    censo_2000_filtered = gpd.GeoDataFrame(censo_2000_filtered, geometry='geometry')
    censo_2000_filtered.crs = agebs_2000.crs
    return censo_2000_filtered[cols]

def load_agebs_2010(agebs_2010_path):
    # Leer las AGEBs
    agebs_2010  = gpd.read_file(agebs_2010_path)

    # Procesamiento de AGEBs
    agebs_2010['CVE_ENT'] = agebs_2010['CVEGEO'].str[:2]
    agebs_2010['CVE_MUN'] = agebs_2010['CVEGEO'].str[2:5]
    return agebs_2010


def load_censo_2010(censo_2010_path):
    # Encuentra todos los csv en un folder 
    xls_files = [file for file in os.listdir(censo_2010_path) if file.endswith(".csv")]

    # Crea un dataframe para concatenar los archivos encontrados
    censo_2010 = pd.DataFrame()
    
    for xls_file in xls_files:
        file_path = os.path.join(censo_2010_path, xls_file)
        temp_df = pd.read_csv(file_path)
        censo_2010 = pd.concat([censo_2010, temp_df], ignore_index=True)

    # Asegurarse de que las columnas tengan el formato deseado
    censo_2010['ENTIDAD'] = censo_2010['ENTIDAD'].apply(lambda x: f'{x:02d}')
    censo_2010['MUN'] = censo_2010['MUN'].apply(lambda x: f'{x:03d}')
    censo_2010['LOC'] = censo_2010['LOC'].apply(lambda x: f'{x:04d}')
    
    # Concatenar las columnas en una sola columna "CVEGEO"
    censo_2010['CVEGEO'] = censo_2010['ENTIDAD'] + censo_2010['MUN'] + censo_2010['LOC'] + censo_2010['AGEB']

    # Seleccionar filas apropiadas y columnas a usar 
    censo_2010_cleaned = censo_2010[censo_2010["NOM_LOC"]=="Total AGEB urbana"][["CVEGEO", "POBTOT"]]
    return censo_2010_cleaned

def filter_censo_2010(agebs_2010, censo_2010_cleaned, zona_metropolitana, cols):
    # Obtener datos del lugar
    #NOM_MUN = PLACE["NOM_MUN"]
    #CVE_ENT = PLACE["CVE_ENT"]
    #CVE_MUN = PLACE["CVE_MUN"]

    # Filtrar únicamente del lugar
    #agebs_2010_filtered = agebs_2010.loc[(agebs_2010["CVE_ENT"]== PLACE["CVE_ENT"]) & (agebs_2010["CVE_MUN"]== PLACE["CVE_MUN"])]

    agebs_2010_filtered = pd.merge(zona_metropolitana, agebs_2010, on=['CVE_ENT', 'CVE_MUN'], how='inner')
    agebs_2010_filtered = gpd.GeoDataFrame(agebs_2010_filtered, geometry='geometry')
    agebs_2010_filtered.crs = agebs_2010.crs
    
    censo_2010_filtered = agebs_2010_filtered.merge(censo_2010_cleaned, on='CVEGEO', how='left')
    censo_2010_filtered = censo_2010_filtered[cols]
    return censo_2010_filtered

def load_agebs_2020(agebs_2020_path):
    agebs_2020  = gpd.read_file(agebs_2020_path)
    return agebs_2020

def load_censo_2020(censo_2020_path):
    # Encuentra todos los csv en el folder
    csv_files = [file for file in os.listdir(censo_2020_path) if file.endswith(".csv")]
    
    censo_2020 = pd.DataFrame()
    
    for csv_file in csv_files:
        file_path = os.path.join(censo_2020_path, csv_file)
        temp_df = pd.read_csv(file_path)
        censo_2020 = pd.concat([censo_2020, temp_df], ignore_index=True)

    # Asegurarse de que las columnas tengan el formato deseado
    censo_2020['ENTIDAD'] = censo_2020['ENTIDAD'].apply(lambda x: f'{x:02d}')
    censo_2020['MUN'] = censo_2020['MUN'].apply(lambda x: f'{x:03d}')
    censo_2020['LOC'] = censo_2020['LOC'].apply(lambda x: f'{x:04d}')
    censo_2020['AGEB'] = censo_2020['AGEB'].apply(lambda x: str(x).zfill(4))
    
    # Concatenar las columnas en una sola columna "CVEGEO"
    censo_2020['CVEGEO'] = censo_2020['ENTIDAD'] + censo_2020['MUN'] + censo_2020['LOC'] + censo_2020['AGEB']
    censo_2020_cleaned = censo_2020[censo_2020["NOM_LOC"]=="Total AGEB urbana"][["CVEGEO", "POBTOT"]]
    return censo_2020_cleaned

def filter_censo_2020(agebs_2020, censo_2020_cleaned, zona_metropolitana, cols):
    # Obtener datos del lugar
    #NOM_MUN = PLACE["NOM_MUN"]
    #CVE_ENT = PLACE["CVE_ENT"]
    #CVE_MUN = PLACE["CVE_MUN"]

    #agebs_2020_filtered = agebs_2020.loc[(agebs_2020["CVE_ENT"]== PLACE["CVE_ENT"]) & (agebs_2020["CVE_MUN"]== PLACE["CVE_MUN"])  & (agebs_2020.Ambito == "Urbana")]

    agebs_2020_filtered = pd.merge(zona_metropolitana, agebs_2020, on=['CVE_ENT', 'CVE_MUN'], how='inner')
    agebs_2020_filtered = gpd.GeoDataFrame(agebs_2020_filtered, geometry='geometry')
    agebs_2020_filtered.crs = agebs_2020.crs
    
    # Filtrar únicamente del lugar
    censo_2020_filtered = agebs_2020_filtered.merge(censo_2020_cleaned, on='CVEGEO', how='left')
    censo_2020_filtered = censo_2020_filtered.reset_index(drop = True)
    return censo_2020_filtered[cols]

def save_censo(censo, year, PLACE, path_data_processed):
    file_path = path_data_processed + "censo_"+str(year)+"_"+PLACE.lower()+".shp"
    censo.to_file(file_path, driver='ESRI Shapefile')


def reproject_ageb_to_mesh(ageb_file, malla_path, level = 8):
    if level == 8:
        malla_path_full = malla_path + "nivel8.shp"
        malla_geo = load_malla(ageb_file, malla_path_full)
    elif level == 9:
        malla_files = [file for file in os.listdir(malla_path) if file.endswith(".shp") and "nivel9" in file] 
        resultados = []
        for malla_file in malla_files:
            malla_geo_parcial = load_malla(ageb_file, malla_path + malla_file)
            resultados.append(malla_geo_parcial)
            
        malla_geo = pd.concat(resultados, ignore_index=True)
        malla_geo.rename(columns = {"CODIGO":"codigo"}, inplace = True)
    malla_geo_reprojected = malla_geo.to_crs(ageb_file.crs)

    # Realizar la intersección espacial entre las geometrías
    ageb_file["ageb_area"] = ageb_file.area
    intersection = ageb_file.sjoin(malla_geo_reprojected.assign(geometry_malla = malla_geo_reprojected.geometry),
                                   how = 'left')    

    # Calcular la proporción de área compartida
    intersection['shared_area'] = intersection.geometry.intersection(intersection.geometry_malla).area / intersection.ageb_area
    
    # Proyectar la población
    intersection['poblacion_proyectada'] = (intersection['shared_area'] * intersection['POBTOT']).apply(np.floor)

    # Calcular la corrección para todas las filas de cada grupo
    grouped_intersection = intersection.groupby('CVEGEO')
    
    correction = (grouped_intersection['POBTOT'].transform('first') - grouped_intersection['poblacion_proyectada'].transform('sum')) * \
                  intersection['shared_area'].eq(intersection.groupby('CVEGEO')['shared_area'].transform('max'))
    
    intersection['poblacion_proyectada'] += correction
    
    # Agrupar por celda de la malla y sumar la población proyectada
    poblacion_por_malla = intersection.groupby('codigo')['poblacion_proyectada'].sum().reset_index()

    # Realizar la fusión entre poblacion_por_malla y el GeoDataFrame de la intersección
    poblacion_por_malla_con_geometria = poblacion_por_malla.merge(malla_geo_reprojected[['codigo', 'geometry']], on='codigo', how='left')
    poblacion_por_malla_con_geometria = gpd.GeoDataFrame(poblacion_por_malla_con_geometria, geometry='geometry')
    return poblacion_por_malla_con_geometria.to_crs(malla_geo.crs)

def load_malla(ageb_file, malla_path):
    # Reproyectar las AGEBs al CRS de la Malla para obtener el bounding box
    bbox = tuple((ageb_file.to_crs("EPSG:6365")).total_bounds)

    # Cargar únicamente datos en el bounding box
    malla_geo = gpd.read_file(
        malla_path,
        bbox=bbox,
    )
    return malla_geo

def make_plots_pob(place_1990, place_2000, place_2010, place_2020, result_df, PLACE_NAME, path_data_modeling, path_data_raw):
    # Encontrar el valor máximo de población en cualquiera de los años
    max_population = result_df[['poblacion_proyectada_2000', 'poblacion_proyectada_2010', 'poblacion_proyectada_2020']].max().max()
    
    # Configuración de la gráfica
    fig, ax = plt.subplots(1, 1, figsize=(15, 15))
    legend_kwds = {'label': "Población Proyectada"}

    # Color Map
    # Definir el color blanco y naranja
    blanco = [1, 1, 1]
    naranja_inicial = [1, 0.8, 0]
    naranja_final = [0.8, 0.4, 0]
    
    # Crear un colormap personalizado
    nuevo_colormap = LinearSegmentedColormap.from_list('blanco_a_naranja', [blanco, naranja_inicial, naranja_final], N=256)

    # Coordenadas del punto objetivo
    ghsl_metro = pd.read_csv(path_data_raw+'GHS_STAT_UCDB2015MT_GLOBE_R2019A/GHS_STAT_UCDB2015MT_GLOBE_R2019A_V1_3.csv')
    ghsl_metro = ghsl_metro.loc[(ghsl_metro.CTR_MN_NM == "Mexico") & (ghsl_metro.UC_NM_MN ==PLACE_NAME)]
    pnt_metro  = ghsl_metro.GCPNT_LON, ghsl_metro.GCPNT_LAT
    start_point = gpd.GeoDataFrame(geometry=[Point(pnt_metro[0], pnt_metro[1])], crs="EPSG:4326")
    
    # Obtener el bounding box alrededor del punto (5 km a la redonda)
    buffer_distance_km = 25
    buffer_geometry = start_point.buffer(buffer_distance_km / 111.32).geometry.iloc[0]  # Extraer la geometría
    
    # Convertir el bounding box a un GeoDataFrame en EPSG:4326
    bbox = gpd.GeoDataFrame(geometry=[buffer_geometry], crs="EPSG:4326")
    
    # Reproyectar el punto y el bbox al CRS deseado (EPSG:6365)
    start_point = start_point.to_crs("EPSG:6365")
    bbox = bbox.to_crs("EPSG:6365")
    xmin, ymin, xmax, ymax = bbox.total_bounds
    target_x, target_y = start_point.geometry.x.values[0], start_point.geometry.y.values[0]

    
    # Filtrar los datos dentro del rango
    #result_df = result_df.cx[bbox.total_bounds[0]:bbox.total_bounds[2], bbox.total_bounds[1]:bbox.total_bounds[3]]
    
        
    # Gráfico para 2020
    result_df[result_df['poblacion_proyectada_2020'] != 0].plot(column='poblacion_proyectada_2020', 
                                                                edgecolor='lightgray',
                                                                cmap=nuevo_colormap, 
                                                                linewidth=0.25, 
                                                                legend=True, 
                                                                ax=ax, 
                                                                legend_kwds=legend_kwds, 
                                                                norm=Normalize(0, max_population), alpha = 0.6)
    place_2020.to_crs("EPSG:6365").plot(ax=ax, color='none', edgecolor='black', linewidth=0.25, alpha = 0.25)  # Contorno de las AGEBs sin color
    ctx.add_basemap(ax, crs=result_df.crs, source='https://tiles.stadiamaps.com/tiles/stamen_toner_lite/{z}/{x}/{y}{r}.png?api_key=a1b11e3f-26c1-414c-8c58-1f8b74f719d0')
    ax.scatter([target_x], [target_y], marker=(5, 1), c='red', s=200, zorder=5)
    plt.title('Población Proyectada de {} en 2020 por Celda de Malla'.format(PLACE_NAME))
    ax.set_xticks([])  # Eliminar marcas del eje x
    ax.set_yticks([])  # Eliminar marcas del eje y
    ax.set_xlim(xmin, xmax)
    ax.set_ylim(ymin, ymax)
    plt.savefig(path_data_modeling+PLACE_NAME.lower()+'_poblacion_2020.png')
    plt.show()
    
    # Gráfico para 2010
    fig, ax = plt.subplots(1, 1, figsize=(15, 15))
    result_df[result_df['poblacion_proyectada_2010'] != 0].plot(column='poblacion_proyectada_2010', 
                                                                edgecolor='lightgray', 
                                                                cmap=nuevo_colormap, 
                                                                linewidth=0.25, 
                                                                legend=True, ax=ax, 
                                                                legend_kwds=legend_kwds, 
                                                                norm=Normalize(0, max_population), alpha = 0.6)
    place_2010.to_crs("EPSG:6365").plot(ax=ax, color='none', edgecolor='black', linewidth=0.25, alpha = 0.25)  # Contorno de las AGEBs sin color
    ctx.add_basemap(ax, crs=result_df.crs, source='https://tiles.stadiamaps.com/tiles/stamen_toner_lite/{z}/{x}/{y}{r}.png?api_key=a1b11e3f-26c1-414c-8c58-1f8b74f719d0')
    ax.scatter([target_x], [target_y], marker=(5, 1), c='red', s=200, zorder=5)
    plt.title('Población Proyectada de {} en 2010 por Celda de Malla'.format(PLACE_NAME))
    ax.set_xticks([])  # Eliminar marcas del eje x
    ax.set_yticks([])  # Eliminar marcas del eje y
    ax.set_xlim(xmin, xmax)
    ax.set_ylim(ymin, ymax)
    plt.savefig(path_data_modeling+PLACE_NAME.lower()+'_poblacion_2010.png')
    plt.show()
    
    # Gráfico para 2000
    fig, ax = plt.subplots(1, 1, figsize=(15, 15))
    # cmap='YlOrBr'
    result_df[result_df['poblacion_proyectada_2000'] != 0].plot(column='poblacion_proyectada_2000', 
                                                                edgecolor='lightgray', cmap=nuevo_colormap, 
                                                                linewidth=0.25, legend=True, ax=ax, 
                                                                legend_kwds=legend_kwds, norm=Normalize(0, max_population), alpha = 0.6)
    place_2000.to_crs("EPSG:6365").plot(ax=ax, color='none', edgecolor='black', linewidth=0.25, alpha = 0.25)  # Contorno de las AGEBs sin color
    ctx.add_basemap(ax, crs=result_df.crs, source='https://tiles.stadiamaps.com/tiles/stamen_toner_lite/{z}/{x}/{y}{r}.png?api_key=a1b11e3f-26c1-414c-8c58-1f8b74f719d0')
    ax.scatter([target_x], [target_y], marker=(5, 1), c='red', s=200, zorder=5)
    plt.title('Población Proyectada de {} en 2000 por Celda de Malla'.format(PLACE_NAME))
    ax.set_xticks([])  # Eliminar marcas del eje x
    ax.set_yticks([])  # Eliminar marcas del eje y
    ax.set_xlim(xmin, xmax)
    ax.set_ylim(ymin, ymax)
    plt.savefig(path_data_modeling+PLACE_NAME.lower()+'_poblacion_2000.png')
    plt.show()


    # Gráfico para 2000
    fig, ax = plt.subplots(1, 1, figsize=(15, 15))
    # cmap='YlOrBr'
    result_df[result_df['poblacion_proyectada_1990'] != 0].plot(column='poblacion_proyectada_1990', 
                                                                edgecolor='lightgray', cmap=nuevo_colormap, 
                                                                linewidth=0.25, legend=True, ax=ax, 
                                                                legend_kwds=legend_kwds, norm=Normalize(0, max_population), alpha = 0.6)
    place_1990.to_crs("EPSG:6365").plot(ax=ax, color='none', edgecolor='black', linewidth=0.25, alpha = 0.25)  # Contorno de las AGEBs sin color
    ctx.add_basemap(ax, crs=result_df.crs, source='https://tiles.stadiamaps.com/tiles/stamen_toner_lite/{z}/{x}/{y}{r}.png?api_key=a1b11e3f-26c1-414c-8c58-1f8b74f719d0')
    ax.scatter([target_x], [target_y], marker=(5, 1), c='red', s=200, zorder=5)
    plt.title('Población Proyectada de {} en 1990 por Celda de Malla'.format(PLACE_NAME))
    ax.set_xticks([])  # Eliminar marcas del eje x
    ax.set_yticks([])  # Eliminar marcas del eje y
    ax.set_xlim(xmin, xmax)
    ax.set_ylim(ymin, ymax)
    plt.savefig(path_data_modeling+PLACE_NAME.lower()+'_poblacion_2000.png')
    plt.show()

def make_plots_dif(place_2020, result_df, PLACE_NAME, zona_metropolitana,path_data_modeling, path_data_raw):
    
    
    # Encontrar el valor máximo absoluto de la diferencia para normalizar el colormap
    #max_diff = abs(result_df['diferencia_2020_2010']).max()
    max_diff = abs(result_df['diferencia_2020_1990'].min())

    # Coordenadas del punto objetivo
    ghsl_metro = pd.read_csv(path_data_raw+'GHS_STAT_UCDB2015MT_GLOBE_R2019A/GHS_STAT_UCDB2015MT_GLOBE_R2019A_V1_3.csv')
    ghsl_metro = ghsl_metro.loc[(ghsl_metro.CTR_MN_NM == "Mexico") & (ghsl_metro.UC_NM_MN ==PLACE_NAME)]
    pnt_metro  = ghsl_metro.GCPNT_LON, ghsl_metro.GCPNT_LAT
    start_point = gpd.GeoDataFrame(geometry=[Point(pnt_metro[0], pnt_metro[1])], crs="EPSG:4326")
    buffer_geometry = start_point.buffer(5 / 111.32).geometry.iloc[0]
    buffer_geometry_limit = start_point.buffer(25 / 111.32).geometry.iloc[0]

    # Convertir el bounding box a un GeoDataFrame en EPSG:4326
    bbox = gpd.GeoDataFrame(geometry=[buffer_geometry_limit], crs="EPSG:4326")
    
    # Reproyectar el punto y el bbox al CRS deseado (EPSG:6365)
    bbox = bbox.to_crs("EPSG:6365")
    xmin, ymin, xmax, ymax = bbox.total_bounds


    
    buffer_geometry_gdf = gpd.GeoDataFrame(geometry=[buffer_geometry], crs="EPSG:6365")

    """
    # Obtener el bounding box alrededor del punto (5 km a la redonda)
    buffer_distance_km = 20
    buffer_geometry = start_point.buffer(buffer_distance_km / 111.32).geometry.iloc[0]  # Extraer la geometría
    
    # Convertir el bounding box a un GeoDataFrame en EPSG:4326
    bbox = gpd.GeoDataFrame(geometry=[buffer_geometry], crs="EPSG:4326")
    
    # Reproyectar el punto y el bbox al CRS deseado (EPSG:6365)
    start_point = start_point.to_crs("EPSG:6365")
    bbox = bbox.to_crs("EPSG:6365")
    xmin, ymin, xmax, ymax = bbox.total_bounds
    """
    start_point = start_point.to_crs("EPSG:6365")
    intersects_result = result_df.intersects(buffer_geometry)
    print(result_df[intersects_result]['diferencia_2020_2010'].sum())
    
    
    target_x, target_y = start_point.geometry.x.values[0], start_point.geometry.y.values[0]

    
    # Asignar colores a los valores negativos y positivos utilizando un colormap divergente (RdBu)
    # Configuración de la gráfica para Diferencia 2020-2010
    """
    fig, ax = plt.subplots(1, 1, figsize=(15, 15))
    
    result_df.plot(column='diferencia_2020_2010', cmap='RdBu', edgecolor='white', linewidth=0.25,
                   norm=Normalize(-max_diff, max_diff), legend=True, ax=ax)
    zona_metropolitana.to_crs(result_df.crs).plot(ax=ax, color='none', edgecolor='black', linewidth=0.25, alpha = 0.25) 
    #place_2020.to_crs("EPSG:6365").boundary.plot(ax=ax, color='none', edgecolor='black', linewidth=0.7)  # Contorno de las AGEBs sin color
    ax.set_xticks([])  # Eliminar marcas del eje x
    ax.set_yticks([])  # Eliminar marcas del eje y
    ax.set_xlim(xmin, xmax)
    ax.set_ylim(ymin, ymax)   
    plt.title('Diferencia de Población 2020-2010 en {}'.format(PLACE_NAME))
    plt.savefig(path_data_modeling+PLACE_NAME.lower()+'_grafico_diferencia_2020_2010.png')
    plt.show()
    """

    # Configuración de la gráfica para Diferencia 2020-2010
    fig, ax = plt.subplots(1, 1, figsize=(15, 15))
    
    # Encontrar el valor máximo absoluto de la diferencia para normalizar el colormap
    #max_diff = abs(result_df['diferencia_2020_2000']).max()
    #max_diff = abs(result_df['diferencia_2020_2000'].min())

    ax.scatter([target_x], [target_y], marker=(5, 1), c='red', s=200, zorder=5)
    
    # Asignar colores a los valores negativos y positivos utilizando un colormap divergente (RdBu)
    result_df.plot(column='diferencia_2020_2010', cmap='RdBu', edgecolor='white', linewidth=0.25,
                   norm=Normalize(-max_diff, max_diff), legend=True, ax=ax, legend_kwds={'aspect': 20}, alpha = 0.6)

    ctx.add_basemap(ax, crs=result_df.crs, source='https://tiles.stadiamaps.com/tiles/stamen_toner_lite/{z}/{x}/{y}{r}.png?api_key=a1b11e3f-26c1-414c-8c58-1f8b74f719d0')
    print(result_df.crs)
    #buffer_geometry_gdf.boundary.plot(color = 'black', edgecolor = 'black', ax = ax)
    #zona_metropolitana.to_crs(result_df.crs).plot(ax=ax, color='none', edgecolor='black', linewidth=0.25, alpha = 0.25) 
    ax.scatter([target_x], [target_y], marker=(5, 1), c='red', s=200, zorder=5)
    #place_2020.to_crs("EPSG:6365").boundary.plot(ax=ax, color='none', edgecolor='black', linewidth=0.7)  # Contorno de las AGEBs sin color
    ax.set_xticks([])  # Eliminar marcas del eje x
    ax.set_yticks([])  # Eliminar marcas del eje y
    ax.set_xlim(xmin, xmax)
    ax.set_ylim(ymin, ymax)    
    plt.title('Diferencia de Población 2020-2010 en {}'.format(PLACE_NAME))
    plt.savefig(path_data_modeling+PLACE_NAME.lower()+'_grafico_diferencia_2020_2010.png')
    plt.show()

    
    
    # Configuración de la gráfica para Diferencia 2020-2000
    fig, ax = plt.subplots(1, 1, figsize=(15, 15))
    
    # Encontrar el valor máximo absoluto de la diferencia para normalizar el colormap
    #max_diff = abs(result_df['diferencia_2020_2000']).max()
    #max_diff = abs(result_df['diferencia_2020_2000'].min())

    ax.scatter([target_x], [target_y], marker=(5, 1), c='red', s=200, zorder=5)
    
    # Asignar colores a los valores negativos y positivos utilizando un colormap divergente (RdBu)
    result_df.plot(column='diferencia_2020_2000', cmap='RdBu', edgecolor='white', linewidth=0.25,
                   norm=Normalize(-max_diff, max_diff), legend=True, ax=ax, legend_kwds={'aspect': 20}, alpha = 0.6)

    ctx.add_basemap(ax, crs=result_df.crs, source='https://tiles.stadiamaps.com/tiles/stamen_toner_lite/{z}/{x}/{y}{r}.png?api_key=a1b11e3f-26c1-414c-8c58-1f8b74f719d0')
    print(result_df.crs)
    #buffer_geometry_gdf.boundary.plot(color = 'black', edgecolor = 'black', ax = ax)
    #zona_metropolitana.to_crs(result_df.crs).plot(ax=ax, color='none', edgecolor='black', linewidth=0.25, alpha = 0.25) 
    ax.scatter([target_x], [target_y], marker=(5, 1), c='red', s=200, zorder=5)
    #place_2020.to_crs("EPSG:6365").boundary.plot(ax=ax, color='none', edgecolor='black', linewidth=0.7)  # Contorno de las AGEBs sin color
    ax.set_xticks([])  # Eliminar marcas del eje x
    ax.set_yticks([])  # Eliminar marcas del eje y
    ax.set_xlim(xmin, xmax)
    ax.set_ylim(ymin, ymax)    
    plt.title('Diferencia de Población 2020-2000 en {}'.format(PLACE_NAME))
    plt.savefig(path_data_modeling+PLACE_NAME.lower()+'_grafico_diferencia_2020_2000.png')
    plt.show()


    # Configuración de la gráfica para Diferencia 2020-1990
    fig, ax = plt.subplots(1, 1, figsize=(15, 15))
    
    # Encontrar el valor máximo absoluto de la diferencia para normalizar el colormap
    #max_diff = abs(result_df['diferencia_2020_2000']).max()
    #max_diff = abs(result_df['diferencia_2020_2000'].min())

    ax.scatter([target_x], [target_y], marker=(5, 1), c='red', s=200, zorder=5)
    
    # Asignar colores a los valores negativos y positivos utilizando un colormap divergente (RdBu)
    result_df.plot(column='diferencia_2020_1990', cmap='RdBu', edgecolor='white', linewidth=0.25,
                   norm=Normalize(-max_diff, max_diff), legend=True, ax=ax, legend_kwds={'aspect': 20}, alpha = 0.6)

    ctx.add_basemap(ax, crs=result_df.crs, source='https://tiles.stadiamaps.com/tiles/stamen_toner_lite/{z}/{x}/{y}{r}.png?api_key=a1b11e3f-26c1-414c-8c58-1f8b74f719d0')
    print(result_df.crs)
    #buffer_geometry_gdf.boundary.plot(color = 'black', edgecolor = 'black', ax = ax)
    #zona_metropolitana.to_crs(result_df.crs).plot(ax=ax, color='none', edgecolor='black', linewidth=0.25, alpha = 0.25) 
    ax.scatter([target_x], [target_y], marker=(5, 1), c='red', s=200, zorder=5)
    #place_2020.to_crs("EPSG:6365").boundary.plot(ax=ax, color='none', edgecolor='black', linewidth=0.7)  # Contorno de las AGEBs sin color
    ax.set_xticks([])  # Eliminar marcas del eje x
    ax.set_yticks([])  # Eliminar marcas del eje y
    ax.set_xlim(xmin, xmax)
    ax.set_ylim(ymin, ymax)    
    plt.title('Diferencia de Población 2020-1990 en {}'.format(PLACE_NAME))
    plt.savefig(path_data_modeling+PLACE_NAME.lower()+'_grafico_diferencia_2020_1990.png')
    plt.show()


    # Configuración de la gráfica para Diferencia 2000-1990
    fig, ax = plt.subplots(1, 1, figsize=(15, 15))
    
    # Encontrar el valor máximo absoluto de la diferencia para normalizar el colormap
    #max_diff = abs(result_df['diferencia_2020_2000']).max()
    #max_diff = abs(result_df['diferencia_2020_2000'].min())

    ax.scatter([target_x], [target_y], marker=(5, 1), c='red', s=200, zorder=5)
    
    # Asignar colores a los valores negativos y positivos utilizando un colormap divergente (RdBu)
    result_df.plot(column='diferencia_2000_1990', cmap='RdBu', edgecolor='white', linewidth=0.25,
                   norm=Normalize(-max_diff, max_diff), legend=True, ax=ax, legend_kwds={'aspect': 20}, alpha = 0.6)

    #ctx.add_basemap(ax, crs=result_df.crs, zoom=10, source=ctx.providers.Stamen.TonerLite)
    #ctx.add_basemap(ax, crs=result_df.crs, zoom=10, source=ctx.providers.Esri.WorldGrayCanvas)
    ctx.add_basemap(ax, crs=result_df.crs, source='https://tiles.stadiamaps.com/tiles/stamen_toner_lite/{z}/{x}/{y}{r}.png?api_key=a1b11e3f-26c1-414c-8c58-1f8b74f719d0')
    
    print(result_df.crs)
    #buffer_geometry_gdf.boundary.plot(color = 'black', edgecolor = 'black', ax = ax)
    #zona_metropolitana.to_crs(result_df.crs).plot(ax=ax, color='none', edgecolor='black', linewidth=0.25, alpha = 0.25) 
    ax.scatter([target_x], [target_y], marker=(5, 1), c='red', s=200, zorder=5)
    #place_2020.to_crs("EPSG:6365").boundary.plot(ax=ax, color='none', edgecolor='black', linewidth=0.7)  # Contorno de las AGEBs sin color
    ax.set_xticks([])  # Eliminar marcas del eje x
    ax.set_yticks([])  # Eliminar marcas del eje y
    ax.set_xlim(xmin, xmax)
    ax.set_ylim(ymin, ymax)    
    plt.title('Diferencia de Población 2000-1990 en {}'.format(PLACE_NAME))
    plt.savefig(path_data_modeling+PLACE_NAME.lower()+'_grafico_diferencia_2000_1990.png')
    plt.show()

    # Configuración de la gráfica para Diferencia 2010-2000
    fig, ax = plt.subplots(1, 1, figsize=(15, 15))
    
    # Encontrar el valor máximo absoluto de la diferencia para normalizar el colormap
    #max_diff = abs(result_df['diferencia_2020_2000']).max()
    #max_diff = abs(result_df['diferencia_2020_2000'].min())

    ax.scatter([target_x], [target_y], marker=(5, 1), c='red', s=200, zorder=5)
    
    # Asignar colores a los valores negativos y positivos utilizando un colormap divergente (RdBu)
    result_df.plot(column='diferencia_2010_2000', cmap='RdBu', edgecolor='white', linewidth=0.25,
                   norm=Normalize(-max_diff, max_diff), legend=True, ax=ax, legend_kwds={'aspect': 20}, alpha = 0.6)

    ctx.add_basemap(ax, crs=result_df.crs, source='https://tiles.stadiamaps.com/tiles/stamen_toner_lite/{z}/{x}/{y}{r}.png?api_key=a1b11e3f-26c1-414c-8c58-1f8b74f719d0')
    print(result_df.crs)
    #buffer_geometry_gdf.boundary.plot(color = 'black', edgecolor = 'black', ax = ax)
    #zona_metropolitana.to_crs(result_df.crs).plot(ax=ax, color='none', edgecolor='black', linewidth=0.25, alpha = 0.25) 
    ax.scatter([target_x], [target_y], marker=(5, 1), c='red', s=200, zorder=5)
    #place_2020.to_crs("EPSG:6365").boundary.plot(ax=ax, color='none', edgecolor='black', linewidth=0.7)  # Contorno de las AGEBs sin color
    ax.set_xticks([])  # Eliminar marcas del eje x
    ax.set_yticks([])  # Eliminar marcas del eje y
    ax.set_xlim(xmin, xmax)
    ax.set_ylim(ymin, ymax)    
    plt.title('Diferencia de Población 2010-2000 en {}'.format(PLACE_NAME))
    plt.savefig(path_data_modeling+PLACE_NAME.lower()+'_grafico_diferencia_2010_2000.png')
    plt.show()

