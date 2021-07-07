import pandas as pd
import gzip
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import Pipeline
from sklearn.neural_network import MLPRegressor

with gzip.open('ar_properties.csv.gz') as f:
    dataset = pd.read_csv(f)
    
#%% PREPROCESAMIENTO DE DATOS
#FILTRAMOS EL DATASET PARA SABER ALQUILERES EN CAPITAL FEDERAL
dataset = dataset[(dataset.l2 == 'Capital Federal') & 
                  (dataset.operation_type == 'Alquiler') &
                  (dataset.price > 0) &
                  (dataset.currency == 'ARS')]

#FILTRO TIPO DE PROPIEDAD PORQUE NO ME QUIERO MUDAR A UNA OFICINA
dataset = dataset[dataset.property_type == ('Departamento' or 'PH' or 'Casa')]

ubicacion = dataset[['lat', 'lon']]

#ELIMINAMOS COLUMNAS INNECESARIAS
drop = ['id', 
        'ad_type', 
        'start_date', 
        'end_date', 
        'created_on', 
        'title', 
        'description', 
        'l1', 'l2', 'l4',  
        'l5', 'l6', 
        'lat', 'lon', 
        'price_period', 
        'operation_type', 
        'currency']

dataset = dataset.drop(drop, axis=1)

#APLICO ONE HOT ENCODING A LOS BARRIOS Y TIPO DE PROPIEDAD
barrios = pd.get_dummies(dataset['l3'])
tipos = pd.get_dummies(dataset['property_type'])

dataset = dataset.drop('l3', axis=1)
dataset = dataset.drop('property_type', axis=1)

dataset = dataset.join(barrios)
dataset = dataset.join(tipos)

#DIVIDO TRAINING Y TEST SET
x_train = dataset.iloc[0:25000].drop('price', axis=1)
y_train = dataset.iloc[0:25000:]['price']

x_test = dataset.iloc[25001:].drop('price', axis=1)
y_test = dataset.iloc[25001:]['price']

#RELLENO Y NORMALIZO CON VALORES DEL TEST SET
def pad_and_normalize(column, media, desv):
    '''
    Remplaza valores vacios de una columna por la media y normaliza
    '''
    column = column.fillna(media)
    column = (column - media) / desv
    return column

(sc_media, sc_desv) = (x_train['surface_covered'].mean(), x_train['surface_covered'].std())
(st_media, st_desv) = (x_train['surface_total'].mean(), x_train['surface_total'].std())

x_train['surface_covered'] = pad_and_normalize(x_train['surface_covered'], sc_media, sc_desv)
x_train['surface_total'] = pad_and_normalize(x_train['surface_total'], st_media, st_desv)
x_train[['rooms', 'bedrooms', 'bathrooms']] = x_train[['rooms', 'bedrooms', 'bathrooms']].fillna(0)

x_test['surface_covered'] = pad_and_normalize(x_test['surface_covered'], sc_media, sc_desv)
x_test['surface_total'] = pad_and_normalize(x_test['surface_total'], st_media, st_desv)
x_test[['rooms', 'bedrooms', 'bathrooms']] = x_test[['rooms', 'bedrooms', 'bathrooms']].fillna(0)

#%% CREANDO Y ENTRENANDO EL MODELO LINEAL
def entrenar_rlineal():
    modelo = LinearRegression()
    modelo.fit(x_train, y_train)
    y_pred = modelo.predict(x_test)

    train_score = modelo.score(x_train, y_train)
    test_score = modelo.score(x_test, y_test)
    
    print('Testing linear model...')
    print('Training score: ', train_score)
    print('Testing score: ', test_score)
    return modelo

#%% AGREGANDO FEATURES POLINOMICAS

def entrenar_poly(n):
    modelo = Pipeline([('poly', PolynomialFeatures(degree=n)), 
                       ('linear', LinearRegression())
                       ])
    modelo.fit(x_train, y_train)
    train_score = modelo.score(x_train, y_train)
    test_score = modelo.score(x_test, y_test)
    
    print(f'Degree: {n}')
    print('Training score: ', train_score)
    print('Testing score: ', test_score)
    return modelo

#%% RED NEURONAL

def entrenar_red(n):
    '''
    Entrena una red 
    n: tupla con en numero de unidades por capa    
    '''
    nn = MLPRegressor(hidden_layer_sizes=n, 
                      activation = 'relu', 
                      solver='adam', 
                      alpha=0.0001, 
                      random_state=1
                      )
    nn.fit(x_train, y_train)
    train_score = nn.score(x_train, y_train)
    test_score = nn.score(x_test, y_test)
    
    print(f'Neural Net with {n} units')
    print('Training score: ', train_score)
    print('Testing score: ', test_score)
    return nn

if __name__ == '__main__':
    
    modelo_lineal = entrenar_rlineal()
    modelo_polinomico = entrenar_poly(2)
    red_simple = entrenar_red((100))
    red_doble = entrenar_red((100, 50))
    red_triple = entrenar_red((100, 50, 25))