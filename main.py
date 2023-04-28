
from collections import defaultdict
import numpy as np
import pandas as pd
import os
from surprise import Dataset, Reader, accuracy, KNNWithMeans
from surprise.model_selection import train_test_split, cross_validate, GridSearchCV
from sklearn.model_selection import train_test_split

USER = 123 # Usuario al que vamos a dirigir la recomendación 
NUM_PREDICTION_REQUEST = 3 # Número de peliculas que queremos que devuelva la función
MIN_RATING_PRECISION = 4 # Rating minímo de la recomendación

# UTILS
def itemIdToName(iid):
  df = pd.read_csv('./ml-100k/u.item', usecols=[0, 1], sep='|', names=['item_id', 'item_title'])
  title = df.loc[df['item_id'] == iid, 'item_title'].iloc[0]
  return title

def getDatasetFromMovielens():
  # Ruta del fichero donde se almacenan los datos requeridos
  file_path = os.path.expanduser('./ml-100k/u.data')

  # Instancia de la clase Reader para poder cargar los datos de manera que el metodo load_from_file pueda procesarlos
  reader = Reader(line_format="user item rating timestamp", sep="\t")

  # Almacenamos los datos del fichero como un dataset de surprise
  data = Dataset.load_from_file(file_path, reader=reader)

  # Devolvemos el dataset
  return data

def getDataForContentBasedAlgo():
  # Obtenemos los identificadores de los usuarios, las peliculas y el rating de cada uno
  usersColumnNames = ['user_id', 'movie_id', 'rating']
  usersDf = pd.read_csv('./ml-100k/u.data', usecols=[0, 1, 2], sep='\t', names=usersColumnNames)

  # Obtenemos los datos de generos disponibles
  moviesGenresColumnNames = ['genre_name', 'genre_id']
  movieGenres = pd.read_csv('./ml-100k/u.genre', usecols=[0], sep='|', names=moviesGenresColumnNames)

  # Obtenemos el nombre de las columnas que tendrá la matriz de peliculas
  moviesColumnNames = pd.concat([pd.Series(['movie_id', 'movie_title']), movieGenres['genre_name']])

  # Obtenemos los datos de las peliculas
  # id, title, genders (los generos están en asignados en columnas por el método One Hot encoder, donde cada columna indica si tiene ese género (1) o no (0))
  moviesDf = pd.read_csv('./ml-100k/u.item', usecols=[0, 1, *range(5, 24)], sep='|', names=moviesColumnNames)

  # Filtramos para recoger solo las que el usuario haya visto
  userDf = usersDf.query(f'user_id == {USER}')

  # Filtramos las peliculas para obtener solo aquellas que haya visto el usuario asignado
  moviesWatchedByUser = moviesDf[moviesDf.iloc[:, 0].isin(userDf['movie_id'])]

  return (userDf, movieGenres, moviesDf, moviesWatchedByUser)


# CONTENT BASED ALGORITHM
def contentBasedAlgorithm(returnOnlyIdAndScore = False, userDf= '', moviesDf='', moviesWatchedByUser= '', movieGenres= ''):

  # Unimos los dos dataframes según el id de la pelicula
  entryUserCalification = pd.merge(userDf, moviesWatchedByUser, on='movie_id')

  # Este dataframe muestra el id de la pelicula, el rating de la pelicula, el titulo, y el genero
  # Nos servirá para obtener la matriz de peso del usuario para obtener el perfil de usuario en función de las peliculas que ha visto
  weightMatrix = pd.DataFrame(columns=['movie_id', *movieGenres['genre_name']])

  for index, row in entryUserCalification.iterrows():
    genresValues = row[movieGenres['genre_name']].values
    dotProduct = np.dot(row['rating'], genresValues)
    rowToInsertInMatrix = np.insert(dotProduct, 0, row['movie_id'])
    rowToInsertInMatrix = rowToInsertInMatrix.tolist()

    weightMatrix.loc[len(weightMatrix)] = rowToInsertInMatrix

  # Una vez tenemos la matriz de peso, creamos el perfil de usuario, que nos dirá a que género es más afin
  # Para ello, primero sumamos los valores de cada una de las columnas
  userProfile = pd.DataFrame(columns=[*movieGenres['genre_name']])
  for col in weightMatrix.columns:
    if col == 'movie_id': continue
    userProfile.at[0, col] = weightMatrix[col].sum()

  # Y a continuación, realizamos una ponderación del valor de cada columna, respecto al valor total de todas las columnas
  totalColumnValue = userProfile.sum(axis=1).values

  for col in userProfile.columns:
      userProfile[col] = userProfile[col] / totalColumnValue

  # Finalmente, una vez tenemos el perfil de usuario, filtramos aquellas peliculas que ya ha visto el usuario
  # De manera que solo tengamos en cuenta aquellas que no ha visto, estas serán las peliculas candidatas a recomendar
  maskForRemoveMoviesWatched = moviesDf['movie_id'].isin(moviesWatchedByUser['movie_id'])
  candidateMovies = moviesDf.drop(moviesDf[maskForRemoveMoviesWatched].index)

  # Multiplicamos
  # cada uno de los valores de las columnas de peliculas no vistas (candidateMovies)
  # con los valores del perfil de usuario de sus respectivas columnas (userProfile)

  # Creamos una copia, para mantener el genero de las peliculas candidatas definido con 1 y 0,
  # Y otro dataframe con el genero de las peliculas ya multiplicado, para poder comparar
  # (No es necesario hacer este paso)
  moviesWeight = candidateMovies.copy()

  # Y ahora multiplicamos para obtener el peso de las peliculas según su categoria
  for col in candidateMovies.columns:
    if col == 'movie_id' or col == 'movie_title': continue
    moviesWeight[col] = candidateMovies[col].apply(lambda x: x * userProfile[col].values[0])

  # calcular sumatorio de cada fila
  # El resultado nos dará el orden de prioridad de las peliculas
  # Cuanto mayor es el valor, más alto estará en el orden de recomendaciones
  moviesWeight['recommendedPriority'] = moviesWeight.iloc[:, 2:].sum(axis=1)

  # Ordenamos las peliculas por prioridad de recomendación ascendente, y reiniciamos el índice
  # Esto nos ayudará a la hora de devolver las peliculas, en un diccionario
  # El formato devuelto será el órden de la pelicula y el nombre de ésta, mostrando el número de peliculas asignado en la función
  moviesWeight = moviesWeight.sort_values(by="recommendedPriority", ascending=False).reset_index(drop=True)
  moviesWeight.index += 1

  if(returnOnlyIdAndScore):
    return moviesWeight[['movie_id']].head(NUM_PREDICTION_REQUEST)
  
  # Queremos que el primer valor sea el índice de la pelicula, y el segundo valor el título de la pelicula
  returnMoviesRequest = dict(zip(moviesWeight.index[:NUM_PREDICTION_REQUEST], moviesWeight['movie_title'][:NUM_PREDICTION_REQUEST]))

  #formato del resultado: {ordernPrioridad: 'titulo de pelicula'...}
  # Devolvemos el resultado
  return returnMoviesRequest


# PRECISION AND RECALL
def precisionAndRecallOfContentBasedAlgo():
  userDf, movieGenres, moviesDf, moviesWatchedByUser = getDataForContentBasedAlgo();

  # dividir el dataframe en conjuntos de entrenamiento y prueba
  train_df, test_df = train_test_split(moviesWatchedByUser, test_size=0.3, train_size=0.7, shuffle=True)

  # Unimos los dos dataframes según el id de la pelicula
  ratingTestMovies = pd.merge(userDf, test_df, on='movie_id')

  ratingTestMovies = ratingTestMovies.loc[:, ['movie_id', 'rating']]

  topNRecommendendations = contentBasedAlgorithm(returnOnlyIdAndScore=True, userDf=userDf, moviesDf=moviesWatchedByUser, moviesWatchedByUser=train_df, movieGenres=movieGenres)

  prefersMovies = ratingTestMovies.loc[ratingTestMovies['rating'] >= MIN_RATING_PRECISION]

  topNInPrefers = pd.merge(topNRecommendendations, prefersMovies, on='movie_id')

  print(f'Numero de pelicuas del top N que son preferidas: {len(topNInPrefers)}')
  print(f'Numero de peliculas recomendadas: {len(topNRecommendendations)}')
  print(f'Precision: {len(topNInPrefers)/len(topNRecommendendations)}')
  print(f'Recall: {len(topNInPrefers)/len(prefersMovies)}')

def precisionAndRecallOfColaborativeFilter(predictions):

  # Recomendados - preferidos
  tp = 0  # True positives

  # Recomendados - no preferidos
  fp = 0  # false positives

  # No recomendados - preferidos
  fn = 0  # false negatives

  for uid, iid, r_ui, est, _ in predictions:
    if est >= 3.5 and r_ui >= 3.5:
        tp += 1
    if est >= 3.5 and r_ui < 3.5:
        fp += 1
    if est < 3.5 and r_ui >= 3.5:
        fn += 1

  # Capacidad de que todos recomendados sean preferidos
  precision = tp / (tp + fp)

  # Capacidad de que todos los preferidos sean recomendados
  recall = tp / (tp + fn)

  print('PRECISION AND RECALL\n-----------------------')
  print(f'Precisión: {precision}')
  print(f'Recall: {recall}')
  print('-----------------------')


# RECOMMENDER SYSTEMS
def colaborativeFilterRecommender(precisionAndRecall):
  
  # Obtenemos el dataset de movielens
  dataset = getDatasetFromMovielens();

  # Dividimos el dataset en dos subconjuntos
  #   trainset: datos de entrenamiento
  #   testset: datos de prueba o testeo

  # test_size para indicar que tamaño serán destinados a pruebas
  trainset, testset = train_test_split(dataset, test_size=.15)

  # KNNWithMeans: Algoritmo para las predicciones basadas en los k-Vecinos más cercanos a un usuario dado
  # ITEM_BASED: Calculo de items cercanos
  # USER_BASED: Calculo de usuarios cercanos
  # Número de vecinos que queremos utilizar -> 50
  # Medida de similutud -> pearson (También podriamos usar la similitud del coseno (cosine) entre otros)
  # userBasedAlgo = KNNWithMeans(k=50, sim_options={'name': 'pearson_baseline', 'user_based': True})

  # También podríamos usar el método GridSearchCV (más recomendado) que nos proporciona surprise
  # GridSearchCV se usa para realizar una búsqueda de los mejores hiperparámetros en un espacio predefinido de posibles valores
  # Evalua el rendimiento del modelo con diferentes combinaciones de hiperparámetros y selecciona el mejor conjunto de estos valores

  # Predefinimos los hiperparámetros, k para el número de vecinos, y sim_options con los posibles valores para obtener la medida de similitud
  param_grid = {'k': [10, 20, 30], 'sim_options': {'name': ['cosine', 'msd', 'pearson'], 'user_based': [True]}}

  # Creamos el grid search
  gs = GridSearchCV(KNNWithMeans, param_grid, measures=['RMSE', 'MAE', 'FCP', 'MSE'], cv=3)

  # Ajustamos el grid search con los datos
  gs.fit(dataset)

  # Recogemos la mejor combinación de hiperparámetros según el RMSE
  bestHyperparameters = gs.best_params['rmse']

  # Creamos el algoritmo con los parametros que obtenemos del grid search
  userBasedAlgo = KNNWithMeans(k=bestHyperparameters['k'], sim_options=bestHyperparameters['sim_options'])

  # Una vez obtenido el algoritmo, lo entrenamos con el dataset de entrenamiento
  userBasedAlgo.fit(trainset)

  # Comprobamos obteniendo la predicción de un usuario con una pelicula en concreto
  # Enviandole el id del usuario y del item y la calificación real que el usuario asignó al item, nos devolverá la perdicción estimada por el modelo
  # Verbose sirve para decirle al metodo si queremos que nos muestre el resultado, aunque igualmente podemos almacenar el resultado en una variable y luego mostrarlo
  userBasedAlgo.predict(uid=str(USER), iid=str(302), r_ui=4, verbose=True)

  # Realizamos una validacion cruzada para obtener la evaluación del algoritmo
  print('Algorithm evaluation\n-----------------------------')
  cross_validate(userBasedAlgo, dataset, measures=['RMSE', 'MAE', 'FCP', 'MSE'], cv=5, verbose=True)

  # Filtramos los items del subcojnunto de entrenamiento, para obtener solo aquellas peliculas que el usuario no valorado
  noRatingByUserSet = trainset.build_anti_testset()

  # Obtenemos las predicciones de las valoraciones del conjunto de peliculas no valoradas por el usuario
  predictions = userBasedAlgo.test(noRatingByUserSet)

  if precisionAndRecall :
    precisionAndRecallOfColaborativeFilter(predictions)

  # Diccionario que almacenará las valoraciones en forma de listas, inicialmente se encuentra vacío para no ocasionar un KeyError
  predictionDict = defaultdict(list)

  # Iteramos las predicciones, y obtenemos el id y la estimación de valoración de cada pelicula
  for uid, iid, r_ui, est, _ in predictions:
    if uid != str(USER) : continue
    predictionDict[itemIdToName(int(iid))].append(est)

  # Ordenamos los items de mayor a menor según su estimación
  sortedPredictionList = sorted(predictionDict.items(), key=lambda x: x[1], reverse=True)
  
  # Iteramos sobre los items de la lista ordenada de predicciones para obtener un número asignado de items
  # Damos formato para devolver un resultado más legible
  # Formato de la lista: {ordenPrioridad: {'title': 'titulo de la pelicula', 'est': 'valoración estimada de la pelicula}}
  sortedPredictionDict = {}
  for i, (title, est) in enumerate(sortedPredictionList, start=1):
    if i > NUM_PREDICTION_REQUEST : break
    sortedPredictionDict[i] = {'title': title, 'est': str(est[0])}


  # Devolvemos las peliculas
  print('\nRecommendations:\n')
  print(sortedPredictionDict)

  #########################
  # EVALUACIÓN DEL MODELO #
  #########################

  # Testeamos el algoritmo con el set de testeo
  # predictions = userBasedAlgo.test(testset)

  # RMSE
  # Mide el error cuadrático promedio entre los valores reales y los predichos
  # Penaliza los errores grandes y es el más adecuado si el objetivo es minimizarlos ya que se elevan al cuadrado para evitar que haya valores negativos
  # print(accuracy.rmse(predictions, verbose=True))

  # MAE
  # Mide la magnitud promedio de los errores absolutos
  # Penaliza los errores por igual, siendo el más adecuado para minimizarlos todos
  # print(accuracy.mae(predictions, verbose=True))

  # MSE
  # Mide la media, de la diferencia cuadrática entre calificaciones reales y predicciones del modelo
  # print(accuracy.mse(predictions, verbose=True))

  # FCP
  # Compara las calificaciones predichas y las calificaciones reales y cuenta cuántas veces el modelo ha colocado correctamente el elemento por encima de otro en comparación con las calificaciones reales.
  # Útil en situaciones en las que la magnitud de las calificaciones no es importante, solo la capacidad del modelo para identificar la ordenación correcta de elementos recomendados
  # print(accuracy.fcp(predictions, verbose=True))

  # También podriamos realizar una validación cruzada
  # Consisnte en dividir el dataset en k subconjuntos, y realizar k iteraciones utilizando por cada iteración, uno de los subconjuntos como conjunto de prueba y los demás como conjunto de entrenamiento
  # Así, se evalua el modelo k veces utilizando diferentes combinaciones de subconjuntos de entrenamiento / prueba
  # Aclaraciones:
  #   El dataset no debe haberse dividido previamente en subconjuntos
  #   El algoritmo no debe haber sido entrenado
  # cross_validate(algo, data, measures=['RMSE', 'MAE', 'FCP', 'MSE'], cv=5, verbose=True) <- line: 174

def contentBasedRecommender(precisionAndRecall):

  userDf, movieGenres, moviesDf, moviesWatchedByUser = getDataForContentBasedAlgo()

  recommendItems = contentBasedAlgorithm(returnOnlyIdAndScore=False, userDf=userDf, movieGenres=movieGenres, moviesDf=moviesDf, moviesWatchedByUser=moviesWatchedByUser)

  if precisionAndRecall :
    precisionAndRecallOfContentBasedAlgo()

  print('\nRecommendations:\n')
  print(recommendItems)


def main():
  systemRecommender = str(input('Enter type of recommendation system:\n\n1: Content based\n2: Colaborative filter \n\n> '))

  if systemRecommender != '1' and systemRecommender != '2':
    print('\nInvalid input, only 1 or 2 is allowed\n')
    return

  precisionAndRecall = input('\nWant precision and recall? (y/n)\n> ')

  if precisionAndRecall!= 'y' and precisionAndRecall!= 'n':
    print('\nInvalid input, only y or n is allowed\n')
    return

  if systemRecommender == '1':
    if precisionAndRecall == 'y':
      contentBasedRecommender(precisionAndRecall=True)
    else:
      contentBasedRecommender(precisionAndRecall=False)
  
  else:
    if precisionAndRecall == 'y':
      colaborativeFilterRecommender(precisionAndRecall=True)
    else:
      colaborativeFilterRecommender(precisionAndRecall=False)

  otherTime = input('\nDo you want to run another time? (y/n)\n> ')
  if otherTime != 'y' and otherTime != 'n':
    print('\nInvalid input, only y or n is allowed\n')
    return
  else:
    if otherTime == 'y':
      main()
    else:
      print('\nok\n')
      return
  
main()
