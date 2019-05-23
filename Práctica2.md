# Sistemas Inteligentes para la Gestión en la Empresa

## Práctica 2: Deep-Learning para multi-clasificación

- Con los datos que tenemos, predecir el tiempo que se tardó, es decir, predecir el tiempo que tardaron en adoptar a esta mascota. Está discretizado el tiempo que se tarda en adoptar entre 0 (mínimo) y 4 (máximo).

- En el dataset a parte de darnos las características de los datos, nos dan varias fotos de los animales. Para cada uno de ellos tenemos varias imágenes asociadas.

- Predecir el tiempo que se va a tardar en adoptar, usando solo las imágenes. Vamos a usar todas las imagenes, esas imagenes estan asociadas a las clases, que es el tiempo que se tardo en adoptar. Cada mascota tiene más de una imagen. Nos da el conjunto ya organizado, tenemos subcarpteas para cada una de las clases, para esta práctica solo vamos a usar las imágenes. Así que un poco, puede ser: ¿cuánto de adorable es el animal?

- Hay unos dos GB de fotos de animales.

- Para los datos test, no sabemos el valor de la salida de esas clases

#### Mejoras

**Ajustar la función de de coste:**

- Es más dificil porque en el tipo de problema que nosotros tenemos, ccomo la entropía cruzada, estamos dando el mismo peso a todos los errores, si en vez de la categoría 1 decimos que es la categoría 2, tenemos que ver que una función que no penalice todos los errores por igual, sino que cuanto más diferenica de error haya, más penalización haga ---> ajuste cuadratico????????

categoria 0    0 0 0
categoria 1    0 0 1
               0 1 0

- NO DIERA la misma perdida

- Se ponga como numérico, se ajuste cuadratico, otra forma es la entropía cruzada.

- Esa función de coste no esta implentada en Keras, keras permite hacer tu propia función

- Le interesa el razonamiento que hagamos más que los resultados, pero esté bien razonada, si conseguimos una mejora


**Ajuste de del algoritmoe de optimización**

- Probar el ada, para ver si funciona mejor

- Lo interesante es ver si converge uno antes que otros

- Si el proceso de entrenamiento converge antes en unos algoritmos que en otros


**Data augmentation**
- Aumentar los datos, hacer transformaciones


**Trasfer learning**
- Juntar trozos de varias redes


**Fine tunnig**


**Uso de datos adicionales**
- Tenemos las imagenes con las que estamos trabajando, pero tambien tenemos los metadatos sobre los animales (datos adicionaes), podemos estudiar como incorporar esos datos en nuestro estudio.

- Podemos tener un clasificador con las imagenes y otro con la tabla solo y poder combinar ambos modelos para ver como podemos hacerlo, como conseguir mejores resultados.

- Si son dos redes neuronales, es más fácil, me monto o cojo el trozo de una , el trozo de otra...

- _Ejemplo_: dos modelos, y calcula la media de ambos (ensamble), usando las capas. Tu lo que haces en esa red, coge las la parte que tienes entrenada

- Coger metadatos chiquititas

**Lo contado hasta aquí es lo referente en la práctica 2**

-----

## Trabajo de investigación: Deep-Learning

- Ampliar nuestros conocimientos del trabajo que vayamos haciendo, de deep learning y aa, en respecto a la práctica.

- Hemos desarollado la práctica, hemos llegado a una solución, y aquí ampliamos en una línea determinada más compleja y lo acompañamos de una parte teórica

- La implemetacion relacionada con la misma idea de los que hemos hecho con la práctica 2

- Un estudo más elaborado, como se materializaría dentro de la práctica


#### Estrategias de binarización

- En vez de ir a un multiclase, probar algunas estrategias de binarización, a ver si tengo mejores resultados haciendo uso de one-vs-one

- Entre adopcion rapida y adopcion lenta

- A ver si podemos afinar dentro de ellos

- "Estan son las opciones que tenemos, mi forma de la práctica es esta, y hasta aqui he llegado"


#### Procesamiento de texto

- Nos dan una descripción de la mascota y nos dan un anuncio de lo que es cada uno. Entre los datos adicionales de los textos, tambien nos dan un análisis de sentimientos básicos, y lo han pasado por una API de Google y te dice que en grado de emoción expresa el texto del animal, si es más postivo que negativo.

  - si el mensaje es muy positivo nos emociona más a la hora de quedarnos con él
  - si el mensaje es muy negativo nos emociona porque ha sufrido

- En lugar de hacer una análisis de sentimientos bajos, hacer un _embeding_, sacárselo a la red, recursos externos, con keras se puede hacer, que modelos se pueden usar para procesar texto

#### Algoritmos de optimización, técnicas de regularización, dropout...

- Hacer un estudio de como aplicar estas tecnicas para evitar el sobreaprendizaje

#### Visualización

- Para ver las redes y tal, y como ha sido el proceso de entrenamiento
  - TensorBoadr, se instala con uan libreria adicional
  - PlotNeuralNet para hacer dibujos bonitos de redes neuronales

----

**Fecha de entrega : 9 de junio**

De la parte de la práctica una memoria en la mísma linea que la práctica 1

Para el trabajo hay que hacer un trabajo con la explicación, no es necesario hacer dos memorias, se puede hacer todo en una.

El día del examen se hace una presentanción del trabjo, yo uso tal, parte más correspondiente al trabajo de unos 15 minutos o así.
