# Dudas trabajo práctica 2

1. ¿Lo entrega sólo una persona de la pareja?


# Anotaciones

### 9 de Mayo
- La columna que nos interesa es **Description** y he estado mirando, y es igual que TID, podríamos mezclar lo de TID y lo aprendido en SIGE.

### 12 de Mayo
- Supongo que la columna que nos interesa es predecir es la de **AdoptionSpeed**, el tiempo que tarda en adoptar a la mascota, y va entre 0 (tiempo mínimo) y 4 (tiempo máximo).
- Por otro lado, al coger la columna de texto, existen signos de puntuación que separan palabras, por lo que al quitarlo las dos palabras se juntan, por tanto tendríamos que poner espacios.
  - (mazuvil)or --> mazuvilor
  - hometown.there --> hometownthere
- He estado probando lo de los sinónimos sin realizar mucho preprocesamiento al texto, y peta (me aborta la sesión), dice que "Words in train file: 897576" y "Error: C stack usage  17589390136116 is too close to the limit". Así que debemos hacer una buena limpieza, si queremos intentar meter lo de agrupación de sinónimos.


# Estructura del trabajo

## Preprocesamiento al texto

1. Crear el corpus
2. Eliminar signos de puntuación y demás
3. Conversión de las mayúsculas a las minúsculas
4. Eliminación de Stopwords
5. Agrupación de sinónimos
6. TF-IDF
7. Stemming
8. Lematización
9. Reglas de asociación
10. Valores perdidos
11. Borrar espacios en blanco
12. Sparsity


# Referencias interesantes

- https://www.tidytextmining.com/
- http://www.mjdenny.com/Text_Processing_In_R.html


# Código

Contenido de una celda de la vrariable Description:

Nibble is a 3+ month old ball of cuteness. He is energetic and playful. I rescued a couple of cats a few months ago but could not get them neutered in time as the clinic was fully scheduled. The result was this little kitty. I do not have enough space and funds to care for more cats in my household. Looking for responsible people to take over Nibble's care.

~~~{r}
data.train <- read.csv("datos/train.csv")
head(data.train, 5)
# La columna que nos interesa es "Description"
~~~
