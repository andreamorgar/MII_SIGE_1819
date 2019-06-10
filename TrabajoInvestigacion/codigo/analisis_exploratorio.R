## ------------------------------------------------------------------------
library(tidyverse)
library(tm)
library(wesanderson)
library(rword2vec) #install_github("mukul13/rword2vec")

## ------------------------------------------------------------------------
data.train <- read.csv("datos/train.csv")
head(data.train, 5)
# La columna de texto que nos interesa es "Description"

## ------------------------------------------------------------------------
data.train %>% 
  count(AdoptionSpeed) %>%
    ggplot(aes(x = factor(AdoptionSpeed), y = n, fill = factor(AdoptionSpeed))) +
        geom_col(color = "black") +
        geom_text(aes(label = n), position = position_stack(vjust = 0.5)) +
        scale_fill_brewer(palette="Paired") +
        labs(x = "Tiempo que se tarda en adoptar", y = "Cantidad de mascotas") +
        ggtitle("Adopciones") +
        theme(legend.position = "none", plot.title = element_text(hjust = 0.5, face = "bold", size=20))

# Sino me equivoco el 0 es el mínimo tiempo en adoptar y el 4 es el máximo tiempo en adoptar

## ------------------------------------------------------------------------
dim(data.train)

## ------------------------------------------------------------------------
library(tidyverse)
library(readr)
library(RColorBrewer)
library(tidytext)
library(wordcloud)
library(tm)

data_new <- read.csv(file="datos/train.csv")

data_new$items <- as.character(data_new$Description) # ordena_mayor_matrix_corpus

# Función para crear el wordcloud 
cloud_negative_positive <- function(data){
#   
  drugstext <- unnest_tokens(data, word, items)
# 
  binarytextscore <- get_sentiments(lexicon = "bing")
#     
  drugscloudbinary <- drugstext %>%
    inner_join(binarytextscore, by = "word") %>%
     count(word, sentiment) %>%
     mutate(color = ifelse(sentiment == "positive", "darkgreen", "red"))
 
   drugscloudbinary
}
#
res <- cloud_negative_positive(data_new)
wordcloud(res$word, res$n, random.order = FALSE, colors = res$color, ordered.colors = TRUE)

## ------------------------------------------------------------------------
# Nos quedamos con la única columna del dataset que nos interesa. 
# Necesitamos obtenerla en forma de vector, y no como un dataframe de una columna, 
# por lo que usamos as.vector para hacer la conversión
data.train.description <- data.train$Description
data.train.description <- as.vector(data.train.description)

# Lo convertimos en la estructura de documento, y lo guardamos ya en el corpus 
# que lo vamos a utilizar
data.train.description.corpus <- (VectorSource(data.train.description))

# Creamos el propio corpus
data.train.description.corpus <- Corpus(data.train.description.corpus)

## ------------------------------------------------------------------------
# Mostrar el contenido por pantalla
data.train.description.corpus[[4]]$content
data.train.description.corpus[[7]]$content

## ------------------------------------------------------------------------
# Frecuencias para benefits en train
# https://rstudio-pubs-static.s3.amazonaws.com/40817_63c8586e26ea49d0a06bcba4e794e43d.html

# Cargamos la librería
# Calculamos la matriz de términos
dtm <- DocumentTermMatrix(data.train.description.corpus)

# Calculamos la frecuencia
freq <- sort(colSums(as.matrix(dtm)), decreasing=FALSE)
wf <- data.frame(word=names(freq), freq=freq)

# Dibujamos el histograma
subset(wf, freq>3050)    %>%
  ggplot(aes(word, freq)) +
  geom_bar(stat="identity", fill="darkred", colour="black") +
  theme(axis.text.x=element_text(angle=45, hjust=1)) + 
  ggtitle("Frecuencias para Description en train") 

## ------------------------------------------------------------------------
# Convertimos la columna Descripcti train a minúsculas
data.train.description.corpus <- tm_map(data.train.description.corpus, content_transformer(tolower))

## ------------------------------------------------------------------------
# Mostrar el contenido por pantalla
data.train.description.corpus[[4]]$content
data.train.description.corpus[[7]]$content

## ------------------------------------------------------------------------
data.train.description.corpus <- tm_map(data.train.description.corpus, content_transformer(removePunctuation))

## ------------------------------------------------------------------------
# Mostrar el contenido por pantalla
data.train.description.corpus[[4]]$content
data.train.description.corpus[[7]]$content

## ------------------------------------------------------------------------
# Datos train
data.train.description.corpus <- tm_map(data.train.description.corpus, content_transformer(removeWords), stopwords("english"))

## ------------------------------------------------------------------------
# Mostrar el contenido por pantalla
data.train.description.corpus[[4]]$content
data.train.description.corpus[[7]]$content

## ------------------------------------------------------------------------
data.train.description.corpus

## ------------------------------------------------------------------------
# Obtenemos su matriz de términos
matrix.train.description.corpus <- TermDocumentMatrix(data.train.description.corpus)

# No tenemos los datos en la matriz que buscamos, sino en un vector
# por tanto, lo convertimos en matriz
matrix.train.description.corpus <- as.matrix(matrix.train.description.corpus)

# Sumamos las filas para obtener la frecuencia de una palabra en benefitsReview
matrix.train.description.corpus <- rowSums(matrix.train.description.corpus)

# Ordenamos de mayor a menor los términos
terms.frecuency.train.description.corpus <- sort(matrix.train.description.corpus, decreasing = TRUE)
terms.frecuency.train.description.corpus <- terms.frecuency.train.description.corpus[1:length(data.train.description.corpus)]
terms.frecuency.train.description.corpus

## ------------------------------------------------------------------------
graph.terms.frecuency.train.description.corpus <- as.matrix(terms.frecuency.train.description.corpus)
barplot(graph.terms.frecuency.train.description.corpus[1:4,],  xlab="Términos", ylab="Número de frecuencia",
        col=wes_palette(n=4, name="Zissou1"))

## ------------------------------------------------------------------------
# Convertimos a matriz "terms.frecuency.train.description.corpus"
terms.frecuency.train.description.corpus <- as.matrix(terms.frecuency.train.description.corpus)
terms.frecuency.train.description.corpus

# Me quedo solo con los términos
terms.train.description.corpus <- rownames(terms.frecuency.train.description.corpus)
terms.train.description.corpus

## ------------------------------------------------------------------------
# PETA HASTA QUE NO SE LIMPIE MUCHO MÁS

data.train$Description[1:2000]

# Escribo en un fichero la columna "description"
write.table(data.train$Description[1:4000], "description.txt", sep = "\t", quote = F, row.names = F)

# Entreno los datos del texto para obtener los vectores de palabras
model.train.description = word2vec(train_file = "description.txt", output_file = "description.bin", binary=1)

dist_terms_benefits_train_corpus = c()
# Obtengo la distancia de las 100 palabras con mayor frecuencia
for (i in 1:length(terms.train.description.corpus)){ # calculamos la distancia de la palabra a sus sinónimos
  dist_terms_benefits_train_corpus[i] = distance(file_name = "description.bin", 
                                                 search_word = terms.train.description.corpus[i], num = 2)
}

# guardamos en un fichero
library(rlist)
list.save(dist_terms_benefits_train_corpus, 'dist_terms.RData')
dist_terms_benefits_train_corpus_new <- list.load('dist_terms.RData')
dist_terms_benefits_train_corpus_new[]

terms.train.description.corpus[3]
dist_terms_benefits_train_corpus_new[[3]]

terms.train.description.corpus[7]
dist_terms_benefits_train_corpus_new[[7]]

terms.train.description.corpus[10]
dist_terms_benefits_train_corpus_new[[10]]

terms.train.description.corpus[100]
dist_terms_benefits_train_corpus_new[[100]]

