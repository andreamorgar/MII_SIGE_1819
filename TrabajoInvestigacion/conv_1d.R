## conv_1d.R ##

# 1D CONVENTS #



# Importar las librerías necesarias
# --------------------------------------------------------------------------------------------------

library(caret)
library(keras)



# Lectura de los datos
# --------------------------------------------------------------------------------------------------

# Se leen y cargan los datos del CSV. Debemos de tener los mismos conjuntos 
data.train <- read.csv("datos/train.csv")

# Particionamiento de los datos
trainIndex <- createDataPartition(data.train$AdoptionSpeed, p = .8, list = FALSE, times = 1)
train <- data.train[trainIndex, ] 
test   <- data.train[-trainIndex, ]

# Cogemos las etiquetas y el campo de texto
labels_train <- train$AdoptionSpeed
texts_train <- train$Description



# Tokenizing the text of the raw Description data
# --------------------------------------------------------------------------------------------------

maxlen <- 100 
training_samples <- 200  
validation_samples <- 10000 
max_words <- 10000 

tokenizer <- text_tokenizer(num_words = max_words) %>%
  fit_text_tokenizer(texts_train)
sequences <- texts_to_sequences(tokenizer, texts_train)
word_index = tokenizer$word_index
cat("Found", length(word_index), "unique tokens.\n")

data <- pad_sequences(sequences, maxlen = maxlen)
labels <- as.array(labels_train)

cat("Shape of data tensor:", dim(data), "\n")
cat('Shape of label tensor:', dim(labels), "\n")

indices <- sample(1:nrow(data)) 
training_indices <- indices[1:training_samples]
validation_indices <- indices[(training_samples + 1):
                                (training_samples + validation_samples)]
x_train <- data[training_indices,]
y_train <- labels[training_indices]
x_val <- data[validation_indices,]
y_val <- labels[validation_indices]



# Parsing the GloVe word-embeddings file
# --------------------------------------------------------------------------------------------------

# https://nlp.stanford.edu/projects/glove/
glove_dir = "glove"
lines <- readLines(file.path(glove_dir, "glove.6B.100d.txt"))

embeddings_index <- new.env(hash = TRUE, parent = emptyenv())
for (i in 1:length(lines)) {
  line <- lines[[i]]
  values <- strsplit(line, " ")[[1]]
  word <- values[[1]]
  embeddings_index[[word]] <- as.double(values[-1])
}

cat("Found", length(embeddings_index), "word vectors.\n")



# Listing 6.11. Preparing the GloVe word-embeddings matrix
# --------------------------------------------------------------------------------------------------

embedding_dim <- 100
embedding_matrix <- array(0, c(max_words, embedding_dim))

for (word in names(word_index)) {
  index <- word_index[[word]]
  if (index < max_words) {
    embedding_vector <- embeddings_index[[word]]
    if (!is.null(embedding_vector))
      embedding_matrix[index+1,] <- embedding_vector
  }
}



# Model definition
# --------------------------------------------------------------------------------------------------

model <- keras_model_sequential() %>%
  layer_embedding(input_dim = max_words, output_dim = embedding_dim, input_length = maxlen) %>%
  layer_conv_1d(filters = 64, kernel_size = 7, activation = "relu") %>%
  layer_max_pooling_1d(pool_size = 5) %>%
  layer_dropout(0.3) %>%
  layer_flatten() %>%
  layer_dense(units = 64, activation = "relu") %>%
  layer_dense(units = 64, activation = "relu") %>%
  layer_dense(units = 5, activation = "sigmoid")

summary(model)



# Loading pretrained word embeddings into the embedding layer
# --------------------------------------------------------------------------------------------------

get_layer(model, index = 1) %>%
  set_weights(list(embedding_matrix)) %>%
  freeze_weights()



# Training and evaluation
# --------------------------------------------------------------------------------------------------

model %>% compile(
  optimizer = "adam", # Mejora un poco con 'adam' que con rmsprop
  loss = "sparse_categorical_crossentropy",
  metrics = c("accuracy")
)

# Queremos obtener el tiempo que tarda en entrenar
tiempo.word.embedding <- proc.time() 

# Entrenamos
history <- model %>% 
  fit (
    x_train, y_train,
    epochs = 15,
    batch_size = 100,
    validation_split = 0.15,
    validation_data = list(x_val, y_val)
  )

# Obtenemos el tiempo de entrenamiento
proc.time()-tiempo.word.embedding   

# Información del entrenamiento
print(tiempo.word.embedding)

save_model_weights_hdf5(model, "datos/conv_1d.h5")



# Tokenizing the data of the test set
# --------------------------------------------------------------------------------------------------

labels_test <- train$AdoptionSpeed
texts_test <- train$Description

sequences <- texts_to_sequences(tokenizer, texts_test)
x_test <- pad_sequences(sequences, maxlen = maxlen)
y_test <- as.array(labels_test)



# Evaluating the model on the test set
# --------------------------------------------------------------------------------------------------

model %>% evaluate(x_test, y_test)



