# Se leen y cargan los datos del CSV. Debemos de tener los mismos conjuntos 
data.train <- read.csv("./petfinder-adoption-data/train.csv")

library(caret)

# Particionamiento de los datos
trainIndex <- createDataPartition(data.train$AdoptionSpeed, p = .8, list = FALSE, times = 1)
train <- data.train[trainIndex, ] 
test   <- data.train[-trainIndex, ]


# Cogemos las etiquetas y el campo de texto
labels_train <- train$AdoptionSpeed
texts_train <- train$Description

# Obtenemos las descripciones
maxlen <- 100 
training_samples <- 11995  
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
x_train_nuevo <- data[training_indices,]
y_train_nuevo <- labels[training_indices]

dim(x_train_nuevo)


# Obtenemos todo menos las descrpciones, PetID y PhotoAmt
train.normal <- train
dim(train.normal)
train.normal$PetID <- NULL
train.normal$PhotoAmt <- NULL
train.normal$Description <- NULL
train.normal$Name <- NULL
train.normal$RescuerID <- NULL
dim(train.normal)

test.normal <- test
dim(test.normal)
test.normal$PetID <- NULL
test.normal$PhotoAmt <- NULL
test.normal$Description <- NULL
test.normal$Name <- NULL
test.normal$RescuerID <- NULL
dim(test.normal)

library(keras)

main_input <- layer_input(shape = c(100), dtype = 'int32', name = 'main_input')
lstm_out <- main_input %>% 
  layer_embedding(input_dim = 10000, output_dim = 512, input_length = 100) %>% 
  layer_lstm(units = 32) 
  

auxiliary_output <- lstm_out %>% 
  layer_dense(units = 5, activation = 'sigmoid', name = 'aux_output')

auxiliary_input <- layer_input(shape = c(19), name = 'aux_input')
main_output <- layer_concatenate(c(lstm_out, auxiliary_input)) %>%
  layer_dense(units = 64, activation = 'relu') %>% 
  layer_dense(units = 64, activation = 'relu') %>% 
  layer_dense(units = 64, activation = 'relu') %>% 
  layer_dense(units = 5, activation = 'sigmoid', name = 'main_output')

model <- keras_model(
  inputs = c(main_input, auxiliary_input), 
  outputs = c(main_output, auxiliary_output)
)
summary(model)

model %>% compile(
  optimizer = 'adam',
  loss = 'sparse_categorical_crossentropy',
  #loss_weights = c(1.0, 0.2),
  metrics = c("accuracy")
)

model %>% fit(
  x = list(x_train_nuevo, as.matrix(train.normal)),
  y = list(train$AdoptionSpeed, train$AdoptionSpeed),
  epochs = 2,
  batch_size = 64
)

#save_model_weights_hdf5(model, "./ensemble_csv.h5")


labels_test <- test$AdoptionSpeed
texts_test <- test$Description

# Obtenemos las descripciones
maxlen <- 100 
test_samples <- 2998  
max_words <- 10000 

tokenizer <- text_tokenizer(num_words = max_words) %>%
  fit_text_tokenizer(texts_test)
sequences <- texts_to_sequences(tokenizer, texts_test)
word_index = tokenizer$word_index

data <- pad_sequences(sequences, maxlen = maxlen)

indices <- sample(1:nrow(data)) 
test_indices <- indices[1:test_samples]
x_test_nuevo <- data[test_indices,]
y_test_nuevo <- labels[test_indices]
dim(x_test_nuevo)



# Evaluating the model on the test set
# --------------------------------------------------------------------------------------------------

model %>% evaluate(list(x_test_nuevo, as.matrix(test.normal)), list(y_test_nuevo,y_test_nuevo))

