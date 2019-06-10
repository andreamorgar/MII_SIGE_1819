# Código actual: 
  
library(imager)
library(magick)
library(OpenImageR)
library(raster)
library(keras)
# ---------------------------------------------------------------------------------------------------------
# OBTENER UN CONJUNTO DE TEST SOBRE EL QUE PODAMOS CONSEGUIR PREDECIR Y DEMÁS

# Quita un trozo de train y lo mete en test
# Importante crear antes la carpeta test_pro y las diferentes carpetas para las etiquetas dentro
separaTrainTest <- function(carpeta_train, carpeta_test, porcentaje = 0.2) {
  clases<-list.dirs(path = carpeta_train, full.names = FALSE)
  
  for (clase in clases){
    if(clase != "") {
      carpeta_clase_train <- paste(carpeta_train,clase,sep = "/")
      carpeta_clase_test <- paste(carpeta_test,clase,sep = "/")
      
      todos <- list.files(path = carpeta_clase_train)
      a_copiar <- sample(todos, length(todos)*porcentaje)
      
      for (fichero in a_copiar){
        file.copy(paste(carpeta_clase_train, fichero, sep = "/"), carpeta_clase_test)
        file.remove(paste(carpeta_clase_train, fichero, sep = "/"))
      }
    }
  }
}

# Solo una vez para generar el conjunto de test
#separaTrainTest("./petfinder-adoption-data/train_images", "./petfinder-adoption-data/my_test")
# separaTrainTest("./petfinder-adoption-data/train_images", "./petfinder-adoption-data/validation_images", porcentaje=0.15)
# ---------------------------------------------------------------------------------------------------------

## CONSTANTES
dataset_dir      <- './petfinder-adoption-data/'
train_images_dir <- paste0(dataset_dir, 'train_images')
train_data_file  <- paste0(dataset_dir, 'train.csv')
#test_images_dir  <- paste0(dataset_dir, 'test_images/')
test_images_dir  <- paste0(dataset_dir, 'my_test/')
test_data_file   <- paste0(dataset_dir, 'test.csv')
val_images_dir  <- paste0(dataset_dir, 'validation_images/')

# ---------------------------------------------------------------------------------------------------------

# Cargamos los datos 

train_datagen      <- image_data_generator(rescale = 1/255) 
validation_datagen <- image_data_generator(rescale = 1/255)
test_datagen       <- image_data_generator(rescale = 1/255)

# Conjunto de train
train_data <- flow_images_from_directory(
  directory = train_images_dir,
  generator = train_datagen,
  target_size = c(150, 150),   # (w, h) --> (150, 150)
  batch_size = 100,             # grupos de 20 imágenes
  class_mode = "categorical"        # etiquetas binarias
)

# Conjunto de test

test_data <- flow_images_from_directory(
  directory =  test_images_dir,
  generator = test_datagen,
  target_size = c(150, 150),   # (w, h) --> (150, 150)
  batch_size = 20,             # grupos de 20 imágenes
  class_mode = "categorical"        # etiquetas binarias
)

# Conjunto de validación 
validation_data <- flow_images_from_directory(
  directory = val_images_dir,
  generator = validation_datagen,
  target_size = c(150, 150),   # (w, h) --> (150, 150)
  batch_size = 100,             # grupos de 20 imágenes
  class_mode = "categorical"        # etiquetas binarias
)

# Modelo 1
# # Configuramos el modelo
# model <- keras_model_sequential() %>%
#   layer_conv_2d(filters = 32,  kernel_size = c(5, 5), activation = "relu", input_shape = c(150, 150, 3)) %>%
#   layer_max_pooling_2d(pool_size = c(2, 2)) %>%
#   layer_conv_2d(filters = 64,  kernel_size = c(3, 3), activation = "relu") %>% layer_max_pooling_2d(pool_size = c(2, 2)) %>%
#   layer_flatten() %>%
#   layer_dense(units = 512, activation = "relu") %>%
#   layer_dense(units = 256, activation = "relu") %>%
#   layer_dense(units = 5, activation = "softmax")
# summary(model)
# model %>% compile(
#   loss = "categorical_crossentropy",
#   optimizer = optimizer_rmsprop(lr = 1e-4),
#   metrics = c("categorical_accuracy")
# )


# Modelo 2
# create the base pre-trained model
base_model <- application_inception_v3(weights = 'imagenet', include_top = FALSE)
# base_model <- application_vgg16(include_top = TRUE, weights = "imagenet",
#                                 input_tensor = NULL, input_shape = NULL, pooling = NULL,
#                                 classes = 1000)

#https://blog.rstudio.com/2017/09/05/keras-for-r/ como dropout
predictions <- base_model$output %>% 
  layer_global_average_pooling_2d() %>% 
  layer_dense(units = 512, activation = 'relu') %>% 
  layer_dropout(rate = 0.4) %>% 
  layer_dense(units = 256, activation = 'relu') %>%
  layer_dense(units = 5, activation = 'softmax')

# this is the model we will train
model <- keras_model(inputs = base_model$input, outputs = predictions)


# first: train only the top layers (which were randomly initialized)
# i.e. freeze all convolutional InceptionV3 layers
# for (layer in base_model$layers)
#   layer$trainable <- FALSE

# freeze_weights(base_model, from = 1, to = length(base_model$layers))
# unfreeze_weights(base_model, from = length(base_model$layers)+1 )

freeze_weights(base_model, from = 1, to = 12)
unfreeze_weights(base_model, from = 13 )

# compile the model (should be done after setting layers to non-trainable)
# model %>% compile(optimizer = 'rmsprop', loss = 'categorical_crossentropy', metrics = c("accuracy"))

model %>% compile(optimizer = optimizer_adam(), loss = 'categorical_crossentropy', metrics = c("accuracy"))

# train the model on the new data for a few epochs
# model %>% fit_generator(...)


# Entrenamos el modelo
history <- model %>% 
  fit_generator(
    train_data,
    steps_per_epoch = 1,
    epochs = 5,
    validation_data = validation_data,
    validation_steps = 1
  )

# Evaluar modelo
# https://tensorflow.rstudio.com/keras/reference/evaluate_generator.html
#model %>% evaluate_generator(test_data, steps = 25)
res <- model %>% evaluate_generator(test_data, steps = 5)
print(res)
#model %>% predict_classes(test_data)


# Guardar modelo (HDF5)
# https://tensorflow.rstudio.com/keras/reference/save_model_hdf5.html
#model %>% save_model_hdf5("dogsVScats.h5")

# # Visualizar entrenamiento
# plot(history)