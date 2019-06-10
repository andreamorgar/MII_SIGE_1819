library(keras)

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



model <- keras_model_sequential() %>%
  layer_conv_2d(filters = 32,  kernel_size = c(3, 3), activation = "relu", input_shape = c(150, 150, 3)) %>%
  layer_max_pooling_2d(pool_size = c(2, 2)) %>%
  layer_conv_2d(filters = 64,  kernel_size = c(3, 3), activation = "relu") %>% layer_max_pooling_2d(pool_size = c(2, 2)) %>%
  layer_max_pooling_2d(pool_size = c(2, 2)) %>%
  #layer_conv_2d(filters = 128, kernel_size = c(3, 3), activation = "relu") %>% layer_max_pooling_2d(pool_size = c(2, 2)) %>%
  #layer_conv_2d(filters = 128, kernel_size = c(3, 3), activation = "relu") %>% layer_max_pooling_2d(pool_size = c(2, 2)) %>%
  layer_dropout(0.3) %>%
  layer_flatten() %>%
  layer_dense(units = 512, activation = "relu") %>%
  layer_dense(units = 5, activation = "softmax")
summary(model)

# Compilar modelo
# https://tensorflow.rstudio.com/keras/reference/compile.html
model %>% compile(
  loss = "categorical_crossentropy",
  optimizer = "adam",
  metrics = c("accuracy")
)

data_augmentation_datagen <- image_data_generator(
  rescale = 1/255,
  rotation_range = 40,
  width_shift_range = 0.2,
  height_shift_range = 0.2,
  shear_range = 0.2,
  zoom_range = 0.2,
  horizontal_flip = TRUE,
  fill_mode = "nearest"
)

train_augmented_data <- flow_images_from_directory(
  directory = train_images_dir,
  generator = train_datagen,
  target_size = c(150, 150),   # (w, h) --> (150, 150)
  batch_size = 100,             # grupos de 20 imágenes
  class_mode = "categorical"        # etiquetas categóricas
)

history <- model %>% 
  fit_generator(
    train_augmented_data,
    steps_per_epoch = 1,
    epochs = 10,
    validation_data = validation_data,
    validation_steps = 1
  )

res <- model %>% evaluate_generator(test_data, steps = 5)
print(res)
