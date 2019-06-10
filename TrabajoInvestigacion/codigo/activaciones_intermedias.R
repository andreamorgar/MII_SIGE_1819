## ----setup, include=FALSE------------------------------------------------
knitr::opts_chunk$set(echo = TRUE)


## ------------------------------------------------------------------------
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



## ------------------------------------------------------------------------
# Configuramos el modelo
model <- keras_model_sequential() %>%
  layer_conv_2d(filters=64,  kernel_size = c(5, 5), 
                activation = "relu", input_shape = c(150, 150, 3)) %>%
  layer_max_pooling_2d(pool_size = c(2, 2)) %>%
  layer_conv_2d(filters = 32,  kernel_size = c(3, 5), 
                activation = "relu") %>%
  layer_max_pooling_2d(pool_size = c(2, 2)) %>%
  layer_dropout(0.3) %>%
  #layer_conv_2d(filters = 64,  kernel_size = c(3, 3), 
  #              activation = "relu") %>% layer_max_pooling_2d(pool_size = c(2, 2)) %>%
  layer_flatten() %>%
  layer_dense(units = 256, activation = "relu") %>%
  layer_dropout(0.5) %>%
  #layer_dense(units = 100, activation = "relu") %>%
  layer_dense(units = 5, activation = "softmax")
summary(model)


model %>% compile(
  loss = "binary_crossentropy",
  optimizer = optimizer_rmsprop(lr = 1e-4),
  metrics = c("accuracy")
)



## ------------------------------------------------------------------------
model <- keras_model_sequential() %>%
  layer_conv_2d(filters = 32,  kernel_size = c(3, 3), activation = "relu", input_shape = c(150, 150, 3)) %>%
  layer_max_pooling_2d(pool_size = c(2, 2)) %>%
  layer_conv_2d(filters = 64,  kernel_size = c(3, 3), activation = "relu") %>% layer_max_pooling_2d(pool_size = c(2, 2)) %>%
  #layer_conv_2d(filters = 128, kernel_size = c(3, 3), activation = "relu") %>% layer_max_pooling_2d(pool_size = c(2, 2)) %>%
  #layer_conv_2d(filters = 128, kernel_size = c(3, 3), activation = "relu") %>% layer_max_pooling_2d(pool_size = c(2, 2)) %>%
  layer_flatten() %>%
  layer_dense(units = 512, activation = "relu") %>%
  layer_dense(units = 5, activation = "sigmoid")

# Compilar modelo
# https://tensorflow.rstudio.com/keras/reference/compile.html
model %>% compile(
  loss = "categorical_crossentropy",
  optimizer = optimizer_rmsprop(lr = 1e-4),
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
  target_size = c(224, 224),   # (w, h) --> (150, 150)
  batch_size = 100,             # grupos de 20 imágenes
  class_mode = "categorical"        # etiquetas binarias
)

history <- model %>% 
  fit_generator(
    train_data,
    steps_per_epoch = 1,
    epochs = 1,
    validation_data = validation_data,
    validation_steps = 1
)

res <- model %>% evaluate_generator(test_data, steps = 5)
print(res)


## ------------------------------------------------------------------------
# Cogemos una imagen que nos sirva de ejemplo
img_path <- "./petfinder-adoption-data/train_images/0/0fd68ca16-1.jpg"
# We preprocess the image into a 4D tensor
img <- image_load(img_path, target_size = c(150, 150))
img_tensor <- image_to_array(img)
img_tensor <- array_reshape(img_tensor, c(1, 150, 150, 3))
# Remember that the model was trained on inputs
# that were preprocessed in the following way:
img_tensor <- img_tensor / 255
dim(img_tensor)
plot(as.raster(img_tensor[1,,,]))


## ------------------------------------------------------------------------
summary(model)


## ------------------------------------------------------------------------
# Extracts the outputs of the top 8 layers:
# Cuidado con lo que ponemos aquí, porque uede que no lleguemos a tener 8 capas, que era el valor que estaba antes,  y que entonces nos de fallo de NULL
layer_outputs <- lapply(model$layers[1:4], function(layer) layer$output)
# Creates a model that will return these outputs, given the model input:
activation_model <- keras_model(inputs = model$input, outputs = layer_outputs)
print(activation_model)


## ------------------------------------------------------------------------
# Returns a list of five arrays: one array per layer activation
activations <- activation_model %>% predict(img_tensor)


## ------------------------------------------------------------------------
first_layer_activation <- activations[[1]]
dim(first_layer_activation)


## ------------------------------------------------------------------------
plot_channel <- function(channel) {
  rotate <- function(x) t(apply(x, 2, rev))
  image(rotate(channel), axes = FALSE, asp = 1, 
        col = terrain.colors(12))
}


## ------------------------------------------------------------------------
plot_channel(first_layer_activation[1,,,5])


plot_channel(first_layer_activation[1,,,7])

# 31 es el numero de filtros que se aplican a la capa
plot_channel(first_layer_activation[1,,,31])



## ------------------------------------------------------------------------
# dir.create("cat_activations")
image_size <- 58
images_per_row <- 16
# NOTA: EN EL FOR VA UN 4, PQ ANTES CREAMOS 4 ACTIVACIONES!!!! pERO VA TODO EN CONJUNTO .
for (i in 1:4) {
  
  layer_activation <- activations[[i]]
  layer_name <- model$layers[[i]]$name
 
  n_features <- dim(layer_activation)[[4]]
  n_cols <- n_features %/% images_per_row
 
  png(paste0("cat_activations/", i, "_", layer_name, ".png"), 
      width = image_size * images_per_row, 
      height = image_size * n_cols)
  op <- par(mfrow = c(n_cols, images_per_row), mai = rep_len(0.02, 4))
  
  for (col in 0:(n_cols-1)) {
    for (row in 0:(images_per_row-1)) {
      channel_image <- layer_activation[1,,,(col*images_per_row) + row + 1]
      plot_channel(channel_image)
    }
  }
  
  par(op)
  dev.off()
}


## ------------------------------------------------------------------------
library(keras)
model <- application_vgg16(
  weights = "imagenet", 
  include_top = FALSE
)
layer_name <- "block3_conv1"
filter_index <- 1
layer_output <- get_layer(model, layer_name)$output
loss <- k_mean(layer_output[,,,filter_index])

## ------------------------------------------------------------------------
# The call to `gradients` returns a list of tensors (of size 1 in this case)
# hence we only keep the first element -- which is a tensor.
grads <- k_gradients(loss, model$input)[[1]] 


## ------------------------------------------------------------------------
# We add 1e-5 before dividing so as to avoid accidentally dividing by 0.
grads <- grads / (k_sqrt(k_mean(k_square(grads))) + 1e-5)


## ------------------------------------------------------------------------
iterate <- k_function(list(model$input), list(loss, grads))
# Let's test it
c(loss_value, grads_value) %<-%
    iterate(list(array(0, dim = c(1, 150, 150, 3))))


## ------------------------------------------------------------------------
# We start from a gray image with some noise
input_img_data <-
  array(runif(150 * 150 * 3), dim = c(1, 150, 150, 3)) * 20 + 128 
step <- 1  # this is the magnitude of each gradient update

for (i in 1:40) { 
  # Compute the loss value and gradient value
  c(loss_value, grads_value) %<-% iterate(list(input_img_data))
  # Here we adjust the input image in the direction that maximizes the loss
  input_img_data <- input_img_data + (grads_value * step)
}


## ------------------------------------------------------------------------
deprocess_image <- function(x) {
 
  dms <- dim(x)
  
  # normalize tensor: center on 0., ensure std is 0.1
  x <- x - mean(x) 
  x <- x / (sd(x) + 1e-5)
  x <- x * 0.1 
  
  # clip to [0, 1]
  x <- x + 0.5 
  x <- pmax(0, pmin(x, 1))
  
  # Reshape to original image dimensions
  array(x, dim = dms)
}


## ------------------------------------------------------------------------
generate_pattern <- function(layer_name, filter_index, size = 150) {
  
  # Build a loss function that maximizes the activation
  # of the nth filter of the layer considered.
  layer_output <- model$get_layer(layer_name)$output
  loss <- k_mean(layer_output[,,,filter_index]) 
  
  # Compute the gradient of the input picture wrt this loss
  grads <- k_gradients(loss, model$input)[[1]]
  
  # Normalization trick: we normalize the gradient
  grads <- grads / (k_sqrt(k_mean(k_square(grads))) + 1e-5)
  
  # This function returns the loss and grads given the input picture
  iterate <- k_function(list(model$input), list(loss, grads))
  
  # We start from a gray image with some noise
  input_img_data <- 
    array(runif(size * size * 3), dim = c(1, size, size, 3)) * 20 + 128
  
  # Run gradient ascent for 40 steps
  step <- 1
  for (i in 1:40) {
    c(loss_value, grads_value) %<-% iterate(list(input_img_data))
    input_img_data <- input_img_data + (grads_value * step) 
  }
  
  img <- input_img_data[1,,,]
  deprocess_image(img) 
}


## ------------------------------------------------------------------------
library(grid)
plot(grid.raster(generate_pattern("block3_conv1", 1)))

