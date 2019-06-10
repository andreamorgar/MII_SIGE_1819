## filtros_sin_imagenet.R ##

library(imager)
library(magick)
library(OpenImageR)
library(raster)
library(keras)
library(dplyr)

# ---------------------------------------------------------------------------------------------------------

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
# separaTrainTest("./petfinder-adoption-prediction-gcloud/train_images", "./petfinder-adoption-prediction-gcloud/my_test")
# separaTrainTest("./petfinder-adoption-prediction-gcloud/train_images", "./petfinder-adoption-prediction-gcloud/validation_images", porcentaje=0.15)

# ---------------------------------------------------------------------------------------------------------

## CONSTANTES
dataset_dir      <- 'datos/nuevos/'
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
  target_size = c(224, 224),   # (w, h) --> (150, 150)
  batch_size = 100,             # grupos de 20 imágenes
  class_mode = "categorical"        # etiquetas binarias
)

# Conjunto de test
test_data <- flow_images_from_directory(
  directory =  test_images_dir,
  generator = test_datagen,
  target_size = c(224, 224),   # (w, h) --> (150, 150)
  batch_size = 20,             # grupos de 20 imágenes
  class_mode = "categorical"        # etiquetas binarias
)

# Conjunto de validación 
validation_data <- flow_images_from_directory(
  directory = val_images_dir,
  generator = validation_datagen,
  target_size = c(224, 224),   # (w, h) --> (150, 150)
  batch_size = 100,             # grupos de 20 imágenes
  class_mode = "categorical"        # etiquetas binarias
)

# Configuramos el modelo
model <- keras_model_sequential() %>%
  layer_conv_2d(filters = 32,  kernel_size = c(5, 5), 
                activation = "relu", input_shape = c(224, 224, 3)) %>%
  layer_max_pooling_2d(pool_size = c(2, 2)) %>%
  layer_conv_2d(filters = 64,  kernel_size = c(3, 3), activation = "relu") %>% 
  layer_max_pooling_2d(pool_size = c(2, 2)) %>%
  layer_flatten() %>%
  layer_dense(units = 512, activation = "relu") %>%
  layer_dense(units = 256, activation = "relu") %>%
  layer_dense(units = 5, activation = "softmax")

summary(model)

model %>% compile(
  loss = "categorical_crossentropy",
  optimizer = "adam",#optimizer_rmsprop(lr = 1e-4),
  metrics = c("categorical_accuracy")
)

# Entrenamos el modelo
history <- model %>% 
  fit_generator(
    train_data,
    steps_per_epoch = 1,
    epochs = 1,
    validation_data = validation_data,
    validation_steps = 1
  )

model %>% save_model_hdf5("datos/nuevos/filtros_sin_imagenet.h5")

# Evaluar modelo
# https://tensorflow.rstudio.com/keras/reference/evaluate_generator.html
# model %>% evaluate_generator(test_data, steps = 25)
res <- model %>% evaluate_generator(test_data, steps = 5)
print(res)


summary(model)



model <- load_model_hdf5("datos/nuevos/filtros_sin_imagenet.h5", custom_objects = NULL, compile = TRUE)

# Start witih image of size 224 224
img <- image_load("datos/nuevos/train_images/1/7bce76c66-3.jpg", target_size = c(224, 224)) %>%
  image_to_array() %>%
  array_reshape(dim = c(1, 224, 224, 3))

plot(image_read("datos/nuevos/train_images/1/7bce76c66-3.jpg"))

preds <- model %>% predict(img)
preds
#imagenet_decode_predictions(preds, top = 3)[[1]]

# This is the image entry in the prediction vector
image_output <- model$output[, which.max(preds[1,])]
image_output
summary(model)
last_conv_layer <- model %>% get_layer("conv2d_5") # ------------------ REPASAR ESTO Y PONER NOMBRE CORRECTO


# This is the gradient of the "african elephant" class with regard to
# the output feature map of `block5_conv3`
grads <- k_gradients(image_output, last_conv_layer$output)[[1]]
print(grads)
grads
# This is a vector of shape (512,), where each entry
# is the mean intensity of the gradient over a specific feature map channel
pooled_grads <- k_mean(grads, axis = c(1, 2, 3))
print(pooled_grads)

# This function allows us to access the values of the quantities we just defined:
# `pooled_grads` and the output feature map of `block5_conv3`,
# given a sample image
iterate <- k_function(list(model$input),
                      list(pooled_grads, last_conv_layer$output[1,,,])) 

iterate
# These are the values of these two quantities, as arrays,
# given our sample image of two elephants
c(pooled_grads_value, conv_layer_output_value) %<-% iterate(list(img))


# We multiply  each channel in the feature map array
# by "how important this channel is" with regard to the elephant class
for (i in 1:64) {
  conv_layer_output_value[,,i] <- 
    conv_layer_output_value[,,i] * pooled_grads_value[[i]] 
}

# The channel-wise mean of the resulting feature map
# is our heatmap of class activation
heatmap <- apply(conv_layer_output_value, c(1,2), mean)
heatmap

heatmap <- pmax(heatmap, 0) 
heatmap
heatmap <- heatmap / max(heatmap)
write_heatmap <- function(heatmap, filename, width = 224, height = 224,
                          bg = "white", col = terrain.colors(12)) {
  png(filename, width = width, height = height, bg = bg)
  op = par(mar = c(0,0,0,0))
  on.exit({par(op); dev.off()}, add = TRUE)
  rotate <- function(x) t(apply(x, 2, rev))
  image(rotate(heatmap), axes = FALSE, asp = 1, col = col)
}

write_heatmap(heatmap, "datos/nuevos/image_heatmap.png")


# Continuaci?n de este script por Andrea (empieza aqui)

library(viridis)
library(dplyr)
library(magick)
library(keras)
# Read the original image and it's geometry
image <- image_read("datos/nuevos/train_images/1/7bce76c66-3.jpg")
info <- image_info(image) 
geometry <- sprintf("%dx%d!", info$width, info$height) 

# Create a blended / transparent version of the heatmap image
pal <- col2rgb(viridis(20), alpha = TRUE) 
alpha <- floor(seq(0, 255, length = ncol(pal))) 
pal_col <- rgb(t(pal), alpha = alpha, maxColorValue = 255)
write_heatmap(heatmap, "datos/nuevos/image_overlay.png", 
              width = 14, height = 14, bg = NA, col = pal_col) 

# Overlay the heatmap
image_read("datos/nuevos/image_overlay.png") %>% 
  image_resize(geometry, filter = "quadratic") %>% 
  image_composite(image, operator = "blend", compose_args = "20") %>%
  plot()




