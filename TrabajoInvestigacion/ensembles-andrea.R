# Ensemble con los modelos convolucionales

library(imager)
library(magick)
library(OpenImageR)
library(raster)
library(keras)
library(dplyr)

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

# Cargamos los datos como siempre
train_datagen      <- image_data_generator(rescale = 1/255) 
validation_datagen <- image_data_generator(rescale = 1/255)
test_datagen       <- image_data_generator(rescale = 1/255)

# Conjunto de train
train_data <- flow_images_from_directory(
  directory = train_images_dir,
  generator = train_datagen,
  target_size = c(224, 224),   # (w, h) --> (150, 150)
  batch_size = 100,             # grupos de 20 imÃ¡genes
  class_mode = "categorical"        # etiquetas binarias
)

# Conjunto de test
test_data <- flow_images_from_directory(
  directory =  test_images_dir,
  generator = test_datagen,
  target_size = c(224, 224),   # (w, h) --> (150, 150)
  batch_size = 20,             # grupos de 20 imÃ¡genes
  class_mode = "categorical"        # etiquetas binarias
)

# Conjunto de validaciÃ³n 
validation_data <- flow_images_from_directory(
  directory = val_images_dir,
  generator = validation_datagen,
  target_size = c(224, 224),   # (w, h) --> (150, 150)
  batch_size = 100,             # grupos de 20 imÃ¡genes
  class_mode = "categorical"        # etiquetas binarias
)

# Cargamos los modelos. Vamos a usar 3, que hemos comprobado que nos dan resultados distintos




model1 <- load_model_hdf5("./filtros_sin_imagenet.h5", custom_objects = NULL, compile = TRUE)
res1 <- model1 %>% evaluate_generator(test_data, steps = 5)
print(res1) #0.2819281


model2 <- load_model_hdf5("./filtros_sin_imagenet-modelo2.h5", custom_objects = NULL, compile = TRUE)
res2 <- model2 %>% evaluate_generator(test_data, steps = 5) #0.238957
print(res2)

model3 <- load_model_hdf5("./filtros_sin_imagenet-modelo3.h5", custom_objects = NULL, compile = TRUE)
res3 <- model3 %>% evaluate_generator(test_data, steps = 5)
print(res3) #0.2819281



model4 <- load_model_hdf5("./filtros_con_imagenet-modelo1.h5", custom_objects = NULL, compile = TRUE)
res4 <- model4 %>% evaluate_generator(test_data, steps = 5)
print(res4) #0.2987392


y_modelo1 <- model1 %>% predict_generator(test_data, steps=5)
y_modelo2 <- model2 %>% predict_generator(test_data, steps=5)
y_modelo3 <- model3 %>% predict_generator(test_data, steps=5)
y_modelo4 <- model4 %>% predict_generator(test_data, steps=5)



# final_preds <- 0.59 * y_modelo1 + 0.01* y_modelo2 + 0.40* y_modelo3 #0.2819281
# final_preds <- 0.35 * y_modelo1 + 0.25* y_modelo3 + 0.4 * y_modelo4 0.2761815

final_preds <- 0.4 * y_modelo1 + 0.25* y_modelo3 + 0.35 * y_modelo4 # 0.2779827

final_preds <- 0.35 * y_modelo1 + 0.1* y_modelo3 + 0.45 * y_modelo4

confusion_matrix(5,final_preds)
etiquetas_predichas <- apply(final_preds, 1, which.max)
etiquetas_predichas <- etiquetas_predichas - 1 
# Matriz de confusión
table(etiquetas_predichas,test_data$labels)

acuracy = sum(test_data$labels == etiquetas_predichas) / length(test_data$labels)
acuracy
# 
etq1 <- apply(y_modelo1, 1, which.max)
etq1 <- etq1 - 1 
table(etq1,test_data$labels)

etq2 <- apply(y_modelo2, 1, which.max)
etq2 <- etq2 - 1 
table(etq2,test_data$labels)

etq3 <- apply(y_modelo3, 1, which.max)
etq3 <- etq3 - 1 
table(etq3,test_data$labels)

etq4 <- apply(y_modelo4, 1, which.max)
etq4 <- etq4 - 1 
table(etq4,test_data$labels)

# Matrices de confusión de las individuales 



length(which(etiquetas_predichas==0))
length(which(etiquetas_predichas==1))
length(which(etiquetas_predichas==2))
length(which(etiquetas_predichas==3))
length(which(etiquetas_predichas==4))
# 
# acuracy = sum(test_data$labels == etiquetas_predichas) / length(test_data$labels)
# acuracy
# 
# 


# #-----------------------------------------------------------------------------------------------
# Este código no va, se ha intentado pero da error raro. Va para textos tambien, asi que se explica en el 
# trabajo de teoría y ale
# model1.name = 'model_1_a'
# model2.name = 'model_2_a'
# model3.name = 'model_3_a'
# 
# shared_input <- layer_input(shape=(get_input_shape_at(model1, 1) %>% unlist))
# model_list   <- c(model1(shared_input), model2(shared_input))
# 
# main_output  <- layer_average(model_list) %>% #trainable = TRUE
#   layer_dense(units = 64, activation = 'relu') %>%
#   layer_dense(units = 5, activation = 'softmax')
# 
# model_final <- keras_model(
#   inputs = c(shared_input), 
#   outputs = c(main_output)
# )
# summary(model_final)
# 
# model_final %>% compile(
#   optimizer = "adam", # Mejora un poco con 'adam' que con rmsprop
#   loss = "categorical_crossentropy",
#   metrics = c("accuracy")
# )
# 
# # Entrenamos el modelo
# history <- model_final %>% 
#   fit_generator(
#     train_data,
#     steps_per_epoch = 1,
#     epochs = 1,
#     validation_data = validation_data,
#     validation_steps = 1
#   )
# 
# get_weights(model_final$layers)
# 
# 
# prediciones <- model_final %>% predict(x_test)
# etiquetas_predichas_final <- apply(prediciones, 1, which.max)
# etiquetas_predichas_menos_uno_finl <- etiquetas_predichas_final - 1 # estas son las etiquetas predichas
# acuracy = sum(y_test == etiquetas_predichas_menos_uno_finl) / length(y_test)
# acuracy
# 
# # -------------------------------------------------------------------------------------------


