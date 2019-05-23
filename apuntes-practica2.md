# Dudas práctica 2

1. ¿Lo entrega sólo una persona de la pareja?


# Anotaciones

### 9 de Mayo
- Parece que la librería `magick` es muy utilizada para estas cosas.
- Ver `resize` para la imagen (https://www.rdocumentation.org/packages/imager/versions/0.41.2/topics/resize).

## 12 de Mayo
- Supongo que tendremos que hacer como en VC, normalizar la imagen entre 0-1, lo de blur..., pero no sé que muchos más en preprocesamiento podemos hacer.


# Referencias interesantes

- https://cran.r-project.org/web/packages/magick/vignettes/intro.html
- https://cran.r-project.org/web/packages/imager/vignettes/gettingstarted.html
- https://dahtah.github.io/imager/imager.html
- https://towardsdatascience.com/advanced-image-processing-in-r-210618ab128a
- https://www.r-bloggers.com/openimager-an-image-processing-toolkit/
- https://heartbeat.fritz.ai/image-manipulation-for-machine-learning-in-r-ff2b92069fef

### Data augmentation
- https://www.r-bloggers.com/what-you-need-to-know-about-data-augmentation-for-machine-learning/
- https://medium.com/nanonets/how-to-use-deep-learning-when-you-have-limited-data-part-2-data-augmentation-c26971dc8ced
- https://www.doctormetrics.com/2018/10/22/datos-aumentados/#.XNgpvdMzby8

# Código

~~~r
library(imager)
library(magick)

# ------------------------------------------------------------------------------------------

# Para leer todas las imágenes
lista.imagenes.train.0 <- list.files(path = "datos/train_images/0/", pattern = "*.jpg",
                                     full.names = TRUE, recursive = TRUE)
length(lista.imagenes.train.0)

for (i in 1:3){
  image.i <- load.image(lista.imagenes.train.0[i])
  plot(image.i)
}

# ------------------------------------------------------------------------------------------

par(mfrow=c(2,2))

# Cargar una imagen
image <- load.image("datos/train_images/0/1a8fd6707-3.jpg")
plot(image)

# Blurry image
image.blurry <- isoblur(image, 10)
plot(image.blurry)

# Edge detector along x-axis
image.xedges <- deriche(image, 2, order=2, axis="x")
plot(image.xedges)

# Edge detector along y-axis
image.yedges <- deriche(image, 2, order=2, axis="y")
plot(image.yedges)

gato <- image_read("datos/train_images/0/1a8fd6707-3.jpg")
image_info(gato)
~~~
