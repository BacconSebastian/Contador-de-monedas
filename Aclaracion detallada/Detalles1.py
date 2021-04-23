# Importe
from cv2 import cv2 

# Confirmacion de version
print(cv2.__version__)

# Importamos en la variable "imagen" la imagen necesaria
imagen = cv2.imread('Imagenes/contorno.jpg')

# Convertimos la imagen en BYN con las funciones del opencv ( cv2.COLOR_BGR2GRAY )
grises = cv2.cvtColor(imagen, cv2.COLOR_BGR2GRAY)

# Umbralizamos la imagen en BYN con opencv ( cv2.threshold ( img, valor minimo RED, valor maximo GREEN, funcion de opencv ) )
# Esta funcion arroja como primer variable: tipo de umbral utilizado
# Como segunda variable: la imagen que necesitamos
# Por eso mismo es que al principio le asignamos "_" como variable para que despliegue el tipo de umbral
_,umbral = cv2.threshold(grises, 100, 255, cv2.THRESH_BINARY)

# Encontrar los contornos
# cv2.findContours( img, como nos retornara los datos, como sera el contorno )
contorno,jerarquia = cv2.findContours(umbral, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)

# Dibujar el contorno
# cv2.drawContours( img en la que se aplican los contornos, img de donde sacar los contornos (umbralizada), el contorno a marcar en ubicacion numerica (-1 si queremos todos), color en rgb, grosor del contorno en px )
cv2.drawContours(imagen, contorno, -1, (236, 82, 82), 2)

# Mostramos la imagen
cv2.imshow('imagen', imagen)
cv2.imshow('grises', grises)
cv2.imshow('umbral', umbral)

# Ponemos el tiempo que necesitamos la imagen. En imagenes se coloca 0, en videos se coloca 1
cv2.waitKey(0)

# Al tocar una tecla se cierran todas las ventanas
cv2.destroyAllWindows()