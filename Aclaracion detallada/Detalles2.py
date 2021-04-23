import cv2 as cv2
import numpy as np



''' OBEJTIVO: LOGRAR SACAR LOS CONTORNOS REDUCIENDO PRIMERO EL RUIDO DE LA IMAGEN'''



# Valor de gauss
vg = 3
# Valor de Kernel
vk = 3

original = cv2.imread('Imagenes/monedas.jpg')
gris = cv2.cvtColor(original,cv2.COLOR_BGR2GRAY)

# Eliminar el ruido:
# cv2.GaussianBlur( img, se introducen los valores para diferentes desenfoques(valor1,valor2), siempre va 0)
gauss = cv2.GaussianBlur(gris, (vg,vg), 0)

# conseguir el contorno:
# cv2.Canny( img, agregar los dos valores del 0 al 255)
canny = cv2.Canny(gauss, 60, 100)

cv2.imshow('con ruido', gris)
cv2.imshow('menos ruido', gauss)
cv2.imshow('contorno', canny)
cv2.waitKey(0)
cv2.destroyAllWindows()