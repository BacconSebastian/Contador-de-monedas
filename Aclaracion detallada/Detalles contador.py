import cv2 as cv2
import numpy as np

# SIEMPRE TIENEN QUE SER IMPARES
# Si no se elimina bien el ruido jugar con estos valores:
vg = 9
vk = 23

original = cv2.imread('Imagenes/monedassoles.jpg')

# Convertimos en escala de grises la imagen original
gris = cv2.cvtColor(original,cv2.COLOR_BGR2GRAY)

# Desenfocamos la imagen en escala de grises
# cv2.GaussianBlur(img en grises, (valor de gauss,valor de gauss), siempre va 0)
gauss = cv2.GaussianBlur(gris, (vg,vg), 0)

# Pasamos a byn y eliminamos más el ruido
# cv2.Canny(img sin ruido, valor del 0 al 255, valor del 0 al 255)
canny = cv2.Canny(gauss, 60, 100)

# Este valor (kernel) es necesario en el proceso de clausura.
# np.ones( ( valor de kernel, valor de kernel ) , propiedad de numpy )
kernel = np.ones( (vk,vk) , np.uint8 )

# Nos encargamos de eliminar el ruido interno con morphologyEx
# cv2.morphologyEx( img, comando cv2 de clausura, valor kernel adquirido anteriormente )
cierre = cv2.morphologyEx(canny, cv2.MORPH_CLOSE, kernel)

# Agregado personal:
# Nos encargamos de eliminar el ruido externo: apertura
# cv2.morphologyEx (img sin ruido interno, cv2.MORPH_OPEN, kernel)
apertura = cv2.morphologyEx (cierre, cv2.MORPH_OPEN, kernel)

# Encontramos los contornos en la imagen sin ruido interno
contorno, jerarquia = cv2.findContours(apertura.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

# Mostramos la cantidad de contornos encontrados (monedas)
print("Monedas encontradas: {}".format(len(contorno)))

# Dibujamos los contornos para verlos manualmente.
cv2.drawContours(original, contorno, -1, (255,0,0), 2)

cv2.imshow('con ruido', gris)
cv2.imshow('sin ruido', gauss)
cv2.imshow('byn sin ruido', canny)
cv2.imshow('clausurada', cierre)
cv2.imshow('contorno', original)
cv2.waitKey(0)
cv2.destroyAllWindows()