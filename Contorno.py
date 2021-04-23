from cv2 import cv2 
print(cv2.__version__)

imagen = cv2.imread('Imagenes/contorno.jpg')
grises = cv2.cvtColor(imagen, cv2.COLOR_BGR2GRAY)
_,umbral = cv2.threshold(grises, 100, 255, cv2.THRESH_BINARY)


contorno,jerarquia = cv2.findContours(umbral, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
cv2.drawContours(imagen, contorno, -1, (236, 82, 82), 2)


cv2.imshow('imagen original', imagen)
cv2.waitKey(0)
cv2.destroyAllWindows()

