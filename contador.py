import cv2 as cv2
import numpy as np

vg = 9
vk = 23

original = cv2.imread('Imagenes/monedassoles.jpg')
gris = cv2.cvtColor(original,cv2.COLOR_BGR2GRAY)
gauss = cv2.GaussianBlur(gris, (vg,vg), 0)
canny = cv2.Canny(gauss, 60, 100)
kernel = np.ones( (vk,vk) , np.uint8 )
cierre = cv2.morphologyEx(canny, cv2.MORPH_CLOSE, kernel)
apertura = cv2.morphologyEx (cierre, cv2.MORPH_OPEN, kernel)
contorno, jerarquia = cv2.findContours(apertura.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)


print("Monedas encontradas: {}".format(len(contorno)))
cv2.drawContours(original, contorno, -1, (255,0,0), 2)

cv2.imshow('contorno', original)
cv2.waitKey(0)
cv2.destroyAllWindows()