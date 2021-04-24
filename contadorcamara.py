import cv2 as cv
import numpy as np

def ordenar(puntos):
    num_puntos = np.concatenate([puntos[0],puntos[1],puntos[2],puntos[3]]).tolist()
    y = sorted(num_puntos, key=lambda num_puntos:num_puntos[1])
    x1 = y[0:2]
    x1 = sorted(x1, key=lambda x1:x1[0])
    x2 = y[2:4]
    x2 = sorted(x2, key=lambda x2:x2[0])
    return [x1[0],x1[1],x2[0],x2[1]]

def alinear(imagen, ancho, alto):
    imagen_alineada = None
    grises = cv.cvtColor(imagen, cv.COLOR_BGR2GRAY)
    tipo,umbral = cv.threshold(grises, 150, 255, cv.THRESH_BINARY)
    contorno = cv.findContours(umbral, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)[0]
    contorno = sorted(contorno, key=cv.contourArea, reverse=True)[:1]
    for c in contorno:
        epsilon = 0.01*cv.arcLength(c, True)
        aprox = cv.approxPolyDP(c, epsilon, True)

        if len (aprox) == 4:
            puntos = ordenar(aprox)
            puntos1 = np.float32(puntos)
            puntos2 = np.float32([[0,0],[ancho,0],[0,alto],[ancho,alto]])
            m = cv.getPerspectiveTransform(puntos1, puntos2)
            imagen_alineada = cv.warpPerspective(imagen, m, (ancho,alto))

    return imagen_alineada

capturavideo = cv.VideoCapture(0)

while True:
    tipocamara,camara = capturavideo.read()
    if tipocamara == False:
        break
    imagenA6 = alinear(camara, ancho=480, alto=677)
    if imagenA6 is not None:
        puntos=[]
        grisesA6 = cv.cvtColor(imagenA6, cv.COLOR_BGR2GRAY)
        gauss = cv.GaussianBlur(grisesA6, (5,5), 1)
        _,umbralA6 = cv.threshold(gauss, 0,255, cv.THRESH_OTSU+cv.THRESH_BINARY_INV)
        contornoA6 = cv.findContours(umbralA6, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)[0]
        cv.drawContours(imagenA6, contornoA6, -1, (255,0,0), 2)
        suma1 = 0.0
        suma2 = 0.0
        for c2 in contornoA6:
            area = cv.contourArea(c2)
            momentos = cv.moments(c2)
            
            if (momentos["m00"]==0):
                momentos["m00"]=1.0
            x = int(momentos["m10"]/momentos["m00"])
            y = int(momentos["m01"]/momentos["m00"])

            if area < 9300 and area > 8000:
                font = cv.FONT_HERSHEY_SIMPLEX
                cv.putText(imagenA6, "0.20 cent", (x,y), font, (0.75, 255,0,0), 1)
                suma1 = suma1 + 0.2

            if area < 7800 and area > 6500:
                font = cv.FONT_HERSHEY_SIMPLEX
                cv.putText(imagenA6, "0.10 cent", (x,y), font, (0.75, 255,0,0), 1)
                suma2 = suma2 + 0.1

        total = suma1 + suma2
        
        print("Sumatoria total en centimos: ", round(total,2))
        cv.imshow("Imagen final", imagenA6)
        cv.imshow("Camara", camara)
    
    if cv.waitKey(1) == ord("s"):
        break

capturavideo.release()
cv.destroyAllWindows()
