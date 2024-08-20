import cv2
import numpy as np

# Funcion para ordenar los puntos de la imagen como cordenadas
def ordenarPuntos(puntos):
    # Concatenamos matrices con np.concatenate y las hacemos una lista con .tolist
    n_puntos= np.concatenate([puntos[0], puntos[1], puntos[2], puntos[3]]).tolist()
    # Orrdenamos en sentido y todos los puntos
    y_order = sorted(n_puntos, key=lambda n_puntos:n_puntos[1])
    # Ordenamos el x pero solo los 2 primero puntos
    x1_order = y_order[:2]
    x1_order = sorted(x1_order, key=lambda x1_order:x1_order[0])
    # Ordenamos el x pero los otros 2 puntos que quedaron
    x2_order = y_order[2:4]
    x2_order = sorted(x2_order, key=lambda x2_order:x2_order[0])
    # Retornamos nuestros puntos ordenados
    return [x1_order[0], x1_order[1], x2_order[0], x2_order[1]]

# Alinear los putos en la imagen  
def alineamento(imagen,altura,anchura):
    imagen_alineada = None
    # Cambiar la imagen a grises
    grises = cv2.cvtColor(imagen, cv2.COLOR_BGR2GRAY)
    # cambiamos nuestra imagen de grises a un umbral para hacer una imagen binaria
    timpoUmbral, umbral = cv2.threshold(grises, 150, 255, cv2.THRESH_BINARY)
    cv2.imshow("Imagen", umbral)
    # Buscamos los contornos de la imagen
    contorno = cv2.findContours(umbral, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[0]
    # Ordenamos los contornos 
    contorno=sorted(contorno,key=cv2.contourArea,reverse=True)[:1]
    for c in contorno:
        # Creamos una variable epsilon con el largo 
        epsilon = 0.01*cv2.arcLength(c, True)
        aproximacion = cv2.approxPolyDP(c, epsilon, True)
        if len(aproximacion) == 4:
            puntos = ordenarPuntos(aproximacion)
            puntos1 = np.float32(puntos)
            puntos2 = np.float32([[0,0], [anchura,0],[0,altura],[anchura,altura]])
            M = cv2.getPerspectiveTransform(puntos1, puntos2)
            imagen_alineada = cv2.warpPerspective(imagen, M, (anchura,altura))
    return imagen_alineada

# Definimos la camara
capturavideo = cv2.VideoCapture(1)

# Hacemos un ciclo para poner los contornos
while True:
    tipoCamara, camara = capturavideo.read()
    # Si la camara no detecta nada se termina el bucle
    if tipoCamara==False:
        break
    # Elejimos una iamgen a6 para identificar las medidad del espacion en este caso de 480X677 pixeles 
    imagen_A6= alineamento(camara, 480, 677)
    # Si imagen A6 no es vacia osea que detecta algo sigue con el proceso de la imagen
    if imagen_A6 is not None:
        puntos = []
        # Pasamos la imagen a escala de grises y por el umbral que emos usado anteriormente 
        imagen_gris = cv2.cvtColor(imagen_A6,cv2.COLOR_BGR2GRAY)
        blur = cv2.GaussianBlur(imagen_gris, (5,5), 1)
        _,umbral2 = cv2.threshold(blur, 0, 255, cv2.THRESH_OTSU+cv2.THRESH_BINARY_INV)
        cv2.imshow("Umbral", umbral2)
        contorno2= cv2.findContours(umbral2, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[0]
        cv2.drawContours(imagen_A6, contorno2, -1, (255,0,0), 2)
        # Aqui estaremos poniendo las etiquetas de cada producto identificado en las images gracias a los momentos 
        suma1 = 0.0
        suma2 = 0.0
        suma3 = 0.0
        suma4 = 0.0
        for c_2 in contorno2:
            area = cv2.contourArea(c_2)
            Momentos = cv2.moments(c_2)
            if(Momentos["m00"]==0):
                Momentos["m00"]=1.0
            x=int(Momentos["m10"]/Momentos["m00"])
            y=int(Momentos["m01"]/Momentos["m00"])
            
            if area > 7000 and area < 7800:
                font = cv2.FONT_HERSHEY_SIMPLEX
                cv2.putText(imagen_A6, "P/ 1.00", (x,y), font, 0.75, (0,255,0), 2)
                suma1 = suma1+1
            
            if area > 8500 and area < 9300:
                font = cv2.FONT_HERSHEY_SIMPLEX
                cv2.putText(imagen_A6, "P/ 2.00", (x,y), font, 0.75, (0,255,0), 2)
                suma2 = suma2+2

            if area > 10700 and area < 11300:
                font = cv2.FONT_HERSHEY_SIMPLEX
                cv2.putText(imagen_A6, "P/ 5.00", (x,y), font, 0.75, (0,255,0), 2)
                suma3 = suma3+5

            if area > 12900 and area < 13700:
                font = cv2.FONT_HERSHEY_SIMPLEX
                cv2.putText(imagen_A6, "P/ 1.00", (x,y), font, 0.75, (0,255,0), 2)
                suma1 = suma1+1
        sumatoria = suma1+suma2+suma3+suma4
        print("Sumatoria de monedas", sumatoria)
        cv2.imshow("imagen_A6", imagen_A6)
        cv2.imshow("Camara", camara)
        cv2.imshow("Umbral", umbral2)
    if cv2.waitKey(1) == ord("q"):
        break
capturavideo.release()
cv2.destroyAllWindows()