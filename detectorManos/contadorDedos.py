import mediapipe as mp
import cv2
import numpy as np
from math import acos, degrees

def centro_palma(lista_cordenadas):
    cordenadas = np.array(lista_cordenadas)
    centro = np.mean(cordenadas, axis=0)
    centro = int(centro[0]), int(centro[1])
    return centro

mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_hands = mp.solutions.hands

cap = cv2.VideoCapture(0,cv2.CAP_DSHOW)

#----pulgar----
puntos_pulgar = [1,2,4]

#indice, medio, anular, meñique----
puntos_palma = [0, 1, 2, 5, 9, 13, 17]
puntos_puntas_dedos = [8, 12, 16, 20]
puntos_base_dedos = [6, 10, 14, 18]

with mp_hands.Hands (
    model_complexity = 1,
    max_num_hands = 1,
    min_detection_confidence = 0.5,
    min_tracking_confidence = 0.5) as hands:
    while True:
        ret, frame = cap.read()
        if ret == False:
            break
        frame = cv2.flip(frame,1)
        height, width, _ = frame.shape
        frame_RGB = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = hands.process(frame_RGB)
        contator_dedos = "_"
        if results.multi_hand_landmarks:
           cordenadas_pulgar = []
           cordenadas_palma = []
           cordenadas_baseD = []
           cordenadas_puntaD = []
           for handLms in results.multi_hand_landmarks:
                for i in puntos_pulgar:
                    x = int(handLms.landmark[i].x * width)
                    y = int(handLms.landmark[i].y * height)
                    cordenadas_pulgar.append([x,y])
                for i in puntos_palma:
                    x = int(handLms.landmark[i].x * width)
                    y = int(handLms.landmark[i].y * height)
                    cordenadas_palma.append([x,y])
                for i in puntos_puntas_dedos:
                    x = int(handLms.landmark[i].x * width)
                    y = int(handLms.landmark[i].y * height)
                    cordenadas_puntaD.append([x,y])
                for i in puntos_base_dedos:
                    x = int(handLms.landmark[i].x * width)
                    y = int(handLms.landmark[i].y * height)
                    cordenadas_baseD.append([x,y])
                
                #--------------------------------------------------------
                #Pulgar
                p1 = np.array(cordenadas_pulgar[0])
                p2 = np.array(cordenadas_pulgar[1])
                p3 = np.array(cordenadas_pulgar[2])

                l1 = np.linalg.norm( p2 - p3)
                l2 = np.linalg.norm( p1 - p3)
                l3 = np.linalg.norm( p1 - p2)

                #Calcular angulo
                angulo = degrees(acos((l1**2 + l3**2 - l2**2) / (2 * l1 * l3)))
                dedo_pulgar = np.array(False)
                texto = "dedo pulgar: no extendido"
                if angulo > 150:
                    dedo_pulgar = np.array(True)
                    texto = "dedo pulgar: extendido"
                
                #Texto que te muestra si el dedo pulgar esta extendido
                # cv2.putText(frame, str(texto), (50,50), cv2.FONT_HERSHEY_COMPLEX,  1, (42, 255, 0), 2, cv2.LINE_AA)

                #--------------------------------------------------------
                #indice, medio, anular, meñique----
                nx, ny = centro_palma(cordenadas_palma)
                cv2.circle(frame,(nx,ny), 3,(0,255,0),2)
                cordenadas_centroides = np.array([nx,ny])
                cordenadas_puntaD = np.array(cordenadas_puntaD)
                cordenadas_baseD = np.array(cordenadas_baseD)

                #------------Distancias------------
                #np.linalg.norm() <-- esta funcion te da la distancia entre dos puntos
                #el axis=1 sirve para que me de los 4 valores (uno por cada dedo) y no solo un valor promedio
                d_centroide_puntaD = np.linalg.norm(cordenadas_centroides - cordenadas_puntaD, axis=1)
                d_centroide_baseD = np.linalg.norm(cordenadas_centroides - cordenadas_baseD, axis=1)
                dif = d_centroide_puntaD - d_centroide_baseD

                dedos = dif > 0
                dedos = np.append(dedo_pulgar,dedos)
                print(dedos)
                contator_dedos = str(np.count_nonzero(dedos == True))

                mp_drawing.draw_landmarks(
                 frame,
                 handLms,
                 mp_hands.HAND_CONNECTIONS,
                 mp_drawing_styles.get_default_hand_landmarks_style(),
                 mp_drawing_styles.get_default_hand_connections_style())
        
        #--------------------------------------------------------
        #Visualisacion
        cv2.rectangle(frame,(0,0),(80,80),(0,255,0),-1)
        cv2.putText(frame, contator_dedos, (25,60), cv2.FONT_HERSHEY_COMPLEX,  2, (255, 255, 255), 2, cv2.LINE_AA)
        cv2.imshow("Frame",frame)
        if cv2.waitKey(1) & 0xFF == 27:
            break
cap.release()
cv2.destroyAllWindows()