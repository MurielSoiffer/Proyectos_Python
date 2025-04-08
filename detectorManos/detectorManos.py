import mediapipe as mp
import cv2
import math

dispositivoCaptura = cv2.VideoCapture(0)

mpManos = mp.solutions.hands

manos = mpManos.Hands(static_image_mode = False, max_num_hands = 2, min_detection_confidence = 0.9, min_tracking_confidence = 0.8)

mpDibujar = mp.solutions.drawing_utils

while True :
    success, img = dispositivoCaptura.read()

    imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    resultado = manos.process(imgRGB)

    if resultado.multi_hand_landmarks:
        for handLms in resultado.multi_hand_landmarks:
            # mpDibujar.draw_landmarks(img, handLms, mpManos.HAND_CONNECTIONS)
            for id,lm in enumerate(handLms.landmark):
                alto,ancho,color = img.shape
                cx,cy = int(lm.x * ancho), int(lm.y*alto)
                if id == 4:
                    cv2.circle(img,(cx,cy),10,(255,255,0),cv2.FILLED)
                    x4,y4 = cx,cy
                # if id == 8:
                #     cv2.circle(img,(cx,cy),10,(255,255,0),cv2.FILLED)
                #     x8,y8 = cx,cy
                if id == 20:
                    cv2.circle(img,(cx,cy),10,(255,255,0),cv2.FILLED)
                    x20,y20 = cx,cy
            mediaX = (x4 + x20) // 2
            mediaY = (y4 + y20) // 2

            distanciaDedos = math.hypot(x20-x4, y20-y4)
            print(distanciaDedos)
            cv2.line(img,(x4,y4),(x20,y20),(0,0,255),3)
            # cv2.line(img,(x4,y4),(x8,y8),(0,0,255),3)
            # cv2.line(img,(x8,y8),(x20,y20),(0,0,255),3)
            if distanciaDedos < 16:
                cv2.putText(img, 'Se tocan', (50,50), cv2.FONT_HERSHEY_COMPLEX,  1, (0,0,0), 2, cv2.LINE_AA)

    cv2.imshow("Image", img)
    cv2.waitKey(1)