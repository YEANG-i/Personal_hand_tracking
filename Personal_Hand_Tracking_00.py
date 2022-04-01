import cv2
import time
import mediapipe as mp

cap = cv2.VideoCapture(0)
mpHands = mp.solutions.hands
hands = mpHands.Hands(static_image_mode=False,
                      max_num_hands=1,  # max 5
                      min_detection_confidence=0.8,  # 0-1
                      min_tracking_confidence=0.5)  # 0-1
mpDraw = mp.solutions.drawing_utils
handPointStyle = mpDraw.DrawingSpec(color=(0, 0, 255), thickness=5)  # landmark style
handConnectionStyle = mpDraw.DrawingSpec(color=(0, 255, 0), thickness=10)  # Connection style among landmarks
pTime = 0
cTime = 0
file_number = 1

while True:

    with open("{0}.log".format(file_number), 'a') as fd:

        ret, img = cap.read()
        a = []
        if ret:
            imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # BGR to RGB
            result = hands.process(imgRGB)
            # Processes an RGB image and returns two fields
            # a "multi_hand_landmarks" field that contains the hand landmarks on each detected hand
            imgHeight = img.shape[0]
            imgWidth = img.shape[1]
            if result.multi_hand_landmarks:
                for handLms in result.multi_hand_landmarks:  # Gets every hand_landmarks
                    mpDraw.draw_landmarks(img, handLms, mpHands.HAND_CONNECTIONS, handPointStyle, handConnectionStyle)
                    #  Draws the landmarks and the connections on the image
                    for i, lm in enumerate(handLms.landmark):  # Gets index and location of every hand_landmark
                        xPos = int(lm.x * imgWidth)
                        yPos = int(lm.y * imgHeight)
                        cv2.putText(img, str(i), (xPos - 25, yPos + 5), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (89, 4, 45),
                                    thickness=2)
                        # 在对应的标志点上标序号
                        if i in [4, 8, 12, 16, 20]:
                            cv2.circle(img, (xPos, yPos), 10, (0, 0, 255))  # Draw a circle
                        print(i, xPos, yPos)
                        a.append([i, xPos, yPos])
                        fd.write(str(a))
                        fd.write("\n")

            cTime = time.time()
            fps = 1 / (cTime - pTime)
            pTime = cTime
            cv2.putText(img, 'FPS:{:.0f}'.format(fps), (30, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 5)

            cv2.namedWindow("window_1", cv2.WINDOW_GUI_NORMAL)
            cv2.imshow('window_1', img)

        if cv2.waitKey(1) in [ord('q'), 27]:  # 按键盘上的 q 或 esc 退出（在英文输入法下）
            break

cap.release()
cv2.destroyAllWindows()
