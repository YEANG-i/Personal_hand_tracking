# version 1.0.1

import math
import cv2
import time
import mediapipe as mp


class HandDetector:
    def __init__(self, static_image_mode=False, max_num_hands=1, min_detection_confidence=0.5,
                 min_tracking_confidence=0.5):

        self.static_image_mode = static_image_mode
        self.max_num_hands = max_num_hands
        self.min_detection_confidence = min_detection_confidence
        self.min_tracking_confidence = min_tracking_confidence

        self.mpHands = mp.solutions.hands
        self.hands = self.mpHands.Hands(self.static_image_mode,
                                        self.max_num_hands,  # max 5
                                        self.min_detection_confidence,  # 0-1
                                        self.min_tracking_confidence)  # 0-1
        self.mpDraw = mp.solutions.drawing_utils

        self.results = None
        self.land_mark_list = None
        self.tipIds = [4, 8, 12, 16, 20]
        self.handPointStyle = self.mpDraw.DrawingSpec(color=(0, 0, 255), thickness=5)
        self.handConnectionStyle = self.mpDraw.DrawingSpec(color=(0, 255, 0), thickness=10)

    def find_hands(self, img_input, draw=True):
        imgRGB = cv2.cvtColor(img_input, cv2.COLOR_BGR2RGB)  # BGR to RGB
        self.results = self.hands.process(imgRGB)
        # Processes an RGB image and returns two fields
        # a "multi_hand_landmarks" field that contains the hand landmarks on each detected hand
        if self.results.multi_hand_landmarks:
            for handLms in self.results.multi_hand_landmarks:  # Gets every hand_landmarks
                if draw:
                    self.mpDraw.draw_landmarks(img_input, handLms, self.mpHands.HAND_CONNECTIONS,
                                               self.handPointStyle, self.handConnectionStyle)
                    #  Draws the landmarks and the connections on the image

        return img_input

    def find_position(self, img_input, draw=True):
        self.land_mark_list = []
        imgHeight = img_input.shape[0]
        imgWidth = img_input.shape[1]
        if self.results.multi_hand_landmarks:
            for handLms in self.results.multi_hand_landmarks:
                for i, lm in enumerate(handLms.landmark):  # Gets index and location of every hand_landmark
                    xPos = int(lm.x * imgWidth)
                    yPos = int(lm.y * imgHeight)
                    self.land_mark_list.append([i, xPos, yPos])
                    if draw:
                        if i in [4, 8, 12, 16, 20]:  # ?????????????????????????????????
                            cv2.circle(img_input, (xPos, yPos), 10, (0, 0, 255))  # Draw a circle
                            print(i, xPos, yPos)
        else:
            print("No Hands!!!!!!!!Give me your Hands!!!!!")
        return self.land_mark_list

    def fingers_up(self):
        fingers = []
        # ?????????
        if self.land_mark_list[self.tipIds[0]][1] > self.land_mark_list[self.tipIds[0] - 1][1]:
            fingers.append(1)
        else:
            fingers.append(0)

        # ????????????
        for i in range(1, 5):
            if self.land_mark_list[self.tipIds[i]][2] < self.land_mark_list[self.tipIds[i] - 2][2]:
                fingers.append(1)
            else:
                fingers.append(0)
        return fingers

    def find_distance(self, p1, p2, img, draw=True, r=8, t=2):
        x1, y1 = self.land_mark_list[p1][1:]
        x2, y2 = self.land_mark_list[p2][1:]
        cx, cy = (x1 + x2) // 2, (y1 + y2) // 2

        if draw:
            cv2.line(img, (x1, y1), (x2, y2), (0, 0, 205), t)  # ??????
            cv2.circle(img, (cx, cy), r, (255, 140, 0), cv2.FILLED)  # ??????
        cv2.circle(img, (x1, y1), r, (255, 0, 255), cv2.FILLED)  # ???????????????????????????????????????????????????????????????
        cv2.circle(img, (x2, y2), r, (255, 0, 255), cv2.FILLED)
        length = math.hypot(x2 - x1, y2 - y1)  # ????????????
        return length


def main():
    cap = cv2.VideoCapture(0)
    hand_detector = HandDetector()
    pTime = 0

    while True:
        ret, img = cap.read()
        img = hand_detector.find_hands(img)
        #
        #
        #
        #
        #
        #

        cTime = time.time()
        fps = 1 / (cTime - pTime)
        pTime = cTime
        cv2.putText(img, 'FPS:{:.0f}'.format(fps), (30, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 5)
        cv2.namedWindow("window_1", cv2.WINDOW_GUI_NORMAL)
        cv2.imshow('window_1', img)
        if cv2.waitKey(1) in [ord('q'), 27]:  # ??????????????? q ??? esc ?????????????????????????????????
            break
    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
