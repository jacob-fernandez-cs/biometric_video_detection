import cv2
import mediapipe as mp

mphands = mp.solutions.hands
mp_face_detection = mp.solutions.face_detection
mp_pose = mp.solutions.pose
hands = mphands.Hands()
mp_drawing = mp.solutions.drawing_utils
cap = cv2.VideoCapture('test.mp4')

_, frame = cap.read()

with mp_pose.Pose(min_detection_confidence=0.5,
    min_tracking_confidence=0.5) as pose:
        with mp_face_detection.FaceDetection(
            model_selection=0, min_detection_confidence=0.5) as face_detection:

            h, w, c = frame.shape



            while True:
                _, frame = cap.read()
                
                framergb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                result = hands.process(framergb)
                hand_landmarks = result.multi_hand_landmarks
                
                

                if hand_landmarks:
                    for handLMs in hand_landmarks:
                        x_max = 0
                        y_max = 0
                        x_min = w
                        y_min = h
                        for lm in handLMs.landmark:
                            x, y = int(lm.x * w), int(lm.y * h)
                            if x > x_max:
                                x_max = x
                            if x < x_min:
                                x_min = x
                            if y > y_max:
                                y_max = y
                            if y < y_min:
                                y_min = y
                        cv2.rectangle(frame, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)
                        mp_drawing.draw_landmarks(frame, handLMs)

                resultsF = face_detection.process(framergb)

                
                framergb.flags.writeable = True
                framergb = cv2.cvtColor(framergb, cv2.COLOR_RGB2BGR)
                if resultsF.detections:
                    for detection in resultsF.detections:
                        print("I made it here!")
                        mp_drawing.draw_detection(frame, detection)


                resultsP = pose.process(framergb)

                #While shrugging the pose should connect to arms to indicate shrugging occured
                mp_drawing.draw_landmarks(frame, resultsP.pose_landmarks, mp_pose.POSE_CONNECTIONS)

                if cv2.waitKey(1) == ord('q'):
                    break
                
                
                        
                cv2.imshow("Frame", frame)

                cv2.waitKey(1)