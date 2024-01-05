import cv2
import mediapipe as mp
import numpy as np
import time, os

actions = ['on', 'off', 'Blur1', 'Blur2', 'Blur3', 'emoticon_ON', 'emoticon_OFF']
seq_length = 30
secs_for_action = 30

# MediaPipe hands model 미디어 파이브에서 가져올 기능들
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
hands = mp_hands.Hands( #손인식을 위한 객체 생성
    max_num_hands=4,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5)

cap = cv2.VideoCapture(0) #비디어 캡쳐 객체를 생성하고 연결을 확인

created_time = int(time.time())
os.makedirs('dataset', exist_ok=True)

while cap.isOpened(): #연결확인
    for idx, action in enumerate(actions):
        data = []

        ret, img = cap.read() #카메라 데이터 읽기

        img = cv2.flip(img, 1) #셀프 카메라처럼 좌우 반전 0보다 크면 좌우 반전/ 0보다 작으면 상하좌우 반전

        cv2.putText(img, f'Waiting for collecting {action.upper()} action...', org=(10, 30), fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=1, color=(255, 255, 255), thickness=2)
        cv2.imshow('img', img)
        cv2.waitKey(3000)

        start_time = time.time()
#30초 동안 while 문으로 반복
        while time.time() - start_time < secs_for_action:
            ret, img = cap.read() #카메라 데이터 읽기

            img = cv2.flip(img, 1)  #셀프 카메라처럼 좌우 반전 0보다 크면 좌우 반전/ 0보다 작으면 상하좌우 반전
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB) #미디어파이프에서 인식 가능한 색공간으로 변경 / 이미지에서 손을 인식하는 부분 이미지를 전달해 손을 인식하고 그 결과를 
            result = hands.process(img) #result 변수에 저장
            img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

            if result.multi_hand_landmarks is not None: #손이 인식되었는지 확인
                for res in result.multi_hand_landmarks: #반복문을 활용해 인식된 손의 주요 부분을 그림으로 그려 구현하고 result.multi_hand_landmarks에 인식된 손의 주요 정보가 담겨있음 x,y,z,의 정보가 리스트 형태로 담겨있음 주요 부분에 해당하는 번호를 인덱스로 사용하여 데이터 접근 가능
                    joint = np.zeros((21, 4))
                    for j, lm in enumerate(res.landmark):
                        joint[j] = [lm.x, lm.y, lm.z, lm.visibility]
                        
                        #print(hand_landmarks.landmark[0].x)

                    # Compute angles between joints 각 손가락 각도 구하는 코드 
                    v1 = joint[[0,1,2,3,0,5,6,7,0,9,10,11,0,13,14,15,0,17,18,19], :3] # Parent joint
                    v2 = joint[[1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20], :3] # Child joint
                    v = v2 - v1 # [20, 3]
                    # Normalize v
                    v = v / np.linalg.norm(v, axis=1)[:, np.newaxis]

                    # Get angle using arcos of dot product
                    angle = np.arccos(np.einsum('nt,nt->n',
                        v[[0,1,2,4,5,6,8,9,10,12,13,14,16,17,18],:],
                        v[[1,2,3,5,6,7,9,10,11,13,14,15,17,18,19],:])) # [15,]

                    angle = np.degrees(angle) # Convert radian to degree

                    angle_label = np.array([angle], dtype=np.float32)
                    angle_label = np.append(angle_label, idx) #라벨 넣어주기 open 일때는 0 / stop 일때는 1

                    d = np.concatenate([joint.flatten(), angle_label]) #100개짜리 행렬 완성

                    data.append(d) #추출된 데이터를 data라는 변수에 전부 넣어주기 

                    mp_drawing.draw_landmarks(img, res, mp_hands.HAND_CONNECTIONS) #손

            cv2.imshow('img', img) #영상 화면에 구현
            if cv2.waitKey(1) == ord('q'):
                break

        data = np.array(data)
        print(action, data.shape)
        np.save(os.path.join('dataset', f'raw_{action}_{created_time}'), data) #mpy 데이터로 저장

        # Create sequence data
        full_seq_data = []
        for seq in range(len(data) - seq_length):
            full_seq_data.append(data[seq:seq + seq_length])

        full_seq_data = np.array(full_seq_data)
        print(action, full_seq_data.shape)
        np.save(os.path.join('dataset', f'seq_{action}_{created_time}'), full_seq_data)
    break

    