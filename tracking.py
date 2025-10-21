import cv2
import mediapipe as mp
import time
import matplotlib.pyplot as plt


beat_interval = 0.5
last_beat_time = time.time()
flash_on = False
min_x_interval = 100

mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=2,
    min_detection_confidence=0.75,
    min_tracking_confidence=0.75,
    model_complexity=1,
)

mp_draw = mp.solutions.drawing_utils
cap = cv2.VideoCapture(0)

coordinates = []
graph = []
encoding = []

small_jump = 100
medium_jump = 200
large_jump = 500

while len(coordinates) < 16:
    success, img = cap.read()
    if not success:
        break

    img = cv2.flip(img, 1)
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    results = hands.process(img_rgb)
    x=0
    y=0
    if results.multi_hand_landmarks and results.multi_handedness:
        for hand_landmarks, handedness in zip(results.multi_hand_landmarks, results.multi_handedness):
            hand_label = handedness.classification[0].label
            if hand_label == 'Right':
                h, w, c = img.shape
                index_finger_ids = [8]
                for id in index_finger_ids:
                    lm = hand_landmarks.landmark[id]
                    cx, cy = int(lm.x * w), int(lm.y * h)
                    x=cx
                    y=cy
                    cv2.circle(img, (cx, cy), 8, (0, 255, 0), cv2.FILLED)

    current_time = time.time()
    if current_time - last_beat_time >= beat_interval:
        if len(graph) == 0 or abs(x - graph[-1][0]) >= min_x_interval:
            graph.append((current_time, y))
        else:
            graph.append((current_time, graph[-1][1]))
        if(len(graph)>1):
            jump = graph[-1][1] - graph[-2][1]
            print(f"Jump: {jump}")
            if abs(jump) > small_jump and abs(jump) <= medium_jump and abs(jump) <= large_jump:
                if jump > 0:
                    encoding.append(2)
                else:
                    encoding.append(5)
            elif abs(jump) > medium_jump and abs(jump) <= large_jump:
                if jump > 0:
                    encoding.append(3)
                else:
                    encoding.append(6)
            elif abs(jump) > large_jump:
                if jump > 0:
                    encoding.append(4)
                else:
                    encoding.append(7)
            else:
                encoding.append(1)
                
        flash_on = not flash_on
        last_beat_time = current_time
        
        coordinates.append((x, y))
        print(encoding)

    if flash_on:
        color = (255, 255, 255) 
    else:
        color = (0, 0, 0)   

    height, width, _ = img.shape
    center_position = (width - 30, 30)
    cv2.circle(img, center_position, 50, color, cv2.FILLED)

    cv2.imshow("Tracking", img)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break


xs, ys = zip(*graph)
plt.figure(figsize=(6, 6))
plt.plot(xs, ys, '-o', color='tab:blue')
for i, (x, y) in enumerate(graph, start=1):
    plt.text(x, y, str(i), color='red', fontsize=9, ha='right', va='bottom')
plt.title("Finger Positions")
plt.xlabel("x ")
plt.ylabel("y ")
plt.gca().invert_yaxis()  # image origin is top-left
plt.grid(True)
plt.tight_layout()
plt.show()
cap.release()
cv2.destroyAllWindows()
