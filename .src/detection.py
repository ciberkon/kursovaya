import cv2
import time
from ultralytics import YOLO

# ================== НАСТРОЙКИ ==================
MODEL_PATH = "C:/Users/konat/runs/detect/train21/weights/best.pt" #местоположение обученной модели
SOURCE = "murIDE_scene_test.mp4"   # видео из симулятора или 0 для камеры
SAVE_VIDEO = True
OUTPUT_PATH = "output.avi"

CONF_THRESHOLD = 0.3
MIN_AREA = 500  # фильтр мелких объектов

# ================== ЗАГРУЗКА МОДЕЛИ ==================
model = YOLO(MODEL_PATH)

# ================== ВИДЕО ==================
cap = cv2.VideoCapture(SOURCE)

if not cap.isOpened():
    print("Ошибка: не удалось открыть источник видео")
    exit()

width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps_input = cap.get(cv2.CAP_PROP_FPS)

if SAVE_VIDEO:
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    out = cv2.VideoWriter(OUTPUT_PATH, fourcc, fps_input, (width, height))

prev_time = 0

# ================== ОСНОВНОЙ ЦИКЛ ==================
while True:
    ret, frame = cap.read()
    if not ret:
        print("Видео завершено")
        break

    # -------- ПРЕДОБРАБОТКА --------
    frame_blur = cv2.GaussianBlur(frame, (5, 5), 0)

    # -------- ДЕТЕКЦИЯ --------
    results = model(frame_blur, conf=CONF_THRESHOLD)

    annotated_frame = frame.copy()
    num_objects = 0

    # -------- ПОСТОБРАБОТКА --------
    if results[0].boxes is not None:
        for box in results[0].boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            area = (x2 - x1) * (y2 - y1)

            if area > MIN_AREA:
                num_objects += 1

                # рамка
                cv2.rectangle(annotated_frame, (x1, y1), (x2, y2),
                              (0, 255, 0), 2)

                # центр объекта
                cx = (x1 + x2) // 2
                cy = (y1 + y2) // 2
                cv2.circle(annotated_frame, (cx, cy), 5,
                           (0, 0, 255), -1)

    # -------- FPS --------
    current_time = time.time()
    fps = 1 / (current_time - prev_time) if prev_time != 0 else 0
    prev_time = current_time

    # -------- ТЕКСТ --------
    cv2.putText(annotated_frame, f"Objects: {num_objects}", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    cv2.putText(annotated_frame, f"FPS: {fps:.2f}", (10, 70),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    # -------- ВЫВОД --------
    cv2.imshow("Pipe Detection", annotated_frame)

    # -------- СОХРАНЕНИЕ --------
    if SAVE_VIDEO:
        out.write(annotated_frame)

    # -------- ЛОГ --------
    print(f"Objects: {num_objects}, FPS: {fps:.2f}")

    # -------- ВЫХОД --------
    if cv2.waitKey(1) == 27:
        break

# ================== ЗАВЕРШЕНИЕ ==================
cap.release()
if SAVE_VIDEO:
    out.release()
cv2.destroyAllWindows()