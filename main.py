import cv2
import numpy as np

cap = cv2.VideoCapture("test2.mp4")

if not cap.isOpened():
    print("Ошибка: не удалось открыть видео")
    exit()

while True:
    ret, frame = cap.read()

    if not ret:
        break

    # В ч/б
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # CLAHE - усиление контраста
    clahe = cv2.createCLAHE(clipLimit=1.5, tileGridSize=(8,8))
    enhanced = clahe.apply(gray)

    # Размытие
    blur = cv2.GaussianBlur(enhanced, (7, 7), 0)

    # Контуры (делаем чувствительнее)
    edges = cv2.Canny(blur, 60, 140)
    kernel = np.ones((2,2), np.uint8)
    edges = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel)

    # 4. Поиск линий (Хаф)
    lines = cv2.HoughLinesP(
        edges,
        rho=1,
        theta=np.pi / 360,
        threshold=120,
        minLineLength=100,
        maxLineGap=12
    )

    # 5. Рисуем линии
    if lines is not None:

        main_lines = []

        for line in lines:
            x1, y1, x2, y2 = line[0]

            # Длина линии
            length = np.sqrt((x2 - x1)**2 + (y2 - y1)**2)

            # Угол линии
            angle = np.degrees(np.arctan2(y2 - y1, x2 - x1))

            # Фильтр: только длинные линии
            if length > 150:

                main_lines.append((x1, y1, x2, y2, angle))

    # Если нашли линии
    if len(main_lines) > 0:

        # Средний угол
        avg_angle = np.mean([l[4] for l in main_lines])

        for x1, y1, x2, y2, angle in main_lines:

            # Оставляем только близкие по углу
            if abs(angle - avg_angle) < 15:

                cv2.line(
                    frame,
                    (x1, y1),
                    (x2, y2),
                    (0, 0, 255),
                    3
                )

    cv2.imshow("Result", frame)
    cv2.imshow("Edges", edges)

    if cv2.waitKey(1) == 27:
        break

cap.release()
cv2.destroyAllWindows()


