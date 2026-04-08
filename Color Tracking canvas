import cv2
import numpy as np

cap = cv2.VideoCapture(0)

canvas = None
prev_x, prev_y = 0, 0

erase_mode = False
pause_mode = False   # NEW

while True:
    success, img = cap.read()
    img = cv2.flip(img, 1)

    if canvas is None:
        canvas = np.zeros_like(img)

    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    lower = np.array([100, 150, 50])
    upper = np.array([140, 255, 255])

    mask = cv2.inRange(hsv, lower, upper)

    kernel = np.ones((5, 5), np.uint8)
    mask = cv2.erode(mask, kernel, iterations=1)
    mask = cv2.dilate(mask, kernel, iterations=2)

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if contours:
        cnt = max(contours, key=cv2.contourArea)
        area = cv2.contourArea(cnt)

        if area > 800:
            x, y, w, h = cv2.boundingRect(cnt)

            cx = x + w // 2
            cy = y + h // 2

            cv2.circle(img, (cx, cy), 8, (0, 255, 0), -1)

            if prev_x == 0 and prev_y == 0:
                prev_x, prev_y = cx, cy

            if not pause_mode:  # 👈 Only draw if not paused
                if erase_mode:
                    cv2.line(canvas, (prev_x, prev_y), (cx, cy), (0, 0, 0), 20)
                else:
                    cv2.line(canvas, (prev_x, prev_y), (cx, cy), (255, 0, 255), 5)

            prev_x, prev_y = cx, cy
    else:
        prev_x, prev_y = 0, 0

    # Merge canvas
    imgGray = cv2.cvtColor(canvas, cv2.COLOR_BGR2GRAY)
    _, imgInv = cv2.threshold(imgGray, 50, 255, cv2.THRESH_BINARY_INV)
    imgInv = cv2.cvtColor(imgInv, cv2.COLOR_GRAY2BGR)

    img = cv2.bitwise_and(img, imgInv)
    img = cv2.bitwise_or(img, canvas)

    # Show modes
    mode_text = "ERASER" if erase_mode else "DRAW"
    pause_text = "PAUSED" if pause_mode else ""

    cv2.putText(img, f"Mode: {mode_text}", (10, 40),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    if pause_mode:
        cv2.putText(img, "PAUSED", (10, 80),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

    cv2.imshow("Finger Drawing", img)
    cv2.imshow("Mask", mask)

    key = cv2.waitKey(1) & 0xFF

    if key == ord('q'):
        break
    elif key == ord('c'):
        canvas = np.zeros_like(img)
    elif key == ord('e'):
        erase_mode = True
    elif key == ord('d'):
        erase_mode = False
    elif key == ord('w'):
        pause_mode = not pause_mode   # 🔥 toggle pause

cap.release()
cv2.destroyAllWindows()
