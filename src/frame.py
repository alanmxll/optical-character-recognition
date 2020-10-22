import cv2
import pytesseract

plate = []
cap = cv2.VideoCapture(0)
while cap.isOpened():
    _, frame = cap.read()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    gray = cv2.blur(gray, (3, 3))
    canny = cv2.Canny(gray, 150, 200)
    canny = cv2.dilate(canny, None, iterations=1)

    cnts, _ = cv2.findContours(canny, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)

    for c in cnts:
        area = cv2.contourArea(c)
        x, y, w, h = cv2.boundingRect(c)
        epsilon = 0.09 * cv2.arcLength(c, True)
        approx = cv2.approxPolyDP(c, epsilon, True)

        if len(approx) == 4 and area > 9000:
            print('area=', area)
            aspect_ratio = float(w) / h

            if aspect_ratio > 2.4:
                plate = gray[y: y + h, x: x + w]
                text = pytesseract.image_to_string(plate, config='--psm 11')
                print('text=', text)
                cv2.imshow('Plate', plate)
                cv2.moveWindow('Plate', 780, 10)
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 3)
                cv2.putText(frame, text, (x - 20, y - 10),
                            1, 2.2, (0, 255, 0), 3)

    cv2.imshow('Image', frame)
    cv2.moveWindow('Image', 45, 10)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
