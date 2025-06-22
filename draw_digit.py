import cv2
import numpy as np

canvas = np.ones((280, 280), dtype=np.uint8) * 255
drawing = False

def draw(event, x, y, flags, param):
    global drawing
    if event == cv2.EVENT_LBUTTONDOWN:
        drawing = True
    elif event == cv2.EVENT_MOUSEMOVE and drawing:
        cv2.circle(canvas, (x, y), 8, 0, -1)
    elif event == cv2.EVENT_LBUTTONUP:
        drawing = False

cv2.namedWindow('Draw Digit')
cv2.setMouseCallback('Draw Digit', draw)

while True:
    cv2.imshow('Draw Digit', canvas)
    key = cv2.waitKey(1)
    if key == ord('s'):
        small = cv2.resize(canvas, (28, 28), interpolation=cv2.INTER_AREA)
        cv2.imwrite('digit.png', small)
        print("Saved as digit.png")
    elif key == 27:
        break

cv2.destroyAllWindows()