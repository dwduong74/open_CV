import socket
import time
import cv2
import numpy as np

sock_video = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
sock_ctrl = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
server_ip = '192.168.2.176'

addr_video = (server_ip, 6000)
addr_ctrl = (server_ip, 6001)


FRAME_WIDTH = 320
FRAME_HEIGHT = 240
FPS = 20

def detect_tennis_ball(frame):
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    lower_yellow = np.array([29, 86, 6])
    upper_yellow = np.array([64, 255, 255])
    mask = cv2.inRange(hsv, lower_yellow, upper_yellow)
    kernel = np.ones((5,5), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if len(contours) == 0:
        return (frame,(-1,-1),-1)

    cnt = max(contours, key=cv2.contourArea)
    area = cv2.contourArea(cnt)
    if area > 100:
        (x, y), radius = cv2.minEnclosingCircle(cnt)
        center = (int(x), int(y))
        radius = int(radius)
        cv2.circle(frame, center, radius, (0, 255, 0), 2)
        coord_text = f'({center[0]}, {center[1]})'
        cv2.putText(frame, coord_text, (center[0] - 40, center[1] - radius - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
        cv2.putText(frame, 'Tenis Ball', (center[0] - 40, center[1] + radius + 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        cv2.line(frame, (int(FRAME_WIDTH/2), center[1]), center, (0, 255, 0), 1, cv2.LINE_AA, 0)
        cv2.circle(frame, (int(FRAME_WIDTH/2), center[1]), 5, (0,0,255), -1, cv2.LINE_4, 0)
    else:
        center = (-1,-1)
        radius = -1
    return (frame,center,radius)
 
def main():
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FPS, FPS)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, FRAME_WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_HEIGHT)

    if not cap.isOpened():
        print("Không mở được camera")
        return

    #cv2.namedWindow('Video', cv2.WINDOW_NORMAL)
    #cv2.resizeWindow('Video', int(1280/2), int(500))

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame, ball, radius = detect_tennis_ball(frame)

        encoded, buffer = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 50])
        data_frame = buffer.tobytes()

        message = f"{ball[0]},{ball[1]},{radius}"

        if len(data_frame) < 60000:
            sock_video.sendto(data_frame, addr_video)
        else:
            print("Frame quá lớn, không gửi được")

        sock_ctrl.sendto(message.encode(), addr_ctrl)

        #cv2.imshow('Video', frame)
        # print(ball,radius)
        if cv2.waitKey(1) & 0xFF == 27:
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()