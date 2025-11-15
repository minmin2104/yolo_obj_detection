import random
import cv2 as cv
from ultralytics import YOLO


def get_color(class_num):
    random.seed(class_num)
    return tuple(random.randint(0, 255) for _ in range(3))


def main():
    yolo = YOLO("yolov8s.pt")
    cap = cv.VideoCapture(0)
    if not cap.isOpened():
        print("Cannot open camera")
        exit()

    while True:
        # Capture frame-by-frame
        ret, frame = cap.read()

        # if frame is read correctly ret is True
        if not ret:
            print("Can't receive frame (stream end?). Exiting ...")
            break
        results = yolo.track(frame, stream=True)
        for result in results:
            class_names = result.names
            for box in result.boxes:
                if box.conf[0] > 0.4:
                    x1, y1, x2, y2 = map(int, box.xyxy[0])

                    cls = int(box.cls[0])
                    class_name = class_names[cls]

                    conf = float(box.conf[0])

                    colour = get_color(cls)

                    cv.rectangle(frame, (x1, y1), (x2, y2), colour, 2)

                    cv.putText(frame, f"{class_name} {conf:.2f}",
                               (x1, max(y1 - 10, 20)),
                               cv.FONT_HERSHEY_SIMPLEX,
                               0.6, colour, 2)
        # Our operations on the frame come here
        # Display the resulting frame
        cv.imshow('YOLO Object Detection ', frame)
        if cv.waitKey(1) == ord('q'):
            break

    # When everything done, release the capture
    cap.release()
    cv.destroyAllWindows()


if __name__ == "__main__":
    main()
