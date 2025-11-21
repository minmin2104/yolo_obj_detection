import time
import random
import cv2 as cv
from ultralytics import YOLO


def get_color(class_num):
    random.seed(class_num)
    return tuple(random.randint(0, 255) for _ in range(3))


def main():
    """
    Reference:
    https://docs.ultralytics.com/reference/engine/model/#ultralytics.engine.model.Model.track
    https://docs.ultralytics.com/reference/engine/results/#ultralytics.engine.results.Results
    https://docs.ultralytics.com/reference/engine/results/#ultralytics.engine.results.Boxes
    """

    src = "./assets/car.mp4"
    # src = 0
    yolo = YOLO("yolov8s.pt")
    cap = cv.VideoCapture(src)
    if not cap.isOpened():
        print("Cannot open camera")
        exit()

    prev_time = time.time()

    while True:
        ret, frame = cap.read()

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

        current_time = time.time()
        fps = 1 / (current_time - prev_time)
        prev_time = current_time
        cv.putText(frame, f"FPS: {fps:.2f}", (10, 30),
                   cv.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        cv.imshow('YOLO Object Detection ', frame)
        if cv.waitKey(1) == ord('q'):
            break

    cap.release()
    cv.destroyAllWindows()


if __name__ == "__main__":
    main()
