"""
Enroll a person's face for recognition.

Usage:
    python enroll.py --name "Alice"                    # capture from webcam
    python enroll.py --name "Alice" --image photo.jpg   # use an existing photo
"""
import argparse

import cv2

from face_utils import add_person


def capture_from_webcam(camera_index=0):
    cap = cv2.VideoCapture(camera_index)
    if not cap.isOpened():
        raise RuntimeError(f"Could not open camera {camera_index}")

    print("Press SPACE to capture, ESC to cancel.")
    frame = None
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        cv2.imshow("Enroll - press SPACE to capture", frame)
        key = cv2.waitKey(1) & 0xFF
        if key == 27:  # ESC
            frame = None
            break
        if key == 32:  # SPACE
            break

    cap.release()
    cv2.destroyAllWindows()
    return frame


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--name", required=True, help="Name of the person to enroll")
    parser.add_argument("--image", help="Path to an existing photo instead of webcam capture")
    parser.add_argument("--camera", type=int, default=0, help="Webcam index")
    args = parser.parse_args()

    if args.image:
        image = cv2.imread(args.image)
        if image is None:
            raise FileNotFoundError(f"Could not read image: {args.image}")
    else:
        image = capture_from_webcam(args.camera)
        if image is None:
            print("Cancelled, no image captured.")
            return

    rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    success = add_person(args.name, rgb_image)

    if not success:
        print("No face detected. Try again with better lighting or a clearer photo.")
        return

    print(f"Enrolled '{args.name}'.")


if __name__ == "__main__":
    main()
