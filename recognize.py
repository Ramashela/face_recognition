"""
Real-time face recognition from webcam.

Usage:
    python recognize.py
    python recognize.py --camera 1 --tolerance 0.5 --process-scale 0.25
"""
import argparse

import cv2
import face_recognition

from face_utils import load_encodings, match_face


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--camera", type=int, default=0, help="Webcam index")
    parser.add_argument("--tolerance", type=float, default=0.6, help="Lower = stricter match")
    parser.add_argument(
        "--process-scale",
        type=float,
        default=0.25,
        help="Scale frame down before processing (speed vs accuracy tradeoff)",
    )
    args = parser.parse_args()

    data = load_encodings()
    known_encodings = data["encodings"]
    known_names = data["names"]
    if not known_names:
        print(
            "No enrolled faces found. Run enroll.py first, e.g.:\n"
            '  python enroll.py --name "Alice"'
        )
        return
    print(f"Loaded {len(known_names)} enrolled face(s).")

    cap = cv2.VideoCapture(args.camera)
    if not cap.isOpened():
        raise RuntimeError(f"Could not open camera {args.camera}")

    print("Press 'q' to quit.")

    while True:
        ok, frame = cap.read()
        if not ok:
            break

        small_frame = cv2.resize(frame, (0, 0), fx=args.process_scale, fy=args.process_scale)
        rgb_small = cv2.cvtColor(small_frame, cv2.COLOR_BGR2RGB)

        face_locations = face_recognition.face_locations(rgb_small)
        face_encodings = face_recognition.face_encodings(rgb_small, face_locations)

        scale_back = 1 / args.process_scale

        for (top, right, bottom, left), face_encoding in zip(face_locations, face_encodings):
            name, _ = match_face(known_encodings, known_names, face_encoding, args.tolerance)
            color = (0, 200, 0) if name != "Unknown" else (0, 0, 255)

            top = int(top * scale_back)
            right = int(right * scale_back)
            bottom = int(bottom * scale_back)
            left = int(left * scale_back)

            cv2.rectangle(frame, (left, top), (right, bottom), color, 2)
            cv2.rectangle(frame, (left, bottom - 25), (right, bottom), color, cv2.FILLED)
            cv2.putText(
                frame, name, (left + 6, bottom - 6),
                cv2.FONT_HERSHEY_DUPLEX, 0.6, (255, 255, 255), 1,
            )

        cv2.imshow("Face Recognition (press q to quit)", frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
