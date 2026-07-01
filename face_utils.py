"""
Shared helpers for loading/saving face encodings and recognizing faces.
Used by both the CLI scripts (enroll.py / recognize.py) and gui_app.py.
"""
import os
import pickle

import face_recognition
import numpy as np

ENCODINGS_PATH = os.path.join("known_faces", "encodings.pkl")


def load_encodings():
    if os.path.exists(ENCODINGS_PATH):
        with open(ENCODINGS_PATH, "rb") as f:
            return pickle.load(f)
    return {"names": [], "encodings": []}


def save_encodings(data):
    os.makedirs("known_faces", exist_ok=True)
    with open(ENCODINGS_PATH, "wb") as f:
        pickle.dump(data, f)


def encode_from_image(rgb_image):
    """Return the encoding of the first detected face in an RGB image, or None."""
    locations = face_recognition.face_locations(rgb_image)
    if not locations:
        return None
    encodings = face_recognition.face_encodings(rgb_image, [locations[0]])
    return encodings[0]


def add_person(name, rgb_image):
    """Encode a face from an image and add it to the saved encodings. Returns True/False."""
    encoding = encode_from_image(rgb_image)
    if encoding is None:
        return False
    data = load_encodings()
    data["names"].append(name)
    data["encodings"].append(encoding)
    save_encodings(data)
    return True


def delete_person(name):
    """Remove all enrolled encodings for a given name."""
    data = load_encodings()
    keep_names, keep_encodings = [], []
    for n, e in zip(data["names"], data["encodings"]):
        if n != name:
            keep_names.append(n)
            keep_encodings.append(e)
    save_encodings({"names": keep_names, "encodings": keep_encodings})


def match_face(known_encodings, known_names, face_encoding, tolerance=0.6):
    """Return (name, distance) for the closest match, or ('Unknown', None)."""
    if not known_encodings:
        return "Unknown", None
    distances = face_recognition.face_distance(known_encodings, face_encoding)
    best_idx = int(np.argmin(distances))
    if distances[best_idx] <= tolerance:
        return known_names[best_idx], distances[best_idx]
    return "Unknown", distances[best_idx]
