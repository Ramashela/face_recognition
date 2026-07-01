# Real-Time Face Recognition

Detects and recognizes faces from your webcam in real time.

## How it works
1. `enroll.py` — register people by capturing their face (or from a photo) and saving a numeric "encoding" for them.
2. `recognize.py` — opens the webcam, detects faces each frame, and matches them against enrolled encodings, drawing a name box around each match.

## Setup

```bash
pip install -r requirements.txt
```

> **Note on `face_recognition`**: it depends on `dlib`, which needs a C++ compiler.
> - **Windows**: easiest via `pip install cmake dlib` first, or install via `conda install -c conda-forge dlib`.
> - **Mac**: `brew install cmake` first.
> - **Linux**: `sudo apt install cmake build-essential` first.
>
> If dlib install is painful, let me know — there's an alternative version of this project using OpenCV's DNN face detector + a lighter embedding model that avoids dlib entirely.

## 1. Enroll a person

From webcam (press SPACE to capture, ESC to cancel):
```bash
python enroll.py --name "Alice"
```

From an existing photo:
```bash
python enroll.py --name "Alice" --image photos/alice.jpg
```

Repeat for each person. Encodings are stored in `known_faces/encodings.pkl`.

## 2. Run real-time recognition

```bash
python recognize.py
```

- Green box + name = recognized
- Red box + "Unknown" = face detected but not enrolled
- Press `q` to quit

Optional flags:
```bash
python recognize.py --camera 1 --tolerance 0.5 --process-scale 0.25
```
- `--camera`: which webcam index to use (default 0)
- `--tolerance`: lower = stricter matching (default 0.6)
- `--process-scale`: shrink frame before processing for speed (default 0.25 = 25%)

## Files
```
face_recognition_project/
├── enroll.py          # register a new face
├── recognize.py       # real-time webcam recognition
├── known_faces/        # stores encodings.pkl (created automatically)
├── requirements.txt
└── README.md
```
