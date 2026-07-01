"""
GUI for real-time face enrollment and recognition.

Usage:
    python gui_app.py
"""
import tkinter as tk
from tkinter import messagebox, simpledialog, ttk

import cv2
from PIL import Image, ImageTk

from face_utils import add_person, delete_person, load_encodings, match_face

TOLERANCE = 0.6
PROCESS_SCALE = 0.25


class FaceRecognitionApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Face Recognition")
        self.root.geometry("900x560")
        self.root.protocol("WM_DELETE_WINDOW", self.on_close)

        self.cap = cv2.VideoCapture(0)
        if not self.cap.isOpened():
            messagebox.showerror("Camera error", "Could not open webcam.")
            self.root.destroy()
            return

        self.pending_enroll_name = None  # set while waiting for a "Capture" click
        self.current_frame = None

        self._build_layout()
        self._refresh_people_list()
        self._update_frame()

    # ---------- UI setup ----------
    def _build_layout(self):
        video_frame = ttk.Frame(self.root)
        video_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=8, pady=8)

        self.video_label = ttk.Label(video_frame)
        self.video_label.pack(fill=tk.BOTH, expand=True)

        self.status_var = tk.StringVar(value="Ready")
        ttk.Label(video_frame, textvariable=self.status_var, anchor="w").pack(fill=tk.X)

        sidebar = ttk.Frame(self.root, width=220)
        sidebar.pack(side=tk.RIGHT, fill=tk.Y, padx=8, pady=8)

        ttk.Label(sidebar, text="Enrolled People", font=("", 11, "bold")).pack(anchor="w")
        self.people_listbox = tk.Listbox(sidebar, height=15)
        self.people_listbox.pack(fill=tk.BOTH, expand=True, pady=(4, 8))

        self.enroll_btn = ttk.Button(sidebar, text="Enroll New Face", command=self.start_enroll)
        self.enroll_btn.pack(fill=tk.X, pady=2)

        ttk.Button(sidebar, text="Delete Selected", command=self.delete_selected).pack(
            fill=tk.X, pady=2
        )
        ttk.Button(sidebar, text="Refresh List", command=self._refresh_people_list).pack(
            fill=tk.X, pady=2
        )

    def _refresh_people_list(self):
        data = load_encodings()
        # de-duplicate names for display, keep them sorted
        names = sorted(set(data["names"]))
        self.people_listbox.delete(0, tk.END)
        for name in names:
            self.people_listbox.insert(tk.END, name)

    # ---------- Enrollment flow ----------
    def start_enroll(self):
        name = simpledialog.askstring("Enroll New Face", "Enter the person's name:")
        if not name:
            return
        self.pending_enroll_name = name
        self.enroll_btn.config(text="Capture Now", command=self.capture_enroll)
        self.status_var.set(f"Position '{name}'s face in frame, then click 'Capture Now'.")

    def capture_enroll(self):
        if self.current_frame is None:
            return
        name = self.pending_enroll_name
        rgb_frame = cv2.cvtColor(self.current_frame, cv2.COLOR_BGR2RGB)
        success = add_person(name, rgb_frame)

        if success:
            self.status_var.set(f"Enrolled '{name}'.")
            self._refresh_people_list()
        else:
            self.status_var.set("No face detected — try again with better lighting/angle.")

        self.pending_enroll_name = None
        self.enroll_btn.config(text="Enroll New Face", command=self.start_enroll)

    def delete_selected(self):
        selection = self.people_listbox.curselection()
        if not selection:
            return
        name = self.people_listbox.get(selection[0])
        if messagebox.askyesno("Delete", f"Remove all enrolled faces for '{name}'?"):
            delete_person(name)
            self._refresh_people_list()
            self.status_var.set(f"Deleted '{name}'.")

    # ---------- Video loop ----------
    def _update_frame(self):
        ok, frame = self.cap.read()
        if ok:
            self.current_frame = frame
            self._draw_recognition_overlays(frame)
            display_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            image = Image.fromarray(display_frame)
            photo = ImageTk.PhotoImage(image=image)
            self.video_label.imgtk = photo  # keep a reference
            self.video_label.configure(image=photo)

        self.root.after(15, self._update_frame)

    def _draw_recognition_overlays(self, frame):
        # Skip recognition entirely while mid-enrollment to keep it simple/fast
        if self.pending_enroll_name is not None:
            return

        import face_recognition  # local import keeps startup snappy

        data = load_encodings()
        known_encodings, known_names = data["encodings"], data["names"]
        if not known_encodings:
            return

        small = cv2.resize(frame, (0, 0), fx=PROCESS_SCALE, fy=PROCESS_SCALE)
        rgb_small = cv2.cvtColor(small, cv2.COLOR_BGR2RGB)
        locations = face_recognition.face_locations(rgb_small)
        encodings = face_recognition.face_encodings(rgb_small, locations)
        scale_back = 1 / PROCESS_SCALE

        for (top, right, bottom, left), encoding in zip(locations, encodings):
            name, _ = match_face(known_encodings, known_names, encoding, TOLERANCE)
            color = (0, 200, 0) if name != "Unknown" else (0, 0, 255)
            top, right, bottom, left = (int(v * scale_back) for v in (top, right, bottom, left))
            cv2.rectangle(frame, (left, top), (right, bottom), color, 2)
            cv2.rectangle(frame, (left, bottom - 22), (right, bottom), color, cv2.FILLED)
            cv2.putText(
                frame, name, (left + 5, bottom - 5),
                cv2.FONT_HERSHEY_DUPLEX, 0.55, (255, 255, 255), 1,
            )

    def on_close(self):
        self.cap.release()
        self.root.destroy()


if __name__ == "__main__":
    root = tk.Tk()
    app = FaceRecognitionApp(root)
    root.mainloop()
