import json
from datetime import datetime, date
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import streamlit as st
from PIL import Image

APP_DIR = Path(__file__).parent
DATA_DIR = APP_DIR / "data"
FACES_DIR = DATA_DIR / "faces"
MODEL_PATH = DATA_DIR / "face_model.yml"
LABELS_PATH = DATA_DIR / "labels.json"
ATTENDANCE_PATH = DATA_DIR / "attendance.csv"

DATA_DIR.mkdir(exist_ok=True)
FACES_DIR.mkdir(exist_ok=True)

st.set_page_config(
    page_title="FaceAttend | Smart Attendance",
    page_icon="◉",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---------- Styling ----------
st.markdown("""
<style>
    .stApp { background: #f6f8fc; }
    [data-testid="stHeader"] { background: rgba(246,248,252,0.92); }
    .block-container { padding-top: 2rem; padding-bottom: 3rem; max-width: 1180px; }
    .hero {
        background: linear-gradient(115deg, #172554 0%, #1d4ed8 100%);
        color: white; padding: 1.6rem 1.8rem; border-radius: 18px;
        margin-bottom: 1.2rem;
    }
    .hero h1 { color: white; margin: 0 0 .35rem 0; font-size: 2rem; }
    .hero p { color: #dbeafe; margin: 0; font-size: 1rem; }
    .metric-card {
        background: white; border: 1px solid #e8edf5; border-radius: 14px;
        padding: 1rem 1.15rem; min-height: 108px;
    }
    .metric-label { color: #64748b; font-size: .88rem; }
    .metric-value { color: #172554; font-size: 1.8rem; font-weight: 700; margin-top: .25rem; }
    .soft-card {
        background: white; border: 1px solid #e8edf5; border-radius: 14px;
        padding: 1.2rem; margin-bottom: .8rem;
    }
    .muted { color: #64748b; }
    div.stButton > button { border-radius: 9px; font-weight: 600; }
    div[data-testid="stFileUploader"] { background: white; border-radius: 12px; }
</style>
""", unsafe_allow_html=True)


def load_labels():
    if LABELS_PATH.exists():
        try:
            return json.loads(LABELS_PATH.read_text())
        except Exception:
            return {}
    return {}


def load_attendance():
    if ATTENDANCE_PATH.exists():
        try:
            return pd.read_csv(ATTENDANCE_PATH)
        except Exception:
            pass
    return pd.DataFrame(columns=["Name", "Date", "Time", "Status"])


def get_detector():
    cascade_path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
    detector = cv2.CascadeClassifier(cascade_path)
    if detector.empty():
        raise RuntimeError("Could not load the OpenCV face detector.")
    return detector


def pil_to_gray(image_file):
    image = Image.open(image_file).convert("RGB")
    rgb = np.array(image)
    return cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)


def detect_faces(gray):
    detector = get_detector()
    return detector.detectMultiScale(
        gray, scaleFactor=1.15, minNeighbors=5, minSize=(65, 65)
    )


def normalized_face(gray, rect):
    x, y, w, h = [int(v) for v in rect]
    face = gray[y:y+h, x:x+w]
    face = cv2.resize(face, (200, 200), interpolation=cv2.INTER_AREA)
    face = cv2.equalizeHist(face)
    return face


def train_model():
    labels = load_labels()
    faces, ids = [], []
    for person_id, person_name in labels.items():
        person_dir = FACES_DIR / person_id
        if not person_dir.exists():
            continue
        for path in person_dir.glob("*.png"):
            img = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
            if img is not None:
                faces.append(cv2.resize(img, (200, 200)))
                ids.append(int(person_id))
    if not faces:
        return False, 0
    recognizer = cv2.face.LBPHFaceRecognizer_create()
    recognizer.train(faces, np.array(ids, dtype=np.int32))
    recognizer.write(str(MODEL_PATH))
    return True, len(faces)


def recognize(gray):
    if not MODEL_PATH.exists():
        return None, None, "No students registered yet. Register students first."
    rects = detect_faces(gray)
    if len(rects) == 0:
        return None, None, "No face detected. Try better lighting and face the camera."
    if len(rects) > 1:
        return None, None, "More than one face detected. Please capture one person at a time."
    recognizer = cv2.face.LBPHFaceRecognizer_create()
    recognizer.read(str(MODEL_PATH))
    face = normalized_face(gray, rects[0])
    person_id, confidence = recognizer.predict(face)
    labels = load_labels()
    name = labels.get(str(person_id))
    # LBPH confidence is a distance: lower values indicate a closer match.
    if name is None or confidence > 65:
        return None, float(confidence), "Face not recognised. This person may not be registered."
    return name, float(confidence), "Recognised"


def add_attendance(name):
    df = load_attendance()
    today = date.today().isoformat()
    if not df.empty and ((df["Name"] == name) & (df["Date"] == today)).any():
        return False
    now = datetime.now()
    row = pd.DataFrame([{
        "Name": name,
        "Date": today,
        "Time": now.strftime("%I:%M:%S %p"),
        "Status": "Present",
    }])
    pd.concat([df, row], ignore_index=True).to_csv(ATTENDANCE_PATH, index=False)
    return True


def metric_card(label, value):
    st.markdown(
        f'<div class="metric-card"><div class="metric-label">{label}</div>'
        f'<div class="metric-value">{value}</div></div>',
        unsafe_allow_html=True,
    )


labels = load_labels()
attendance = load_attendance()
today = date.today().isoformat()
today_count = 0 if attendance.empty else int((attendance["Date"] == today).sum())

with st.sidebar:
    st.markdown("## ◉ FaceAttend")
    st.caption("AI-powered attendance demo")
    page = st.radio(
        "Navigation",
        ["Dashboard", "Register students", "Mark attendance", "Attendance records"],
        label_visibility="collapsed",
    )
    st.divider()
    st.caption("Built with Python, OpenCV and Streamlit")
    st.caption("Educational prototype — verify attendance manually.")

st.markdown(
    '<div class="hero"><h1>Face Recognition Attendance</h1>'
    '<p>A simple way to register students, recognise faces and record attendance.</p></div>',
    unsafe_allow_html=True,
)

if page == "Dashboard":
    labels = load_labels()
    attendance = load_attendance()
    today_count = 0 if attendance.empty else int((attendance["Date"] == today).sum())
    c1, c2, c3 = st.columns(3)
    with c1: metric_card("Registered students", len(labels))
    with c2: metric_card("Present today", today_count)
    with c3: metric_card("Total attendance entries", len(attendance))

    st.write("")
    left, right = st.columns([1.15, 0.85])
    with left:
        st.markdown('<div class="soft-card">', unsafe_allow_html=True)
        st.subheader("How it works")
        st.markdown(
            "1. **Register** each student with a few clear face photos.\n\n"
            "2. **Train** the recogniser using the registered face samples.\n\n"
            "3. **Capture** one face at a time to identify the student.\n\n"
            "4. **Record** the date and time when attendance is marked."
        )
        st.markdown("</div>", unsafe_allow_html=True)
    with right:
        st.markdown('<div class="soft-card">', unsafe_allow_html=True)
        st.subheader("Getting started")
        st.write("Start by adding at least one student.")
        if labels:
            st.success(f"{len(labels)} student(s) registered.")
        else:
            st.info("No students registered yet.")
        st.write("Use the sidebar to open **Register students**.")
        st.markdown("</div>", unsafe_allow_html=True)

    st.warning(
        "Privacy note: Use only photos collected with clear consent. "
        "This demo stores face samples and attendance in the app's local data folder."
    )

elif page == "Register students":
    st.subheader("Register a student")
    st.write("Enter a name and upload 3–5 clear photos of the same person. "
             "Use front-facing photos with good lighting and no face coverings.")
    with st.form("register_form", clear_on_submit=True):
        name = st.text_input("Student name", placeholder="e.g. Aanya Sharma")
        uploads = st.file_uploader(
            "Upload face photos",
            type=["jpg", "jpeg", "png"],
            accept_multiple_files=True,
            help="Upload 3–5 photos of one person. Each photo should contain one clear face.",
        )
        submitted = st.form_submit_button("Register student", type="primary")

    if submitted:
        clean_name = " ".join(name.strip().split())
        if not clean_name:
            st.error("Please enter a student name.")
        elif not uploads:
            st.error("Please upload at least one face photo.")
        else:
            labels = load_labels()
            if any(existing.lower() == clean_name.lower() for existing in labels.values()):
                st.error("That student name is already registered.")
            else:
                next_id = max([int(k) for k in labels.keys()], default=0) + 1
                person_id = str(next_id)
                person_dir = FACES_DIR / person_id
                person_dir.mkdir(parents=True, exist_ok=True)
                saved = 0
                problems = []
                for idx, upload in enumerate(uploads):
                    try:
                        gray = pil_to_gray(upload)
                        rects = detect_faces(gray)
                        if len(rects) != 1:
                            problems.append(f"{upload.name}: expected one face, found {len(rects)}.")
                            continue
                        face = normalized_face(gray, rects[0])
                        cv2.imwrite(str(person_dir / f"sample_{idx+1}.png"), face)
                        saved += 1
                    except Exception as exc:
                        problems.append(f"{upload.name}: could not process image ({exc}).")
                if saved:
                    labels[person_id] = clean_name
                    LABELS_PATH.write_text(json.dumps(labels, indent=2))
                    ok, count = train_model()
                    st.success(f"Registered {clean_name} using {saved} face sample(s).")
                    if ok:
                        st.success(f"Recognition model trained with {count} face sample(s).")
                else:
                    try:
                        for file in person_dir.glob("*"):
                            file.unlink()
                        person_dir.rmdir()
                    except Exception:
                        pass
                    st.error("No usable face photos were saved. Please try clearer images.")
                for problem in problems:
                    st.warning(problem)

    st.divider()
    st.subheader("Registered students")
    labels = load_labels()
    if labels:
        st.dataframe(
            pd.DataFrame([{"Student ID": k, "Name": v} for k, v in labels.items()]),
            use_container_width=True,
            hide_index=True,
        )
    else:
        st.caption("Your registered students will appear here.")

elif page == "Mark attendance":
    st.subheader("Recognise a student")
    st.write("Allow camera access or upload a photo. Capture one person at a time.")
    source = st.radio("Image source", ["Use camera", "Upload photo"], horizontal=True)
    image_file = None
    if source == "Use camera":
        image_file = st.camera_input("Take a face photo")
    else:
        image_file = st.file_uploader("Choose a face photo", type=["jpg", "jpeg", "png"])

    if image_file is not None:
        try:
            gray = pil_to_gray(image_file)
            preview = cv2.cvtColor(gray, cv2.COLOR_GRAY2RGB)
            rects = detect_faces(gray)
            for x, y, w, h in rects:
                cv2.rectangle(preview, (x, y), (x+w, y+h), (37, 99, 235), 3)
            st.image(preview, caption="Face detection preview", use_container_width=True)
            if st.button("Recognise and mark attendance", type="primary"):
                name, confidence, message = recognize(gray)
                if name:
                    st.success(f"Recognised: {name}")
                    if confidence is not None:
                        st.caption(f"Match distance: {confidence:.1f} (lower is a closer match)")
                    if add_attendance(name):
                        st.success(f"Attendance marked for {name}.")
                    else:
                        st.info(f"{name} is already marked present today.")
                else:
                    st.error(message)
                    if confidence is not None:
                        st.caption(f"Match distance: {confidence:.1f}")
        except Exception as exc:
            st.error(f"Could not process this image: {exc}")

    st.info("Tip: Use consistent lighting, keep the face near the centre, and remove sunglasses.")

elif page == "Attendance records":
    st.subheader("Attendance history")
    attendance = load_attendance()
    if attendance.empty:
        st.info("No attendance has been recorded yet.")
    else:
        col1, col2 = st.columns([1, 2])
        with col1:
            selected_date = st.selectbox(
                "Filter by date",
                ["All dates"] + sorted(attendance["Date"].dropna().unique().tolist(), reverse=True),
            )
        filtered = attendance if selected_date == "All dates" else attendance[attendance["Date"] == selected_date]
        st.dataframe(filtered.sort_values(["Date", "Time"], ascending=False),
                     use_container_width=True, hide_index=True)
        st.download_button(
            "Download attendance CSV",
            data=filtered.to_csv(index=False).encode("utf-8"),
            file_name="attendance_records.csv",
            mime="text/csv",
        )

st.divider()
st.caption(
    "FaceAttend is an educational prototype. It uses OpenCV Haar Cascade for face detection "
    "and LBPH for face matching. Recognition can be inaccurate; do not use it as the sole "
    "basis for official attendance or access decisions."
)
