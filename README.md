# FaceAttend — Face Recognition Attendance Demo

A simple Streamlit application for a first-year AI project. It demonstrates:
- Face detection using OpenCV Haar Cascade
- Face matching using the LBPH face recogniser
- Student registration from consented face photos
- Attendance logging with date and time
- CSV download of attendance records

## Run locally

Python 3.10 or 3.11 is recommended.

```bash
pip install -r requirements.txt
streamlit run app.py
```

## Deploy on Streamlit Community Cloud

1. Upload `app.py`, `requirements.txt`, and `README.md` to the root of a public GitHub repository.
2. In Streamlit Community Cloud, create an app from that repository.
3. Set the main file path to `app.py`.
4. Deploy.

## Important notes

- Register 3–5 clear photos per person for better results.
- The app expects one face per registration photo and one face per attendance capture.
- LBPH confidence is a distance; lower values indicate a closer match. The app uses a simple threshold and may make mistakes.
- Streamlit Community Cloud's local filesystem is not permanent. Uploaded face samples and attendance records may be lost when the app is restarted, redeployed, or its environment is recreated. This is a classroom demonstration, not a production attendance system.
- Obtain explicit consent before collecting or processing anyone's face images. Keep the data private and delete it when it is no longer needed.
- This is an educational prototype, not a secure identity-verification system. Do not use it as the sole basis for official attendance, access, or disciplinary decisions.
