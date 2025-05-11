# 🧠 Emotion Detection Project

This project consists of a FastAPI backend serving a machine learning model and a Vite-powered frontend for testing the results.

## 🔧 Setup Instructions

### 1. Download and Install the Model

* Download the model file from Google Drive:
  👉 [Download Model](https://drive.google.com/file/d/1xwXCERgUet7gBg96WAdz19kNhDWNn6Jv/view?usp=sharing)
* Once downloaded, place the file in the `emotion_model/` directory:

  ```
  emotion_model/
  └── your_model_file_here
  ```

---

### 2. Run the FastAPI Backend

The API is used to serve the emotion detection model.

```bash
python ModelAPI.py
```

This will start a FastAPI server (usually at `http://127.0.0.1:8000`).

---

### 3. Run the Frontend (Vite Project)

Navigate to the frontend project directory:

```bash
cd user_interface/project
```

Install dependencies:

```bash
npm install
```

Start the development server:

```bash
npm run dev
```

The app will run at `http://localhost:5173` (or the default Vite port). Make sure your backend is also running to see the results.

---

## ✅ Summary

* 🧠 Model → Put in `emotion_model/`
* 🚀 Backend → `python ModelAPI.py`
* 🌐 Frontend → `cd user_interface/project && npm install && npm run dev`
