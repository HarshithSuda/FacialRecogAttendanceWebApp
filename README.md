# 🎓 Facial Recognition Attendance System

A complete facial recognition-based attendance system for colleges, including:

- 📹 Video demonstration
- 📄 Project report
- 📊 Presentation (PPT)
- 💻 Source code

## 📌 Project Summary

A full-stack facial recognition-based attendance tracking system developed to automate classroom attendance and reduce manual effort.

---

## 🔧 Key Features

- 🔍 Real-time Face Detection: Implemented using the **Viola-Jones algorithm** (Haar Cascade Classifier) for fast and accurate facial localization.
- 🧠 Deep Face Recognition: Leveraged **ResNet-34 Convolutional Neural Network** to extract robust facial features and generate 128-dimensional embeddings.
- 📏 Face Matching: Used **Euclidean Distance** for identity verification — more effective than **Cosine Similarity** for ResNet-based vectors.
- 🗄️ Database Integration: Designed a scalable backend to store student details and facial embeddings using PostgreSQL.
- 🏫 Deployment-Ready: Built for real-world deployment within the college, significantly reducing attendance tracking time.

---

## 🧰 Tech Stack

| Layer       | Technologies                          |
|-------------|----------------------------------------|
| Frontend    | HTML, CSS, JavaScript                 |
| Backend     | Flask (Python)                        |
| Recognition | OpenCV, Haar Cascade, ResNet-34 (CNN) |
| Database    | PostgreSQL                            |


