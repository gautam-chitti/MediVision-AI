# MediVision AI - Advanced Medical Diagnostics

MediVision AI is a next-generation platform for automated anomaly detection in medical X-ray images using AI and advanced machine learning. The platform offers intelligent fracture and TB diagnostics for patients and doctors, a modern, mobile-first UI, and a secure Python/Flask backend.

---

## 🚀 Key Features
- **AI-Powered Fracture/TB Detection:** Instant analysis of X-rays with 99%+ model accuracy using XGBoost & OpenCV.
- **Modern UI:** Beautiful, animated frontend with glass-morphism, TailwindCSS, and dark mode support.
- **Doctor & Patient Dashboards:** Personalized insights, user roles, diagnosis history, and secure, role-based access.
- **PDF Reports & Data Visualization:** One-click report export, interactive chart metrics & explainable results.
- **Mobile Friendly:** Responsive and accessible design for seamless use on any device.
- **Security:** HTTPS-ready, password hashing, data consent management, audit logs, and compliance best practices.

---

## 📦 Folder Structure

```
CancerDetectionWeb/
│
├── app.py                # Main Flask app
├── run.py                # App startup script
├── requirements.txt      # All dependencies
├── Fracture_XGBoost      # ML Model (binary, required)
├── TB_XGBoost            # ML Model (binary, required)
├── templates/            # HTML templates
│    ├── landing.html     # Animated landing page
│    ├── index.html       # Main dashboard
│    ├── login.html       # Login page
│    ├── signup.html      # Signup page
│    ...
├── static/
│    ├── css/             # Stylesheets
│    ├── js/              # JS scripts
│    └── images/          # Images and logos
├── uploads/              # Uploaded medical images (auto-generated)
├── instance/             # SQLite DB files (auto-generated)
└── .gitignore            # Git ignore rules
```

---

## 🛠️ Getting Started

1. **Clone the repository**
```sh
git clone <your-repo-url>
cd CancerDetectionWeb
```

2. **Create a virtual environment (recommended):**
```sh
python -m venv .venv
.venv\Scripts\activate  # (Windows)
# or
source .venv/bin/activate  # (Mac/Linux)
```

3. **Install dependencies:**
```sh
pip install -r requirements.txt
```

4. **Download Models:**
- Ensure files `Fracture_XGBoost`, `TB_XGBoost` are in the project root (see [GitHub Releases](#)).

5. **Run the application:**
```sh
python run.py
```
Visit `http://localhost:5000` or `https://localhost:5000` in your browser.


---

## ✨ Demo Screenshots

![Landing Page](./static/images/demo_landing.png)
![Diagnosis Dashboard](./static/images/demo_dashboard.png)

---

## 🙋‍♂️ Contributing
1. Fork the repo
2. Open a feature branch for your changes
3. Submit a Pull Request!

---

## 📜 License
[MIT License](LICENSE)

---

## 👤 Author Contact
Built by Gauta (and contributors). For questions, file an issue or reach out at yourname@email.com

