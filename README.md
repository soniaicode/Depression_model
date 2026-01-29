# MindCare - Depression Prediction System

**By Soni, PhD Scholar**

A Flask-based web application for depression prediction using machine learning models, voice analysis, and AI-powered therapy assistance.

## 🚀 Quick Start

### Prerequisites
- Python 3.8+
- MongoDB
- pip

### Installation

1. **Clone the repository**
```bash
git clone <repository-url>
cd <project-directory>
```

2. **Install dependencies**
```bash
pip install -r requirements.txt
```

3. **Set up environment variables**
Create a `.env` file in the root directory:
```env
FLASK_SECRET_KEY=your-secret-key-here
MONGO_URI=mongodb://localhost:27017/depression_db
GEMINI_API_KEY=your-gemini-api-key
ADMIN_USERNAME=admin
ADMIN_PASSWORD=Admin@123
```

4. **Start MongoDB**
```bash
mongod
```

5. **Seed initial data**
```bash
python seed_data.py
```

6. **Run the application**
```bash
python app.py
```

Visit: http://127.0.0.1:5000

## 📁 Project Structure

```
├── app.py                          # Main Flask application
├── auth.py                         # Authentication module
├── data.py                         # Data handling utilities
├── requirements.txt                # Python dependencies
├── render.yaml                     # Deployment configuration
├── .env                           # Environment variables (create this)
├── .gitignore                     # Git ignore rules
│
├── config/                        # Configuration files
│
├── data/                          # Data directory
│   ├── audio_samples/            # Voice samples
│   ├── models/                   # Trained ML models
│   ├── processed/                # Processed data
│   └── raw/                      # Raw data
│
├── docs/                          # Documentation
│
├── results/                       # Analysis results
│
├── scripts/                       # Utility scripts
│   ├── data_generator.py         # Generate synthetic data
│   ├── preprocess_data.py        # Data preprocessing
│   ├── train_models.py           # Train ML models
│   ├── train_models_simple.py    # Simple training script
│   └── visualize_confusion_matrix.py
│
├── templates/                     # HTML templates
│   ├── admin/                    # Admin templates
│   ├── base.html                 # Base template
│   ├── landing.html              # Landing page
│   ├── login.html                # Login page
│   ├── signup.html               # Signup page
│   ├── dashboard.html            # User dashboard
│   ├── profile.html              # User profile
│   ├── predict.html              # Prediction page
│   ├── voice_predict.html        # Voice prediction
│   └── ...                       # Other templates
│
└── Core Modules:
    ├── combined_assessment.py     # Combined assessment logic
    ├── gemini_integration.py      # Google Gemini AI integration
    ├── gemini_therapy.py          # AI therapy features
    ├── seed_data.py              # Database seeding
    ├── train_voice_model.py      # Voice model training
    ├── voice_analysis.py         # Voice analysis module
    └── voice_prediction_api.py   # Voice prediction API
```

## 🎯 Features

### User Features
- User registration and authentication
- Depression risk assessment
- Voice-based analysis
- AI-powered therapy chat (Gemini)
- Personal dashboard
- Assessment history
- Profile management

### Admin Features
- Admin dashboard
- User management
- Prediction analytics
- System monitoring
- Model usage statistics

### ML Models
- Logistic Regression
- Random Forest
- Gradient Boosting (XGBoost)
- Enhanced Multimodal
 (TensorFlow)
- Voice Analysis Model

## 🔧 Configuration

### MongoDB Setup
The application uses MongoDB for data storage. Ensure MongoDB is running:
```bash
mongod
```

### Environment Variables
Required variables in `.env`:
- `FLASK_SECRET_KEY`: Flask session secret
- `MONGO_URI`: MongoDB connection string
- `GEMINI_API_KEY`: Google Gemini API key
- `ADMIN_USERNAME`: Admin username
- `ADMIN_PASSWORD`: Admin password

## 📊 Training Models

To train or retrain ML models:
```bash
python scripts/train_models.py
```

For voice model training:
```bash
python train_voice_model.py
```

## 🧪 Testing

Run the application and test:
- Landing page: http://127.0.0.1:5000/
- User signup: http://127.0.0.1:5000/signup
- User login: http://127.0.0.1:5000/login
- Admin login: http://127.0.0.1:5000/admin/login

## 🚀 Deployment

The application is configured for deployment on Render.com using `render.yaml`.

## 📝 API Endpoints

### Public
- `GET /` - Landing page
- `GET /signup` - User registration
- `GET /login` - User login
- `GET /api/health` - Health check

### Protected (User)
- `GET /home` - User home
- `GET /dashboard` - User dashboard
- `GET /profile` - User profile
- `POST /predict` - Make prediction

### Protected (Admin)
- `GET /admin/login` - Admin login
- `GET /admin/dashboard` - Admin dashboard
- `GET /admin/users` - User management
- `GET /admin/predictions` - Predictions view

## 🛠️ Technologies

- **Backend**: Flask, Python
- **Database**: MongoDB
- **ML/AI**: scikit-learn, TensorFlow, Google Gemini
- **Voice**: librosa, soundfile
- **Frontend**: HTML, CSS, JavaScript, Chart.js

## 📄 License

[Add your license here]

## 👥 Contributors

[Add contributors here]

## 📞 Support

For issues or questions, please open an issue on GitHub.


---

## 👤 Author & Copyright

**© 2025 MindCare - Women's Mental Health Platform**

All rights reserved to **Soni, PhD Scholar**

This project is part of PhD research on AI-powered mental health screening and depression detection using multimodal machine learning approaches.

### Research Focus
- Depression detection using questionnaire-based assessment
- Voice-based depression screening using acoustic features
- AI-powered therapy assistance with Gemini integration
- Multimodal machine learning for mental health prediction

### Contact
For research collaboration or inquiries, please contact through the platform.

---

**Disclaimer**: This is a research project and screening tool, not a diagnostic tool. Always consult qualified healthcare professionals for proper mental health evaluation and treatment.
