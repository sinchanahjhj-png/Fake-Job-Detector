# Fake-Job-Detector using Machine Learning
📌 Project Overview

Fake Job Detector is a Machine Learning web application that detects whether a job posting is Genuine or Fraudulent.

With the increasing number of online job scams, this project helps users identify suspicious job descriptions using:

        NLP (Natural Language Processing)
        TF-IDF Vectorization
        Logistic Regression Model
        Rule-based keyword detection

🎯 Problem Statement

Many fake job postings:
Promise high salary with no experience
Ask to contact via WhatsApp
Request registration fees
Use urgent language like "Immediate Hiring"
Contain suspicious email IDs

This project aims to automatically detect such fraudulent job postings.

🧠 Machine Learning Model

Algorithm Used: Logistic Regression
Feature Extraction: TF-IDF Vectorization
Dataset: Real or Fake Job Posting Dataset (Kaggle)

Output Classes:

0 → Genuine Job

1 → Fraudulent Job


⚠️ Additional Security Layer

Apart from ML prediction, the app also performs manual rule checks for suspicious keywords like:

"WhatsApp"
"Fee"
"Earn money"
"Immediate hiring"
"No experience required"
Gmail/Yahoo contact emails

If suspicious content is detected, the app warns the user even if ML confidence is high.

🛠️ Technologies Used

Python
Pandas
NumPy
Scikit-learn
NLTK
Streamlit
Pickle

📂 Project Structure
FakeJobDetector/
│
├── app.py                  # Streamlit Web App
├── model.py                # Model Training Script
├── model.pkl               # Trained ML Model
├── vectorizer.pkl          # TF-IDF Vectorizer
├── fake_job_dataset.csv    # Dataset
├── requirements.txt        # Dependencies
└── README.md


▶️ How to Run the Project Locally
1️⃣ Clone Repository
git clone https://github.com/your-username/FakeJobDetector.git
cd FakeJobDetector

2️⃣ Create Virtual Environment (Recommended)
python -m venv venv
venv\Scripts\activate

3️⃣ Install Dependencies
pip install -r requirements.txt

4️⃣ Train the Model (Only First Time)
python model.py


This will generate:

model.pkl

vectorizer.pkl

5️⃣ Run the Streamlit App
streamlit run app.py


💻 Web App Features

✍️ Enter job description

🔍 ML prediction with confidence score

🚨 Suspicious keyword detection

Clean and user-friendly UI

Real-time fraud detection

📊 Example Output
🔍 Prediction Result  
❌ Fraudulent / Suspicious Job Detected  

⚠ Suspicious keywords detected (WhatsApp / Fee / Urgency pattern)

🔐 Why This Project Is Important

✔ Protects job seekers
✔ Demonstrates NLP skills
✔ Shows real-world ML application
✔ Strong portfolio project for Python/ML roles

📈 Future Improvements

Add deep learning model (LSTM / BERT)

Deploy on Streamlit Cloud

Add email scam detection

Add API integration

Improve UI design

👩‍💻 Author

Sinchana H J
MCA Graduate | Python Developer | Machine Learning Enthusiast
