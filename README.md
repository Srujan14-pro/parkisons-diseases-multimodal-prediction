Parkinson’s Disease Detection Using Multimodal Deep Learning
MRI + Speech + Gait Analysis | Late Fusion | Optimization | Streamlit Deployment

Parkinson’s Disease (PD) is a progressive neurodegenerative disorder that affects motor and speech functions. 
Early detection is crucial. This project builds a multimodal AI-based diagnostic system that analyzes MRI images, speech recordings, and gait patterns to predict Parkinson’s disease with high accuracy.
This work integrates deep learning, machine learning, Horned Lizard Optimization Algorithm (HLOA), and late fusion techniques into a single deployed system.
 Key Features
Multi-modal system using:
 MRI classification with EfficientNet-B4 Speech signal analysis using MFCC + ML classifiers
 Gait pattern analysis (stride length, cadence, variability)
Late Fusion Technique to combine probabilities from all three models
Horned Lizard Optimization (HLOA) for hyperparameter tuning
Streamlit Web App Deployment (User uploads MRI/Speech/Gait data)
98% Accuracy (MRI model), strong performance across all modalities
Explainability Ready (GradCAM, feature visualizations – optional)


 Model Details
1️⃣ MRI Classification (Deep Learning)
Model: EfficientNet-B4
Transfer learning + fine-tuning
Input size: 224×224
Output: Binary classification (PD / Normal)
Achieved: 97.6% accuracy, high AUC
2️⃣ Speech-Based Classification
Features: MFCC, jitter, shimmer, pitch variations
Classifiers tested: SVM, RandomForest, XGBoost
Best model: Random Forest (RF)
Captures vocal biomarkers of PD
3️⃣ Gait-Based Classification
Extracted features:
Stride length
Cadence
Step-time variability
ML models trained for gait abnormality detection
Uses biomechanical patterns linked to PD
🔗 Late Fusion Method

Each model outputs its own PD probability:
MRI:    P1
Speech: P2
Gait:   P3


Fusion formula (weighted):
Final Score = w1*P1 + w2*P2 + w3*P3
Default weights (adjustable):
MRI: 0.40
Speech: 0.35
Gait: 0.25

If one input is missing, weights auto-normalize.

 Streamlit Web App

A simple and intuitive UI where users can:
Upload MRI image
Upload speech audio
Upload gait CSV
Get:
Individual modality predictions
Final fused prediction
PD vs Normal output
Run locally:
pip install -r requirements.txt
streamlit run app.py
📷 App Features
Home page
Learn page (disease information)
Chatbot (Q&A)
MRI prediction page
Gait analysis page
Speech analysis page
(You can add screenshots here)
 Tech Stack
Deep Learning
TensorFlow / Keras
EfficientNet
Machine Learning
Scikit-learn
Random Forest
XGBoost
Audio Processing
Librosa
Image Processing
OpenCV
Numpy
Deployment
Streamlit
Pickle/Joblib
Python
Results 
Summary
Modality	Best Model	Accuracy
MRI	EfficientNet-B4	97.6%
Speech	Random Forest	94–96%
Gait	ML Classifier	91–95%
Late Fusion	Weighted Voting	98%+
Future Improvements
Add Grad-CAM visualizations for MRI
Integrate LSTM models for gait time-series
Build a cloud deployment (AWS/Render/HuggingFace)
Add explainability (SHAP) for speech + gait
Expand dataset with clinical contributions
Feel free to submit pull requests or open issues.

MIT License 
 Author
Srujan
Deep Learning & Medical AI Researcher
Streamlit / Computer Vision / ML Engineering
