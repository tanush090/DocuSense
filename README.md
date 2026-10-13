# DocuSense – Smart PDF Document Classifier


[![Open Live App](https://img.shields.io/badge/🚀_Open-Live_App-FF4B4B?style=for-the-badge)](https://docusense-ai-classifier.streamlit.app)

**DocuSense** is an AI/ML-powered application that classifies documents into Bills, Invoices, and Receipts.

🌐 **Live Demo:** https://docusense-ai-classifier.streamlit.app

## 📌 Project Overview
DocuSense is a machine learning project that classifies documents into three categories: **Bill, Invoice, and Receipt**. It extracts text from PDF documents and scanned files and uses a trained machine learning model to predict the document category.

## 🎯 Objectives
- Automate document classification.
- Extract text from PDF files and scanned documents.
- Apply machine learning techniques to text classification.
- Display predicted categories and confidence scores through a user interface.

## 🛠️ Technologies Used
- Python
- Pandas
- Scikit-learn
- TF-IDF Vectorizer
- Logistic Regression
- Streamlit
- PyMuPDF
- Tesseract OCR
- Joblib

## 📂 Dataset
The model was trained using a synthetic dataset containing **3,000 text records** across three classes:

| Document Category | Number of Records |
|---|---:|
| Bill | 1,000 |
| Invoice | 1,000 |
| Receipt | 1,000 |
| **Total** | **3,000** |

The dataset was divided into training and testing sets using an 80:20 split.

## ⚙️ Project Workflow
1. Collect and prepare document text data.
2. Clean and preprocess the text.
3. Convert text into numerical features using TF-IDF.
4. Train a Logistic Regression classifier.
5. Evaluate the trained model using test data.
6. Save the trained model using Joblib.
7. Extract text from uploaded documents using PyMuPDF and OCR where required.
8. Predict the document category through the Streamlit application.

## 📊 Model Evaluation
The model achieved **100% accuracy on the 600-record synthetic test set** during the reported experiment.

**Note:** This result applies to the synthetic test dataset. Performance on real-world bills, invoices, and receipts may differ and requires further validation.

## 🚀 How to Run the Project

### 1. Clone the repository
```bash
git clone YOUR_GITHUB_REPOSITORY_URL
cd DocuSense
```

Replace `YOUR_GITHUB_REPOSITORY_URL` with your actual GitHub repository URL.

### 2. Install dependencies
```bash
pip install -r requirements.txt
```

### 3. Install Tesseract OCR
Install the Tesseract OCR engine on your system if it is not already installed.

### 4. Run the application
```bash
streamlit run app.py
```

## 📁 Project Structure
```text
DocuSense/
├── app.py
├── document_classifier.pkl
├── requirements.txt
├── packages.txt
├── README.md
├── .gitignore
└── src/
    ├── train.py
    ├── evaluate.py
    └── predict.py
```

## 🔮 Future Improvements
- Test the model on more real-world documents.
- Improve performance on scanned and low-quality documents.
- Add support for additional document categories.
- Deploy the application for public use.

## 👨‍💻 Author
**Tanush Kumar**

B.Tech CSE (AI & ML)

---

*DocuSense – Making document classification smarter and simpler.*
