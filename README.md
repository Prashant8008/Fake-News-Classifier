# Fake News Detection Project 📰

A streamlined project for detecting fake news using a fine-tuned BERT model. This tool is designed for ease of deployment and high accuracy in identifying fake and real news.

---
dataset link = [https://www.kaggle.com/code/kerneler/starter-liar-preprocessed-dataset-29eb2a99-2/input]

## 🚀 Features
- **Pre-Trained Model**: Utilizes BERT for accurate predictions without the need for extensive retraining.
- **Customizable Preprocessing**: Includes tokenization, stopword removal, and lemmatization.
- **Lightweight Design**: Easy-to-understand structure for quick deployment.
- **Supports Real-Time Testing**: Test the model with a provided dataset or live text input.

---

## 🛠️ Tech Stack
- **Programming Language**: Python 3.8+
- **Libraries**: Transformers, NLTK, PyTorch, Pandas, Scikit-learn
- **Tools**: Kaggle Kernel, Jupyter Notebook

---

## 📂 Project Structure
```
├── data/                 # Dataset files (training and testing CSVs)
│   ├── train.csv         # Training dataset
│   └── test.csv          # Testing dataset
├── models/               # Directory for storing trained models
│   └── bert_fake_news.pth  # Trained BERT model file
├── notebooks/            # Jupyter Notebooks for experimentation
│   └── fake_news_detection.ipynb
├── app/                  # Application for deployment (e.g., Streamlit)
│   └── app.py            # Main application script
├── requirements.txt      # Dependencies for the project
└── README.md             # Project documentation
```

---

## 📝 Usage Instructions

### 1. Setup Environment
- Install required dependencies:
  ```bash
  pip install -r requirements.txt
  ```

### 2. Train the Model (Optional)
- Train the model from scratch using the `fake_news_detection.ipynb` notebook.

### 3. Test the Model
- Use the `test.csv` dataset for validation.
- Run the evaluation script:
  ```bash
  python evaluate.py
  ```

### 4. Deploy the Model
- Deploy using Streamlit or Flask for real-time predictions:
  ```bash
  streamlit run app/app.py
  ```

---

## 🤝 Contributing
Contributions are welcome! Please check our Contributing Guidelines before submitting a PR.

---

## 📜 License
This project is licensed under the MIT License.

---

## 🛡️ Disclaimer
This tool is for educational purposes only and should not be used for critical decision-making processes.

