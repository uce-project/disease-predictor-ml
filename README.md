# Disease Prediction Model

This project is a web-based application that predicts diseases based on the symptoms provided by the user. It utilizes a machine learning model trained on a dataset of diseases and their corresponding symptoms. The application is built with Streamlit, a Python library for creating web apps for machine learning and data science.

## Features

- Predicts diseases from a wide range of symptoms.
- Provides precautions for the predicted disease.
- Simple and intuitive user interface.
- Built with Streamlit, making it easy to run and deploy.

## How to Run

1. **Clone the repository:**
   ```bash
   git clone https://github.com/your-username/disease-predictor-ml.git
   ```
2. **Navigate to the project directory:**
   ```bash
   cd disease-predictor-ml
   ```
3. **Install the dependencies:**
   ```bash
   pip install -r requirements.txt
   ```
4. **Run the Streamlit application:**
   ```bash
   streamlit run app.py
   ```
The application will open in your web browser.

## Model Training

The disease prediction model is a `RandomForestClassifier` from the scikit-learn library. The model was trained on a dataset of diseases and symptoms. The training process involves the following steps:

1. **Data Preprocessing:** The symptoms are one-hot encoded using `MultiLabelBinarizer` to handle multiple symptoms for a single disease. The disease labels are encoded using `LabelEncoder`.
2. **Train-Test Split:** The dataset is split into training and testing sets, with 80% of the data used for training and 20% for testing.
3. **Model Training:** A `RandomForestClassifier` is trained on the training data.
4. **Model Saving:** The trained model, along with the label encoders, are saved to disk for later use in the application.

The training code can be found in the `model/train.ipynb` Jupyter notebook.

## Dataset

The dataset used for training the model is located in the `model/dataset/` directory and consists of three CSV files:

- `dataset.csv`: Contains the main dataset with diseases and their corresponding symptoms.
- `symptom_Description.csv`: Provides a description for each disease.
- `symptom_precaution.csv`: Contains the precautions for each disease.

## Dependencies

The following libraries are required to run the application:

- streamlit
- joblib
- scikit-learn
- pandas
- graphviz
- numpy
- streamlit-tags

You can install them using the `requirements.txt` file:
```bash
pip install -r requirements.txt
```

## License

This project is licensed under the MIT License. See the `LICENSE` file for more details.
