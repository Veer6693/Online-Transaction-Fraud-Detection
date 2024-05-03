# Online Transaction Fraud Detection

This project tackles the challenge of identifying fraudulent transactions. Users can input customer data, transaction details, and the model predicts whether the transaction is legitimate or fraudulent.

## Project Highlights

- **Multi-Model Training:** Explores various algorithms including Logistic Regression, BernoulliNB, Decision Tree, Random Forest, and XGBoost.
- **Hyperparameter Tuning:** Utilizes Optuna for optimal hyperparameter tuning of XGBoost, maximizing its performance.
- **Modular Design:** Well-structured code for maintainability and scalability.
- **End-to-End Functionality:** Seamless integration of data preprocessing, model training, prediction, and deployment.
- **Flask Deployment:** Web application built with Flask, making the model accessible through a user-friendly interface.
- **Render Deployment:** Project hosted on Render, allowing easy online access.

## Getting Started

This project requires Python libraries like pandas, scikit-learn, xgboost, optuna, and Flask. Ensure you have them installed before proceeding.

To run the project locally:

1. Clone this repository.
2. Install dependencies: `pip install -r requirements.txt`
4. Run the application: `python app.py`
5. Access the web app at [http://127.0.0.1:5000/](http://127.0.0.1:5000/)

## Project Structure
- **data:** Contains the dataset used for training and evaluation.
- **models:** Houses the implementation of various fraud detection models.
- **preprocessing:** Includes code for data cleaning and feature engineering.
- **utils:** Utility functions used throughout the project.
- **app.py:** The main Flask application script.

## Usage Example
The web app provides a form to enter transaction details. Once submitted, the model predicts the transaction's legitimacy and displays the result.

## Future Improvements
- **Enhanced Model Interpretability:** Develop features to explain model predictions, increasing user trust and understanding.
- **Improved User Interface:** Enhance the web app's design and user experience for better usability.

## Deployment
This project is currently deployed on Render. [Link](https://online-transaction-fraud-detection.onrender.com)

## Website Preview
![Screenshot 2024-05-03 203810](https://github.com/Veer6693/PowerBI-Dashboard/assets/102231617/97962f46-0205-4dca-86b0-727c1456564a)
![Screenshot 2024-05-03 203941](https://github.com/Veer6693/PowerBI-Dashboard/assets/102231617/0d389b48-9c01-43ae-9db3-f8ea5def2c16)
![Screenshot 2024-05-03 204047](https://github.com/Veer6693/PowerBI-Dashboard/assets/102231617/5da082f6-0496-478d-9962-be4e907ba031)

