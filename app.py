from flask import Flask, render_template, request
import pandas as pd
from datetime import date, datetime

from src.pipeline.predict_pipeline import PredictPipeline

app = Flask(__name__)


@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['GET', 'POST'])
def predict():
    if request.method == 'POST':
        # Get the input values from the form
        transaction_amount = float(request.form['transaction_amount'])

        product_category = request.form['product_category']

        payment_method = request.form['payment_method']

        account_create_day_str = request.form['account_create_days']
        account_create_day = datetime.strptime(account_create_day_str, '%Y-%m-%d')
        account_age_days = (datetime.now() - account_create_day).days

        customer_age = float(request.form['customer_age'])

        transaction_date_str = request.form.get('transaction_date')
        transaction_date = datetime.strptime(transaction_date_str, '%Y-%m-%dT%H:%M')
        transaction_hour = int(transaction_date.hour)
        transaction_day = transaction_date.day
        transaction_dow = transaction_date.weekday()
        transaction_month = transaction_date.month

        quantity = int(request.form['quantity'])

        device_used = request.form['device_used']

        billing_address = request.form['billing_address']
        shipping_address  = request.form['shipping_address']

        if  billing_address == shipping_address:
            is_address_match = 1
        else:
            is_address_match = 0


        data = pd.DataFrame({
            'Transaction Amount': [transaction_amount],
            'Payment Method': [payment_method],
            'Product Category': [product_category],
            'Quantity': [quantity],
            'Customer Age': [customer_age],
            'Device Used': [device_used],
            'Account Age Days': [account_age_days],
            'Transaction Hour': [transaction_hour],
            'Transaction Day': [transaction_day],
            'Transaction DOW': [transaction_dow],
            'Transaction Month': [transaction_month],
            'Is Address Match': [is_address_match]
        })
     
        predict_pipeline = PredictPipeline()
        prediction = predict_pipeline.predict(data)[0]

        if prediction == 1:
            predict = 'Transaction is Fraudulent'
        else:
            predict = 'Transaction is not Fraudulent'

        return render_template('input.html', prediction=predict)

    return render_template('input.html')

if __name__ == "__main__":
    app.run(host='0.0.0.0', port=8080)
