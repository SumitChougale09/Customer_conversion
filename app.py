from flask import Flask, request, render_template, redirect, url_for
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from src.pipeline.prediction_pipeline import CustomData, PredictPipeline

application = Flask(__name__)

app = application

@app.route('/')
def index():
    return render_template('home.html')

@app.route('/predictdata', methods=['GET', 'POST'])
def predict_datapoint():
    if request.method == 'GET':
        return redirect(url_for('index'))
    else:
        try:
            data = CustomData(
                gender=request.form.get('gender'),
                age=int(request.form.get('age', 0)),
                income=float(request.form.get('income', 0)),
                campaign_channel=request.form.get('campaign_channel'),
                campaign_type=request.form.get('campaign_type'),
                ad_spend=float(request.form.get('ad_spend', 0)),
                click_through_rate=float(request.form.get('click_through_rate', 0)),
                conversion_rate=float(request.form.get('conversion_rate', 0)),
                website_visits=int(request.form.get('website_visits', 0)),
                pages_per_visit=float(request.form.get('pages_per_visit', 0)),
                time_on_site=float(request.form.get('time_on_site', 0)),
                social_shares=int(request.form.get('social_shares', 0)),
                email_opens=int(request.form.get('email_opens', 0)),
                email_clicks=int(request.form.get('email_clicks', 0)),
                previous_purchases=int(request.form.get('previous_purchases', 0)),
                loyalty_points=int(request.form.get('loyalty_points', 0))
            )
            pred_df = data.get_data_as_data_frame()
            predict_pipeline = PredictPipeline()
            results = predict_pipeline.predict(pred_df)
            return render_template('result.html', result=results[0])
        except ValueError as e:
            error_message = f"Invalid input: {str(e)}"
            return render_template('home.html', error=error_message)

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5001, debug=True)