import sys
import pandas as pd
from src.exception import CustomException
from src.utils import load_object
import os

class PredictPipeline:
    def __init__(self):
        pass
  
    def predict(self, features):
        try:
            model_path = 'artifacts/model.pkl' # Define the correct path to your model file
            preprocessor_path = 'artifacts/proprocessor.pkl'
            print("Before Loading")
            model = load_object(file_path=model_path)
            preprocessor = load_object(file_path=preprocessor_path)
            print("After Loading")
            data_scaled = preprocessor.transform(features)
            preds = model.predict(data_scaled)
            return preds
        
        except Exception as e:
            raise CustomException(e, sys)

class CustomData:
    def __init__(self,
                 gender: str,
                 age: int,
                 income: float,
                 campaign_channel: str,
                 campaign_type: str,
                 ad_spend: float,
                 click_through_rate: float,
                 conversion_rate: float,
                 website_visits: int,
                 pages_per_visit: float,
                 time_on_site: float,
                 social_shares: int,
                 email_opens: int,
                 email_clicks: int,
                 previous_purchases: int,
                 loyalty_points: int):

        self.gender = gender
        self.age = age
        self.income = income
        self.campaign_channel = campaign_channel
        self.campaign_type = campaign_type
        self.ad_spend = ad_spend
        self.click_through_rate = click_through_rate
        self.conversion_rate = conversion_rate
        self.website_visits = website_visits
        self.pages_per_visit = pages_per_visit
        self.time_on_site = time_on_site
        self.social_shares = social_shares
        self.email_opens = email_opens
        self.email_clicks = email_clicks
        self.previous_purchases = previous_purchases
        self.loyalty_points = loyalty_points

    def get_data_as_data_frame(self):
        try:
            custom_data_input_dict = {
                "Gender": [self.gender],
                "Age": [self.age],
                "Income": [self.income],
                "CampaignChannel": [self.campaign_channel],
                "CampaignType": [self.campaign_type],
                "AdSpend": [self.ad_spend],
                "ClickThroughRate": [self.click_through_rate],
                "ConversionRate": [self.conversion_rate],
                "WebsiteVisits": [self.website_visits],
                "PagesPerVisit": [self.pages_per_visit],
                "TimeOnSite": [self.time_on_site],
                "SocialShares": [self.social_shares],
                "EmailOpens": [self.email_opens],
                "EmailClicks": [self.email_clicks],
                "PreviousPurchases": [self.previous_purchases],
                "LoyaltyPoints": [self.loyalty_points]
            }

            return pd.DataFrame(custom_data_input_dict)

        except Exception as e:
            raise CustomException(e, sys)