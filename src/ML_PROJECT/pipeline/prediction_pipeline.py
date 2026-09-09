import os
import joblib
import pandas as pd


class PredictPipeline:

    def __init__(self):
        pass

    def predict(self, data):
        try:
            model_path = os.path.join("models", "train_model.pkl")
            scaler_path = os.path.join("models", "scaler.pkl")

            # Load model and scaler
            model = joblib.load(model_path)
            scaler = joblib.load(scaler_path)

            # Scale input data
            scaled_data = scaler.transform(data)

            # Prediction
            prediction = model.predict(scaled_data)

            return prediction

        except Exception as e:
            raise Exception(e)


class CustomData:

    def __init__(
        self,
        age: int,
        bmi: float,
        children: int,
        sex_male: int,
        smoker_yes: int,
        region_northwest: int,
        region_southeast: int,
        region_southwest: int
    ):

        self.age = age
        self.bmi = bmi
        self.children = children
        self.sex_male = sex_male
        self.smoker_yes = smoker_yes
        self.region_northwest = region_northwest
        self.region_southeast = region_southeast
        self.region_southwest = region_southwest

    def get_data_as_dataframe(self):

        data = {
            "age": [self.age],
            "bmi": [self.bmi],
            "children": [self.children],
            "sex_male": [self.sex_male],
            "smoker_yes": [self.smoker_yes],
            "region_northwest": [self.region_northwest],
            "region_southeast": [self.region_southeast],
            "region_southwest": [self.region_southwest]
        }

        return pd.DataFrame(data)