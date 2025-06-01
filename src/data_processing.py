import os
import pandas as pd
import numpy as np
import joblib

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.feature_selection import chi2, SelectKBest

from src.logger import get_logger
from src.custom_exception import CustomException


logger = get_logger(__name__)


class DataProcessing:
    def __init__(self, input_path, output_path):
        self.input_path = input_path
        self.output_path = output_path
        self.label_encoder = {}
        self.scaler = StandardScaler()
        self.df = None
        self.X = None
        self.y = None
        self.selected_features = []

        os.makedirs(output_path, exist_ok=True)
        logger.info("Data Processing Initialized....")

    def load_data(self):
        try:
            self.df = pd.read_csv(self.input_path)
            logger.info("Data Loaded successfully")
        except Exception as e:
            logger.error(f"Error while loading Data : {e}")
            raise CustomException("Failed to load data", e)
        
    def preprocess_data(self):
        try:
            self.df = self.df.drop(columns=['Patient_ID'])
            self.X = self.df.drop(columns=["Survival_Prediction"])
            self.y = self.df['Survival_Prediction']

            categorical_cols = self.X.select_dtypes(include=['object']).columns
            for col in categorical_cols:
                le = LabelEncoder()
                self.X[col] = le.fit_transform(self.X[col])
                self.label_encoder[col] = le

            logger.info("Data processing - Label encoding done...")
        except Exception as e:
            logger.error(f"Error while processing Data : {e}")
            raise CustomException("Failed to process data", e)
        
    def feature_selection(self):
        try:
            X_train, X_test, y_train, y_test = train_test_split(self.X, self.y, test_size=0.2, random_state=42)
            X_cat = X_train.select_dtypes(include=['int64', 'float64'])
            chi2_selector = SelectKBest(score_func=chi2, k='all')
            chi2_selector.fit(X_cat, y_train)
            chi2_scores = pd.DataFrame({"Features": X_cat.columns,
                            "Chi2_Scores": chi2_selector.scores_}).sort_values(by=['Chi2_Scores'], ascending=False)
            top_features = chi2_scores.head(5)["Features"].tolist()
            self.selected_features = top_features
            logger.info(f"Selected features are {top_features}")
            
            self.X = self.X[top_features]
            logger.info("Feature selection completed")
        except Exception as e:
            logger.error(f"Error while Feature selection : {e}")
            raise CustomException("Failed to select features", e)

    def split_and_scale(self):
        try:
            X_train, X_test, y_train, y_test = train_test_split(self.X, self.y, test_size=0.2, random_state=42, stratify=self.y)
            X_train = self.scaler.fit_transform(X_train)
            X_test = self.scaler.transform(X_test)

            logger.info("Splitting and scaling done successfully")

            return X_train, X_test, y_train, y_test
        except Exception as e:
            logger.error(f"Error while splitting and scaling data : {e}")
            raise CustomException("Failed to split and scale data", e)

    def save_data_and_scaler(self, X_train, X_test, y_train, y_test):
        try:
            joblib.dump(X_train, os.path.join(self.output_path, 'X_train.pkl'))
            joblib.dump(X_test, os.path.join(self.output_path, 'X_test.pkl'))
            joblib.dump(y_train, os.path.join(self.output_path, 'y_train.pkl'))
            joblib.dump(y_test, os.path.join(self.output_path, 'y_test.pkl'))
            joblib.dump(self.scaler, os.path.join(self.output_path, 'scaler.pkl'))

            logger.info("Files saved successfully")
        except Exception as e:
            logger.error(f"Error while saving data : {e}")
            raise CustomException("Failed to save data", e)
        
    def run(self):
        self.load_data()
        self.preprocess_data()
        self.feature_selection()
        X_train, X_test, y_train, y_test = self.split_and_scale()
        self.save_data_and_scaler(X_train, X_test, y_train, y_test)

        logger.info("Data Processing pipeline executed successfully...")


if __name__ == "__main__":
    input_path = "artifacts/raw/data.csv"
    output_path = "artifacts/processed"

    processor = DataProcessing(input_path, output_path)
    processor.run()