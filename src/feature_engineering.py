import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin


class FeatureEngineering(BaseEstimator, TransformerMixin):

    def fit(self, X, y=None):
        return self

    def transform(self, X):

        X = X.copy()

        # Tenure groups
        X["TenureGroup"] = pd.cut(
            X["tenure"],
            bins=[0, 12, 24, 48, 100],
            labels=["New", "Mid", "Established", "Loyal"]
        )

        # Average monthly spending
        X["AvgMonthlySpend"] = X["TotalCharges"] / X["tenure"].replace(0, 1)

        # Charge level
        X["ChargeLevel"] = pd.qcut(
            X["MonthlyCharges"],
            q=3,
            labels=["Low", "Medium", "High"]
        )

        return X