import pandas as pd
from sklearn.preprocessing import StandardScaler


def preprocessing():

    df = pd.read_csv("Customer Data.csv")
    scalar = StandardScaler()

    # filling mean value in place of missing values in the dataset
    df["MINIMUM_PAYMENTS"] = df["MINIMUM_PAYMENTS"].fillna(df["MINIMUM_PAYMENTS"].mean())
    df["CREDIT_LIMIT"] = df["CREDIT_LIMIT"].fillna(df["CREDIT_LIMIT"].mean())



    # drop CUST_ID column because it is not used
    df.drop(columns=["CUST_ID"],axis=1,inplace=True)


    scaled_df = scalar.fit_transform(df)

    return scaled_df


if __name__ == "__main__":
    preprocessing()