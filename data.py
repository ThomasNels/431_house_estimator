import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
pd.set_option("display.max_columns", None)

class ZillowDataProcessor:
    def __init__(self, prop_2016_path, prop_2017_path, train_2016_path, train_2017_path):
        self.prop_2016_path = prop_2016_path
        self.prop_2017_path = prop_2017_path
        self.train_2016_path = train_2016_path
        self.train_2017_path = train_2017_path

        self.drop_cols = [
            'calculatedbathnbr', 'fullbathcnt', 'latitude', 'longitude',
            'propertycountylandusecode', 'censustractandblock', 'regionidneighborhood',
            'unitcnt', 'taxvaluedollarcnt', 'structuretaxvaluedollarcnt', 'landtaxvaluedollarcnt',
            'assessmentyear', 'airconditioningtypeid', 'heatingorsystemtypeid',
            'buildingqualitytypeid', 'propertyzoningdesc'
        ]

        self.fill_mode = [
            'regionidcity', 'regionidzip', 'fips', 'propertylandusetypeid',
            'rawcensustractandblock', 'regionidcounty'
        ]
        self.fill_median = [
            'lotsizesquarefeet', 'finishedsquarefeet12', 'yearbuilt',
            'calculatedfinishedsquarefeet', 'taxamount', 'roomcnt',
            'bathroomcnt', 'bedroomcnt'
        ]
        self.fill_zero = ['numberofstories', 'garagecarcnt', 'garagetotalsqft']

        self.min_log = -0.4
        self.max_log = 0.4

    def load_data(self):
        self.properties_2016 = pd.read_csv(self.prop_2016_path, low_memory=False)
        self.properties_2017 = pd.read_csv(self.prop_2017_path, low_memory=False)
        self.train_2016 = pd.read_csv(self.train_2016_path, low_memory=False)
        self.train_2017 = pd.read_csv(self.train_2017_path, low_memory=False)

    def drop_high_missing(self, df, name, threshold=0.8):
        missing_percent = df.isnull().mean()
        cols_to_drop = missing_percent[missing_percent > threshold].index.tolist()
        print(f"🛑 Dropping {len(cols_to_drop)} columns from {name} (>{threshold*100}% missing)")
        return df.drop(columns=cols_to_drop)

    def clean_properties(self):
        self.properties_2016 = self.drop_high_missing(self.properties_2016, "Properties 2016")
        self.properties_2017 = self.drop_high_missing(self.properties_2017, "Properties 2017")

        self.properties_2016.drop(columns=self.drop_cols, errors='ignore', inplace=True)
        self.properties_2017.drop(columns=self.drop_cols, errors='ignore', inplace=True)

        for col in self.fill_mode:
            if col in self.properties_2016.columns:
                mode_val = self.properties_2016[col].mode()[0]
                self.properties_2016[col] = self.properties_2016[col].fillna(mode_val)
                self.properties_2017[col] = self.properties_2017[col].fillna(mode_val)

        for col in self.fill_median:
            if col in self.properties_2016.columns:
                median_val = self.properties_2016[col].median()
                self.properties_2016[col] = self.properties_2016[col].fillna(median_val)
                self.properties_2017[col] = self.properties_2017[col].fillna(median_val)

        for col in self.fill_zero:
            if col in self.properties_2016.columns:
                self.properties_2016[col] = self.properties_2016[col].fillna(0)
                self.properties_2017[col] = self.properties_2017[col].fillna(0)

    def merge_data(self):
        print("\n🔗 Merging training data with property data")
        train_2016_merged = self.train_2016.merge(self.properties_2016, on='parcelid', how='left')
        train_2017_merged = self.train_2017.merge(self.properties_2017, on='parcelid', how='left')
        print(f"✅ Train 2016 shape: {train_2016_merged.shape}")
        print(f"✅ Train 2017 shape: {train_2017_merged.shape}")

        self.train_properties = pd.concat([train_2016_merged, train_2017_merged], axis=0).reset_index(drop=True)

        self.train_properties['transactiondate'] = pd.to_datetime(self.train_properties['transactiondate'])
        self.train_properties['transactiondate'] = self.train_properties['transactiondate'].dt.strftime('%Y%m').astype(int)

    def filter_logerror(self):
        self.train_properties = self.train_properties[
            (self.train_properties['logerror'] >= self.min_log) &
            (self.train_properties['logerror'] <= self.max_log)
        ]

    def prepare(self):
        print("📦 Loading data...")
        self.load_data()
        print("🧹 Cleaning property datasets...")
        self.clean_properties()
        self.merge_data()
        print("✂️ Filtering outliers in logerror...")
        self.filter_logerror()
        print("✅ Data ready for modeling!")

    def get_processed_data(self):
        return self.train_properties