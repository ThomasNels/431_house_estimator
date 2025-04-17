import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
pd.set_option("display.max_columns", None)

properties_2016 = pd.read_csv('./zillow-prize-1/properties_2016.csv', low_memory=False)
properties_2017 = pd.read_csv('./zillow-prize-1/properties_2017.csv', low_memory=False)
train_2016 = pd.read_csv('./zillow-prize-1/train_2016_v2.csv', low_memory=False)
train_2017 = pd.read_csv('./zillow-prize-1/train_2017.csv', low_memory=False)
sample_submission = pd.read_csv('./zillow-prize-1/sample_submission.csv', low_memory=False)

def explore_data(df, name):
    """
    Function to explore a dataset.
    - Prints basic information
    - Shows missing values percentage
    - Displays numerical feature distributions
    """
    # Missing values analysis
    missing_percent = df.isnull().mean() * 100
    missing_percent = missing_percent[missing_percent > 0].sort_values(ascending=False)


    if not missing_percent.empty:
        plt.figure(figsize=(10, 4))
        sns.barplot(x=missing_percent.index, y=missing_percent.values)
        plt.xticks(rotation=90)
        plt.ylabel("Missing Values (%)")
        plt.title(f"Missing Data in {name}")
        plt.show()

    # Distribution of numerical features
    df.hist(figsize=(20, 20), bins=30)
    plt.suptitle(f"Numerical Feature Distributions - {name}", fontsize=14)
    plt.show()

explore_data(properties_2016, "Properties 2016")

def drop_high_missing_cols(df, name, threshold=0.8):
    """
    Drops columns with missing values above a certain threshold.
    :param df: DataFrame to process
    :param name: Name of the dataset for printing
    :param threshold: Proportion of missing values to decide column removal
    :return: Cleaned DataFrame
    """
    missing_percent = df.isnull().mean()
    cols_to_drop = missing_percent[missing_percent > threshold].index.tolist()

    print(f"🛑 Dropping {len(cols_to_drop)} columns from {name} (>{threshold*100}% missing values)")

    return df.drop(columns=cols_to_drop)

# Apply function to both properties datasets
properties_2016 = drop_high_missing_cols(properties_2016, "Properties 2016")
properties_2017 = drop_high_missing_cols(properties_2017, "Properties 2017")

drop_cols = [
    'calculatedbathnbr', 'fullbathcnt', 'latitude', 'longitude', 
    'propertycountylandusecode', 'censustractandblock', 'regionidneighborhood', 
    'unitcnt', 'taxvaluedollarcnt', 'structuretaxvaluedollarcnt', 'landtaxvaluedollarcnt', 'assessmentyear', 'airconditioningtypeid',
    'heatingorsystemtypeid', 'buildingqualitytypeid', 'propertyzoningdesc'
]

# Drop the columns
properties_2016_cleaned = properties_2016.drop(columns=drop_cols, errors="ignore")
properties_2017_cleaned = properties_2017.drop(columns=drop_cols, errors="ignore")

fill_mode = [
    'regionidcity', 'regionidzip', 'fips', 'propertylandusetypeid', 
    'rawcensustractandblock', 'regionidcounty'
]
fill_median = [
    'lotsizesquarefeet', 'finishedsquarefeet12', 'yearbuilt', 
    'calculatedfinishedsquarefeet', 'taxamount', 'roomcnt', 
    'bathroomcnt', 'bedroomcnt'
]
fill_zero = ['numberofstories', 'garagecarcnt', 'garagetotalsqft']

# Apply mode filling
for col in fill_mode:
    properties_2017_cleaned[col] = properties_2017_cleaned[col].fillna(properties_2016[col].mode()[0])
    properties_2016_cleaned[col] = properties_2016_cleaned[col].fillna(properties_2016[col].mode()[0])

# Apply median filling
for col in fill_median:
    properties_2017_cleaned[col] = properties_2017_cleaned[col].fillna(properties_2016[col].median())
    properties_2016_cleaned[col] = properties_2016_cleaned[col].fillna(properties_2016[col].median())

# Apply zero filling
for col in fill_zero:
    properties_2017_cleaned[col] = properties_2017_cleaned[col].fillna(0)
    properties_2016_cleaned[col] = properties_2016_cleaned[col].fillna(0)




def merge_train_properties(train_df, properties_df, name):
    """
    Merges train data with property data on parcelid.
    """
    print(f"\n🔗 Merging {name} train data with properties data")
    merged_df = train_df.merge(properties_df, on='parcelid', how='left')
    print(f"✅ Merged dataset shape: {merged_df.shape}")
    return merged_df

# Merge train datasets with properties datasets
train_2016_merged = merge_train_properties(train_2016, properties_2016_cleaned, "2016")
train_2017_merged = merge_train_properties(train_2017, properties_2017_cleaned, "2017")

plt.figure(figsize=(10, 6))

sns.kdeplot(train_2016['logerror'], label='2016', fill=True, alpha=0.5)
sns.kdeplot(train_2017['logerror'], label='2017', fill=True, alpha=0.5)

plt.title("Logerror Distribution for 2016 vs 2017")
plt.xlabel("Logerror")
plt.ylabel("Density")
plt.legend()
plt.show()
train_properties = pd.concat([train_2016_merged, train_2017_merged], axis=0).reset_index(drop=True)

# Convert 'transactiondate' to YYYYMM format
train_properties['transactiondate'] = pd.to_datetime(train_properties['transactiondate'])
train_properties['transactiondate'] = train_properties['transactiondate'].dt.strftime('%Y%m').astype(int)

min_log, max_log = -0.4, 0.4

train_properties_v2 = train_properties[
    (train_properties["logerror"] >= min_log) &
    (train_properties["logerror"] <= max_log)
]

sns.kdeplot(train_2016['logerror'], label='2016', fill=True, alpha=0.5)
sns.kdeplot(train_2017['logerror'], label='2017', fill=True, alpha=0.5)

plt.title("Logerror Distribution for 2016 vs 2017")
plt.xlabel("Logerror")
plt.ylabel("Density")
plt.legend()
plt.show()
t
