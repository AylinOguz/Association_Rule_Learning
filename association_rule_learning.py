############################################
# 1. Data Preprocessing
############################################

# !pip install mlxtend
import pandas as pd
pd.set_option('display.max_columns', None)
pd.set_option('display.width', 500)
from mlxtend.frequent_patterns import apriori, association_rules

# Dataset: https://archive.ics.uci.edu/ml/datasets/Online+Retail+II

df_ = pd.read_excel("datasets/online_retail_II.xlsx",
                    sheet_name="Year 2010-2011")
df = df_.copy()
df.head()

# Data overview
df.describe(include=["int","float"]).T
df.isnull().sum()
df.shape


# --- Data Cleaning ---
def retail_data_prep(dataframe):
    dataframe.dropna(inplace=True)
    dataframe = dataframe[~dataframe["Invoice"].str.contains("C", na=False)]
    dataframe = dataframe[dataframe["Quantity"] > 0]
    dataframe = dataframe[dataframe["Price"] > 0]
    return dataframe

df = retail_data_prep(df)
df.describe(include=["int","float"]).T


# --- Outlier Detection and Replacement ---
def outlier_threshold(dataframe, variable):
    q1 = dataframe[variable].quantile(0.01)
    q3 = dataframe[variable].quantile(0.99)
    inter_quantile = q3 - q1
    up_limit = q3 + 1.5 * inter_quantile
    low_limit = q1 - 1.5 * inter_quantile
    return up_limit, low_limit

def replace_threshold(dataframe, variable):
    up_limit, low_limit = outlier_threshold(dataframe, variable)
    dataframe.loc[dataframe[variable] > up_limit, variable] = up_limit
    dataframe.loc[dataframe[variable] < low_limit, variable] = low_limit


replace_threshold(df, "Quantity")


def outlier_thresholds(dataframe, variable):
    quartile1 = dataframe[variable].quantile(0.01)
    quartile3 = dataframe[variable].quantile(0.99)
    interquantile_range = quartile3 - quartile1
    up_limit = quartile3 + 1.5 * interquantile_range
    low_limit = quartile1 - 1.5 * interquantile_range
    return low_limit, up_limit

def replace_with_thresholds(dataframe, variable):
    low_limit, up_limit = outlier_thresholds(dataframe, variable)
    dataframe.loc[(dataframe[variable] < low_limit), variable] = low_limit
    dataframe.loc[(dataframe[variable] > up_limit), variable] = up_limit


def retail_data_prep(dataframe):
    dataframe.dropna(inplace=True)
    dataframe = dataframe[~dataframe["Invoice"].str.contains("C", na=False)]
    dataframe = dataframe[dataframe["Quantity"] > 0]
    dataframe = dataframe[dataframe["Price"] > 0]
    replace_with_thresholds(dataframe, "Quantity")
    replace_with_thresholds(dataframe, "Price")
    return dataframe

df = retail_data_prep(df)
df.isnull().sum()
df.describe(include=["int","float"]).T
df["Country"].unique()

############################################
# 2. Preparing the ARL Structure (Invoice-Product Matrix)
############################################

# Here, each invoice represents a basket, and each column represents a product.
# The goal is to create a matrix that indicates whether a product was purchased in that invoice.

df.head()

# Since we want to generate rules specifically for France, we’ll filter the dataset by country.
df_fr = df[df['Country'] == "France"]

df_fr.groupby(['Invoice', 'Description']).agg({"Quantity": "sum"}).head(20)

# unstack() converts MultiIndex rows into columns (pivot-like structure)
df_fr.groupby(['Invoice', 'Description']).agg({"Quantity": "sum"}).unstack().iloc[0:5, 0:5]

# Replace NaN values with 0
df_fr.groupby(['Invoice', 'Description']).agg({"Quantity": "sum"}).unstack().fillna(0).iloc[0:5, 0:5]

# Replace quantities greater than 0 with 1 (binary basket representation)
df_fr.groupby(['Invoice', 'StockCode']).\
    agg({"Quantity": "sum"}).\
    unstack().\
    fillna(0).\
    applymap(lambda x: 1 if x > 0 else 0).iloc[0:5, 0:5]


def create_invoice_product_df(dataframe, id=False):
    if id:
        return dataframe.groupby(['Invoice', "StockCode"])['Quantity'].sum().unstack().fillna(0). \
            applymap(lambda x: 1 if x > 0 else 0)
    else:
        return dataframe.groupby(['Invoice', 'Description'])['Quantity'].sum().unstack().fillna(0). \
            applymap(lambda x: 1 if x > 0 else 0)

fr_inv_pro_df = create_invoice_product_df(df_fr, id=True)
fr_inv_pro_df.iloc[0:5, 0:5]


# Helper function to find the product name by stock code
def check_id(dataframe, stock_code):
    product_name = dataframe[dataframe["StockCode"] == stock_code][["Description"]].values[0].tolist()
    print(product_name)

check_id(df_fr, "84029E")


############################################
# 3. Generating Association Rules
############################################

frequent_itemsets = apriori(fr_inv_pro_df,
                            min_support=0.01,
                            use_colnames=True)

frequent_itemsets.sort_values("support", ascending=False).head(50)

rules = association_rules(frequent_itemsets,
                          metric="support",
                          min_threshold=0.01)

rules[(rules["support"]>0.05) & (rules["confidence"]>0.1) & (rules["lift"]>5)].\
    sort_values(["support","lift"],ascending=[False,False])

# Example interpretation:
# | antecedents | consequents | support | confidence | lift |
# | {milk}      | {bread}     | 0.20    | 0.66       | 1.65 |
#
# support: 20% of transactions contain both milk and bread.
# confidence: 66% of customers who bought milk also bought bread.
# lift: 1.65 means milk and bread are bought together 1.65 times more often than random chance.


############################################
# 4. Building Reusable Script Functions
############################################

# The goal is to automate the entire workflow with reusable Python functions.

def outlier_thresholds(dataframe, variable):
    quartile1 = dataframe[variable].quantile(0.01)
    quartile3 = dataframe[variable].quantile(0.99)
    interquantile_range = quartile3 - quartile1
    up_limit = quartile3 + 1.5 * interquantile_range
    low_limit = quartile1 - 1.5 * interquantile_range
    return low_limit, up_limit

def replace_with_thresholds(dataframe, variable):
    low_limit, up_limit = outlier_thresholds(dataframe, variable)
    dataframe.loc[(dataframe[variable] < low_limit), variable] = low_limit
    dataframe.loc[(dataframe[variable] > up_limit), variable] = up_limit

def retail_data_prep(dataframe):
    dataframe.dropna(inplace=True)
    dataframe = dataframe[~dataframe["Invoice"].str.contains("C", na=False)]
    dataframe = dataframe[dataframe["Quantity"] > 0]
    dataframe = dataframe[dataframe["Price"] > 0]
    replace_with_thresholds(dataframe, "Quantity")
    replace_with_thresholds(dataframe, "Price")
    return dataframe

def create_invoice_product_df(dataframe, id=False):
    if id:
        return dataframe.groupby(['Invoice', "StockCode"])['Quantity'].sum().unstack().fillna(0). \
            applymap(lambda x: 1 if x > 0 else 0)
    else:
        return dataframe.groupby(['Invoice', 'Description'])['Quantity'].sum().unstack().fillna(0). \
            applymap(lambda x: 1 if x > 0 else 0)

def check_id(dataframe, stock_code):
    product_name = dataframe[dataframe["StockCode"] == stock_code][["Description"]].values[0].tolist()
    print(product_name)

def create_rules(dataframe, id=True, country="France"):
    dataframe = dataframe[dataframe['Country'] == country]
    dataframe = create_invoice_product_df(dataframe, id)
    frequent_itemsets = apriori(dataframe, min_support=0.01, use_colnames=True)
    rules = association_rules(frequent_itemsets, metric="support", min_threshold=0.01)
    return rules

df = retail_data_prep(df)
rules = create_rules(df, country="United Kingdom")
rules.sort_values("lift", ascending=False).to_csv("rules.csv")

############################################
# 5. Product Recommendation for Basket Stage Users
############################################

# Example:
# A user adds product with ID 22492 to their basket
product_id = 22492
check_id(df, product_id)
# -> 'MINI PAINT SET VINTAGE'

# Generate recommendations
sorted_rules = rules.sort_values("lift", ascending=False)
recommendation_list = []

for i, product in enumerate(sorted_rules["antecedents"]):
    for j in list(product):
        if j == product_id:
            recommendation_list.append(list(sorted_rules.iloc[i]["consequents"])[0])

recommendation_list[0:3]

check_id(df, 22326)


def arl_recommender(rules_df, product_id, sort_by="lift", rec_count=1):
    sorted_rules = rules_df.sort_values(sort_by, ascending=False)
    recommendation_list = []
    for i, product in enumerate(sorted_rules["antecedents"]):
        for j in list(product):
            if j == product_id:
                recommendation_list.append(list(sorted_rules.iloc[i]["consequents"])[0])
    return recommendation_list[0:rec_count]


arl_recommender(rules, 22492, rec_count=3)
# → [22556, 22551, 22326]
