import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
data=pd.read_csv('Online_Retail.csv', encoding="latin1")
print(data.head)
print(data.shape)
print(data.columns)
print(data.describe)
data_null = round(100+(data.isnull().sum())/len(data), 2)
print(data.isnull)
data.drop(['StockCode'], axis=1, inplace=True)
print(data.shape)
data['CustomerID'] = data['CustomerID'].astype(str)
print(data.info)

#DATA PREPARATION
data['Amount'] =  data['Quantity']+data['UnitPrice']
data_monetary=data.groupby('CustomerID')['Amount'].sum()
print(data_monetary.head())

#THE PRODUCT SOLD MOST
data_monetary=data.groupby('Description')['Quantity'].sum().sort_values(ascending=False)
print(data_monetary.head())

data_monetary=data.groupby('CustomerID')['Quantity'].sum().sort_values(ascending=False)
print(data_monetary.head())
data_monetary=data.groupby('Description')['InvoiceNo'].count().sort_values(ascending=False)

print(data_monetary.head())
data_monetary=data_monetary.reset_index()

#THE SALES FOR THE LAST MONTH
data['InvoiceDate'] = pd.to_datetime(data['InvoiceDate'], format='%m/%d/%Y %H:%M')
print(data.head())

#MAXIMUM DATE
max_date = max(data['InvoiceDate'])
max_date
print(data.head())

#MINIMUM DATE
min_date = min(data['InvoiceDate'])
min_date
print(data.head())

#THEMINIMUM DAYS AND MAXIMUM DAYS
max_date=max(data['InvoiceDate'])
print(min_date)
diffrence=(max_date-min_date)
print(diffrence)

last_month =  (max_date - pd.DateOffset(months=1)).month
last_month_year =  (max_date - pd.DateOffset(months=1)).year

last_month_sales =  data[
    (data['InvoiceDate'].dt.month == last_month) & (data['InvoiceDate'].dt.year == last_month_year)
    ]

#LAST MONTH TOTAL SALES
print('last month sales Data:')
print(last_month_sales)

#Total sales
Totalsales = last_month_sales['Quantity'] * last_month_sales['UnitPrice']
totalsales = Totalsales.sum()
print(f'Total Sales for last month: {totalsales}')


total_Amount_sales=data['Amount'].count()
print(total_Amount_sales)
from sklearn.cluster import KMeans
# Define the input data for clustering
X = data[['Amount']]
#Determine the optimal number of clusters using the Elbow Method
wcss = []
for i in range(1, 11):
    kmeans = KMeans(n_clusters=i, init='k-means++', random_state=42)
    kmeans.fit(X)
    wcss.append(kmeans.inertia_)
    
# Plotting the Elbow Method
plt.figure(figsize=(10,6))
plt.plot(range(1, 11), wcss, marker='o', linestyle='--')
plt.title('Elbow Method')
plt.xlabel('Number of clusters')
plt.ylabel('WCSS')
plt.xticks(range(1, 11))
plt.grid(True)
plt.show()
from sklearn.metrics import silhouette_score
import numpy as np

# Calculate silhouette scores for different number of clusters
silhouette_scores = []
for i in range(2, 11):
    kmeans = KMeans(n_clusters=i, init='k-means++', random_state=42)
    kmeans.fit(X)
    silhouette_scores.append(silhouette_score(X, kmeans.labels_))

# Plotting the silhouette scores
plt.figure(figsize=(10,6))
plt.plot(range(2, 11), silhouette_scores, marker='o', linestyle='-')
plt.title('Silhouette Method')
plt.xlabel('Number of clusters')
plt.ylabel('Silhouette Score')
plt.xticks(range(2, 11))
plt.grid(True)
plt.show()
