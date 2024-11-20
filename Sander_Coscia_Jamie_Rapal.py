import pandas as pd 
import os
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.preprocessing import StandardScaler 
from sklearn.cluster import KMeans, MiniBatchKMeans
from sklearn.decomposition import PCA 
from sklearn.metrics import silhouette_score, davies_bouldin_score
import matplotlib.pyplot as plt
import seaborn as sns

folder_path = "./EDA Graphs/"
os.environ["LOKY_MAX_CPU_COUNT"] = "4"

df = pd.read_csv('onlineRetail.csv', encoding='ISO-8859-1')
df["InvoiceDate"] = pd.to_datetime(df["InvoiceDate"])
df["Month"] = df["InvoiceDate"].dt.month

country_grouped = df.groupby('Country').agg({
    'Quantity': ['sum', 'mean', 'std', 'count'], 
    'UnitPrice': ['sum', 'mean', 'std'], 
    'CustomerID': 'nunique'
}).reset_index()

country_grouped.columns = [ 
    'Country', 'Total Quantity', 'Average Quantity', 'Quantity Std Dev', 'Total Count', 'Total UnitPrice', 'Average UnitPrice', 'UnitPrice Std Dev', 'Unique Customers'
]

# Plot Total Quantity by Country 
plt.figure(figsize=(14, 8)) 
barplot = sns.barplot(x='Total Quantity', y='Country', data=country_grouped) 
for bar in barplot.patches: 
    barplot.annotate(f'{bar.get_width():.2f}', (bar.get_width(), bar.get_y() + bar.get_height() / 2), ha='left', va='center')
plt.savefig(f'{folder_path}Total Quantity by Country.png', format='png', dpi = 100)
#plt.show() 

# Plot Entries by Country 
plt.figure(figsize=(14, 8)) 
barplot = sns.barplot(x='Total Count', y='Country', data=country_grouped) 
for bar in barplot.patches: 
    barplot.annotate(f'{bar.get_width():.2f}', (bar.get_width(), bar.get_y() + bar.get_height() / 2), ha='left', va='center')
plt.savefig(f'{folder_path}Total Count by Country.png', format='png', dpi = 100)

# Plot Average Unit Price by Country 
plt.figure(figsize=(14, 8))
barplot = sns.barplot(x='Average UnitPrice', y='Country', data=country_grouped)
for bar in barplot.patches:
    barplot.annotate(f'{bar.get_width():.2f}', 
                     (bar.get_width(), bar.get_y() + bar.get_height() / 2), 
                     ha='left', va='center')
plt.title('Average Unit Price by Country')
plt.xlabel('Average Unit Price')
plt.ylabel('Country')
plt.savefig(f'{folder_path}Average Unit by Country.png', format='png', dpi = 100)
#plt.show()

# Plot Number of Unique Customers by Country 
plt.figure(figsize=(14, 8))
barplot = sns.barplot(x='Unique Customers', y='Country', data=country_grouped)
for bar in barplot.patches:
    barplot.annotate(f'{bar.get_width():.0f}', 
                     (bar.get_width(), bar.get_y() + bar.get_height() / 2), 
                     ha='left', va='center')
plt.title('Number of Unique Customers by Country')
plt.xlabel('Unique Customers')
plt.ylabel('Country')
plt.savefig(f'{folder_path}Unique Customers by Country.png', format='png', dpi = 100)
#plt.show()

country_monthly_counts = df.groupby(['Month']).size().reset_index(name = 'Monthly Counts')
plt.figure(figsize=(14, 8))
barplot = sns.barplot(x='Month', y='Monthly Counts', data=country_monthly_counts)
for bar in barplot.patches:
    barplot.annotate(f'{bar.get_height():.0f}', 
                     (bar.get_x() + bar.get_width() / 2, bar.get_height()),  
                     ha='center', va='bottom')  
plt.title('Invoices by Month')
plt.xlabel('Month')
plt.ylabel('Monthly Counts')
plt.savefig(f'{folder_path}Invoices by Month')

#-----------------------------------
#---------------Task 4--------------
#-----------------------------------
cluster_df = pd.read_csv('onlineRetail.csv', encoding='ISO-8859-1')
cluster_df.dropna()
cluster_df.drop_duplicates()
cluster_df['TotalPrice'] = cluster_df['Quantity'] * cluster_df['UnitPrice']
customer_data = cluster_df.groupby('CustomerID').agg({
    'Quantity': 'sum',
    'TotalPrice': 'sum',
    'InvoiceNo': 'nunique',
    'Country': 'first'
}).reset_index()

customer_data.columns = ['CustomerID', 'Total Quantity', 'Total Spent', 'Unique Invoices', 'Country']
customer_data = pd.get_dummies(customer_data, columns=['Country'])
scaler = StandardScaler()
scaled_features = scaler.fit_transform(customer_data.drop(['CustomerID'], axis=1))

#-----------------------------------
#---------------Task 5--------------
#-----------------------------------
kmeans = KMeans(n_clusters=3, init='k-means++', max_iter=300, n_init=10, random_state=0)
clusters = kmeans.fit_predict(scaled_features)

sil_score = silhouette_score(scaled_features, clusters)
db_score = davies_bouldin_score(scaled_features, clusters)
inertia = kmeans.inertia_

print("Vanilla K Means (3 Clusters)")
print(f'\tSilhouette Score: {sil_score}')
print(f'\tDavies-Bouldin Score: {db_score}')
print(f'\tInertia: {inertia}')

#------PCA Optimization----------
pca = PCA(n_components=2)
principal_components = pca.fit_transform(scaled_features)

kmeans_pca = KMeans(n_clusters=3, init='k-means++', max_iter=300, n_init=10, random_state=0)
clusters_pca = kmeans_pca.fit_predict(principal_components)

sil_score = silhouette_score(principal_components, clusters_pca)
db_score = davies_bouldin_score(principal_components, clusters_pca)
inertia = kmeans_pca.inertia_

print("\nPCA K Means (3 Clusters)")
print(f'\tSilhouette Score: {sil_score}')
print(f'\tDavies-Bouldin Score: {db_score}')
print(f'\tInertia: {inertia}')

#------Elbow Method Optimization----------

# Uncomment these lines to recreate the Elbow Method Graph indicating
# The optimal number of clusters

# wcss = []
# for i in range(1, 51):
#     kmeans_opt = KMeans(n_clusters=i, init='k-means++', max_iter=300, n_init=10, random_state=0)
#     kmeans_opt.fit(scaled_features)
#     wcss.append(kmeans_opt.inertia_)

# plt.plot(range(1, 51), wcss)
# plt.title('Elbow Method')
# plt.xlabel('Number of clusters')
# plt.ylabel('WCSS')
# plt.savefig('K Means Elbow Method Results')

optimal_clusters = 38 #This is based on the plot above (Determined to be around 40)
kmeans_opt = KMeans(n_clusters=optimal_clusters, init='k-means++', max_iter=300, n_init=10, random_state=0)
clusters_opt = kmeans_opt.fit_predict(scaled_features)

# Evaluation metrics
sil_score_opt = silhouette_score(scaled_features, clusters_opt)
db_score_opt = davies_bouldin_score(scaled_features, clusters_opt)
inertia_opt = kmeans_opt.inertia_

print(f'\nElbow Method K Means({optimal_clusters} Clusters)')

print(f'\tSilhouette Score (Optimal Clusters): {sil_score_opt}')
print(f'\tDavies-Bouldin Score (Optimal Clusters): {db_score_opt}')
print(f'\tInertia (WCSS) (Optimal Clusters): {inertia_opt}')

#------Feature Selection Optimization-------------
X = customer_data.drop(['CustomerID'], axis = 1)
y = clusters

selector = SelectKBest(score_func=f_classif, k=10)
X_new = selector.fit_transform(X, y)

selected_features = X.columns[selector.get_support()]
#print(selected_features)

scaler = StandardScaler()
scaled_features_selected = scaler.fit_transform(X_new)

kmeans_selected = KMeans(n_clusters = optimal_clusters, init='k-means++', max_iter=300, n_init=10, random_state=0)
clusters_selected = kmeans_selected.fit_predict(scaled_features_selected)

sil_score_selected = silhouette_score(scaled_features_selected, clusters_selected) 
db_score_selected = davies_bouldin_score(scaled_features_selected, clusters_selected) 
inertia_selected = kmeans_selected.inertia_ 

print(f'\nFeature Selection K Means ({optimal_clusters} Clusters)')

print(f'\tSilhouette Score (Feature Selection): {sil_score_selected}') 
print(f'\tDavies-Bouldin Score (Feature Selection): {db_score_selected}')
print(f'\tInertia (WCSS) (Feature Selection): {inertia_selected}')





