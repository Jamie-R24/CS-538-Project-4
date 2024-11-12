import pandas as pd 
from sklearn.preprocessing import StandardScaler 
from sklearn.cluster import KMeans 
from sklearn.decomposition import PCA 
import matplotlib.pyplot as plt
import seaborn as sns

folder_path = "./EDA Graphs/"

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

