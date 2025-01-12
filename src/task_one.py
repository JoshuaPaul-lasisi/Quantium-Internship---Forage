# Import necessary libraries
import pandas as pd
import numpy as np
import os
import seaborn as sns
import matplotlib.pyplot as plt
from IPython.display import display, Markdown

# Load data
purchase = pd.read_csv('../data/raw/QVI_purchase_behaviour.csv')
transaction = pd.read_excel('../data/raw/QVI_transaction_data.xlsx')

# Display the first 3 rows of the datasets
print("Purchase Data:")
display(purchase.head(3))

print("\nTransaction Data:")
display(transaction.head(3))

# Inspect the data
print("\nData Information:")
print("Purchase Data Info:")
print(purchase.info())

print("\nTransaction Data Info:")
print(transaction.info())

# Clean transaction data: Extract pack size and brand from product name
transaction['PROD_NAME'] = transaction['PROD_NAME'].str.replace(r'g$', '', regex=True)
transaction['PACK_SIZE(g)'] = transaction['PROD_NAME'].str.extract(r'(\d+)').astype(int)
transaction['PROD_NAME'] = transaction['PROD_NAME'].str.replace(r'\d+', '', regex=True)
transaction['PROD_NAME'] = transaction['PROD_NAME'].str.strip()
transaction['BRAND'] = transaction['PROD_NAME'].str.split().str[0]
transaction['PROD_NAME'] = transaction['PROD_NAME'].apply(lambda x: ' '.join(x.split()[1:]))

# Display new columns for verification
print("\nTransaction Data with Cleaned Columns:")
display(transaction[['PROD_NAME', 'PACK_SIZE(g)', 'BRAND']].head(3))

# Merge datasets
data = pd.merge(transaction, purchase, on='LYLTY_CARD_NBR')

# Identify and remove outliers in TOT_SALES
q1_sales, q3_sales = np.percentile(data['TOT_SALES'], [25, 75])
iqr_sales = q3_sales - q1_sales
lower_bound_sales = q1_sales - 1.5 * iqr_sales
upper_bound_sales = q3_sales + 1.5 * iqr_sales
data = data[(data['TOT_SALES'] >= lower_bound_sales) & (data['TOT_SALES'] <= upper_bound_sales)]

# Ensure TOT_SALES is numeric and drop NaN values
data['TOT_SALES'] = pd.to_numeric(data['TOT_SALES'], errors='coerce')
data = data.dropna(subset=['TOT_SALES'])

# Confirm there are no missing values in TOT_SALES
print("\nMissing TOT_SALES After Cleanup:")
print(data['TOT_SALES'].isna().sum())

# Create folder for processed data if not exists
folder_path = '../data/processed/'
os.makedirs(folder_path, exist_ok=True)
data.to_csv(f'{folder_path}QVI_merged_data.csv', index=False)

# Display the processed data
print("\nProcessed Data Preview:")
display(data.head())

# Create folder for visualizations if not exists
image_folder = '../reports/images/'
os.makedirs(image_folder, exist_ok=True)

# Sales by customer segment (LIFESTAGE and PREMIUM_CUSTOMER)
segment_sales = data.groupby(['LIFESTAGE', 'PREMIUM_CUSTOMER'])['TOT_SALES'].sum().reset_index()
segment_sales = segment_sales.sort_values(by='TOT_SALES', ascending=False)

# Plot total sales by customer segment
plt.figure(figsize=(12, 6))
sns.barplot(data=segment_sales, x='LIFESTAGE', y='TOT_SALES', hue='PREMIUM_CUSTOMER')
plt.title('Total Sales by Customer Segment')
plt.ylabel('Total Sales')
plt.xticks(rotation=45)
plt.savefig(f'{image_folder}sales_by_customer_segment.png')
plt.close()

# Sales by brand
brand_sales = data.groupby('BRAND')['TOT_SALES'].sum().reset_index()
top_brands = brand_sales.sort_values(by='TOT_SALES', ascending=False).head(10)

# Plot top 10 brands by total sales
plt.figure(figsize=(10, 6))
sns.barplot(data=top_brands, x='BRAND', y='TOT_SALES', color='skyblue')
plt.title('Top 10 Brands by Total Sales')
plt.ylabel('Total Sales')
plt.xticks(rotation=45)
plt.savefig(f'{image_folder}top_brands_by_sales.png')
plt.close()

# Sales by pack size
pack_size_sales = data.groupby('PACK_SIZE(g)')['TOT_SALES'].sum().reset_index()

# Plot total sales by pack size
plt.figure(figsize=(10, 6))
sns.barplot(data=pack_size_sales, x='PACK_SIZE(g)', y='TOT_SALES', color='teal')
plt.title('Total Sales by Pack Size')
plt.ylabel('Total Sales')
plt.xlabel('Pack Size (g)')
plt.xticks(rotation=45)
plt.savefig(f'{image_folder}pack_size_sales.png')
plt.close()

# Monthly sales trends
data['MONTH'] = data['DATE'].dt.to_period('M').astype(str)
monthly_sales = data.groupby('MONTH')['TOT_SALES'].sum().reset_index()

# Plot monthly sales trends
plt.figure(figsize=(12, 6))
sns.lineplot(data=monthly_sales, x='MONTH', y='TOT_SALES', marker='o', color='green')
plt.title('Monthly Sales Trends')
plt.ylabel('Total Sales')
plt.xticks(rotation=45)
plt.savefig(f'{image_folder}monthly_sales_trends.png')
plt.close()

# Save key metrics to CSV
segment_sales.to_csv(f'{folder_path}QVI_segment_sales.csv', index=False)
brand_sales.to_csv(f'{folder_path}QVI_brand_sales.csv', index=False)
pack_size_sales.to_csv(f'{folder_path}QVI_pack_size_sales.csv', index=False)
monthly_sales.to_csv(f'{folder_path}QVI_monthly_sales.csv', index=False)

# Insights and Recommendations (could be included in a markdown file)
insights = """
### 1. Customer Segment Analysis
#### Insights:
- Older Families and Young Singles/Couples are top contributors to sales.
- Premium Customers generate the highest sales.
- New Families and Older Singles/Couples show limited engagement.

#### Recommendations:
- Focus marketing efforts on Premium Customers, especially in high-performing segments.
- Tailor offerings for underperforming lifestages like New Families.

### 2. Brand Performance
#### Insights:
- Kettle leads in sales, followed by Smiths and Doritos.
- Smaller brands like Pringles and Cobs show lower sales.

#### Recommendations:
- Maintain Kettle's dominance through marketing and product innovation.
- Explore partnerships with high-performing brands.

### 3. Pack Size Analysis
#### Insights:
- 175g pack size leads in sales.
- Smaller sizes like 70g, 90g, and 110g show lower sales.

#### Recommendations:
- Focus on promoting high-demand sizes like 175g and 135g.
- Consider reducing or discontinuing low-performing sizes.

### 4. Monthly Sales Trends
#### Insights:
- Sales show seasonal trends with peaks and dips.

#### Recommendations:
- Align promotions with high-sales months.
- Address low-sales months through targeted campaigns.

### 5. General Recommendations for Strategic Growth
- Focus on high-performing customer segments and products.
- Optimize inventory and supply chain based on sales data.
- Explore customer lifetime value (CLTV) for long-term retention strategies.
"""

# Display insights
display(Markdown(insights))
