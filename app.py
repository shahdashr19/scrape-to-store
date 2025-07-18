import requests
from bs4 import BeautifulSoup
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import sqlite3

# الأساس
base_url = "http://books.toscrape.com/catalogue/"
books = []

# نجرب على أول 3 صفحات
for page in range(1, 50):
    url = f"http://books.toscrape.com/catalogue/page-{page}.html"
    response = requests.get(url)
    soup = BeautifulSoup(response.content, "html.parser")

    items = soup.find_all("article", class_="product_pod")

    for item in items:
        title = item.h3.a["title"]
        price = item.find("p", class_="price_color").text.strip()
        availability = item.find("p", class_="instock availability").text.strip()
        rating = item.p["class"][1]
        link = base_url + item.h3.a["href"]

        books.append({
            "Title": title,
            "Price": price,
            "Availability": availability,
            "Rating": rating,
            "Link": link
        })

# تحويل إلى DataFrame وحفظ CSV
df = pd.DataFrame(books)
df.to_csv("books_scraped.csv", index=False)

print("✅ CSV saved as books_scraped.csv")

# الاتصال بقاعدة بيانات SQLite (هينشئها لو مش موجودة)
conn = sqlite3.connect("books.db")

# حفظ البيانات في جدول اسمه books
df.to_sql("books", conn, if_exists="replace", index=False)

# إغلاق الاتصال بقاعدة البيانات
conn.close()

print("sucsifully storage")



data = pd.read_csv(r"C:\Users\marii\Downloads\books_scraped.csv")
df=pd.DataFrame(data)
df
df.head()
df.shape
df.describe() #show statistics values
df.duplicated().any() #there is no duplicated data
missing_values=print(df.isnull().sum())
df.info()
df["Price"] = df["Price"].str.replace("£", "").astype(float)
df.info()
print(df.nunique())
print("\n Null values:\n", df.isnull().sum())
plt.figure(figsize=(10, 6))
sns.boxplot(x='Price', data=df,color='cyan')
plt.title('Box Plot of Price')
plt.xlabel('Price',fontsize=14)
plt.ylabel('Box Plot Values', fontsize=14)
plt.show()
# Select only numeric columns to check if it has negative values or not
numeric_df = df.select_dtypes(include=['number'])

# Check for any negative values in numeric columns
negative_values = print((numeric_df < 0).any().any())

numeric_cols = df.select_dtypes(include='number')
Q1 = numeric_cols.quantile(0.25) #25%
Q3 = numeric_cols.quantile(0.75) #75%
IQR = Q3 - Q1

# Identify outliers for numeric columns as outlier is a value higher or lower than 1.5*IQR
outliers = ((numeric_cols < (Q1 - 1.5 * IQR)) | (numeric_cols > (Q3 + 1.5 * IQR)))
# Check if there are outliers and print the result
if outliers.any().any():
    print("Outliers detected in the following columns:")
    print(outliers.sum()) #shows each column have how many outlier values
else:
    print("No outliers detected in the dataset.")

Price_outliers = df[(outliers['Price'] == True)]
# print outlier values in Price
print("Outliers in 'Price':")
print(Price_outliers[['Price']])
print('___________________________________________________________')
cleaned_data_df = df[~outliers.any(axis=1)]
cleaned_data_df

import re
# مثال على استخراج أسماء الكتب
def extract_title(title):
    pattern = r".+"  # يمكن ضبطه حسب الحاجة
    match = re.match(pattern, title)
    return match.group(0) if match else None

df['Extracted_Title'] = df['Title'].apply(extract_title)

# وظيفة لاستخراج التواريخ
def extract_date(text):
    pattern = r"\b\d{1,2}/\d{1,2}/\d{4}\b"  # مثال على تنسيق التاريخ
    match = re.findall(pattern, text)
    return match

df['Extracted_Dates'] = df['Title'].apply(extract_date)  # فقط كمثال إذا كان هناك تاريخ في العنوان

import re

# وظيفة لتنظيف الروابط
def clean_link(link):
    pattern = r"http[s]?://[^\s]+"  # تعبير نمطي لمطابقة الروابط
    match = re.match(pattern, link)
    return match.group(0) if match else None

df['Cleaned_Link'] = df['Link'].apply(clean_link)




# Step 7: Visualization & Analysis

# Map Rating to numeric
rating_map = {
    "One": 1, "Two": 2, "Three": 3, "Four": 4, "Five": 5
}
df['Numeric_Rating'] = df['Rating'].map(rating_map)

# 1. Price Distribution
plt.figure(figsize=(12, 6))
sns.histplot(df['Price'], bins=20, kde=True, color='skyblue')
plt.title("Distribution of Book Prices")
plt.xlabel("Price (£)")
plt.ylabel("Frequency")
plt.show()

# 2. Price vs Rating
plt.figure(figsize=(10, 6))
sns.scatterplot(x='Numeric_Rating', y='Price', data=df, palette='viridis')
plt.title("Price vs. Rating")
plt.xlabel("Rating (1-5)")
plt.ylabel("Price (£)")
plt.grid(True)
plt.show()

# 3. Price Comparison by Availability
plt.figure(figsize=(10, 6))
sns.boxplot(x='Availability', y='Price', data=df, palette='Set2')
plt.title("Price Comparison: Available vs Not Available Books")
plt.xlabel("Availability")
plt.ylabel("Price (£)")
plt.xticks(rotation=45)
plt.show()

# 4. Count of Books by Rating
plt.figure(figsize=(8, 5))
sns.countplot(x='Numeric_Rating', data=df, palette='coolwarm')
plt.title("Number of Books by Rating")
plt.xlabel("Rating (1-5)")
plt.ylabel("Count")
plt.show()

# 5. Top 10 Most Expensive Books
top_expensive = df.sort_values(by='Price', ascending=False).head(10)
print("Top 10 Expensive Books:\n")
print(top_expensive[['Title', 'Price', 'Availability', 'Rating']])

# Step 7: Visualization & Analysis

# Map Rating to numeric
rating_map = {
    "One": 1, "Two": 2, "Three": 3, "Four": 4, "Five": 5
}
df['Numeric_Rating'] = df['Rating'].map(rating_map)

# 1. Price Distribution
plt.figure(figsize=(12, 6))
sns.histplot(df['Price'], bins=20, kde=True, color='skyblue')
plt.title("Distribution of Book Prices")
plt.xlabel("Price (£)")
plt.ylabel("Frequency")
plt.show()

# 2. Price vs Rating
plt.figure(figsize=(10, 6))
sns.scatterplot(x='Numeric_Rating', y='Price', data=df, palette='viridis')
plt.title("Price vs. Rating")
plt.xlabel("Rating (1-5)")
plt.ylabel("Price (£)")
plt.grid(True)
plt.show()

# 3. Price Comparison by Availability
plt.figure(figsize=(10, 6))
sns.boxplot(x='Availability', y='Price', data=df, palette='Set2')
plt.title("Price Comparison: Available vs Not Available Books")
plt.xlabel("Availability")
plt.ylabel("Price (£)")
plt.xticks(rotation=45)
plt.show()

# 4. Count of Books by Rating
plt.figure(figsize=(8, 5))
sns.countplot(x='Numeric_Rating', data=df, palette='coolwarm')
plt.title("Number of Books by Rating")
plt.xlabel("Rating (1-5)")
plt.ylabel("Count")
plt.show()

# 5. Top 10 Most Expensive Books
top_expensive = df.sort_values(by='Price', ascending=False).head(10)
print("Top 10 Expensive Books:\n")
print(top_expensive[['Title', 'Price', 'Availability', 'Rating']])

# 6. Price vs Availability (Violin Plot)
plt.figure(figsize=(10, 6))
sns.violinplot(x='Availability', y='Price', data=df, palette='pastel')
plt.title("Does Price Affect Availability?")
plt.xlabel("Availability")
plt.ylabel("Price")
plt.xticks(rotation=45)
plt.show()

# إزالة علامة £ وتحويل السعر إلى float
# Check if the 'Price' column is of type object (string) before applying str.replace
if df['Price'].dtype == 'object':
    df["Price"] = df["Price"].str.replace("£", "").astype(float)
else:
    print("The 'Price' column is already numeric. No need for string replacement.")

# متوسط السعر
print("Average price:", df["Price"].mean())

# أعلى سعر وأقل سعر
print("Max price:", df["Price"].max())
print("Min price:", df["Price"].min())

#changing one two to 1 2
rating_map = {
    "One": 1,
    "Two": 2,
    "Three": 3,
    "Four": 4,
    "Five": 5
}
df["Rating"] = df["Rating"].map(rating_map)

# متوسط التقييم
print("Average rating:", df["Rating"].mean())

# توزيع التقييمات
print(df["Rating"].value_counts())

# كم كتاب متوفر؟
available_count = df[df["Availability"].str.contains("In stock")].shape[0]
print("Available books:", available_count)

# نسبة الكتب المتوفرة
print("Availability %:", available_count / len(df) * 100)

# أعلى 5 كتب بالسعر
top_price = df.sort_values(by="Price", ascending=False).head()
print(top_price[["Title", "Price", "Rating"]])
print()

# أعلى 5 كتب بالتقييم
top_rating = df.sort_values(by="Rating", ascending=False).head()
print(top_rating[["Title", "Price", "Rating"]])

#ما هو التقييم الاكثر شيوعا
most_common_rating = df["Rating"].mode()[0]
print("Most common rating:", most_common_rating)

#كم عدد الكتب بالتقييمات المنخفضة 1و2
low_rated_books = df[df["Rating"] <= 2]
print("Number of low-rated books:", len(low_rated_books))


#متوسط السعر لكل تقييم
avg_price_per_rating = df.groupby("Rating")["Price"].mean()
print("Average price per rating:")
print(avg_price_per_rating)

#كم عدد الكتب بالتقييمات المنخفضة 1و2
low_rated_books = df[df["Rating"] <= 2]
print("Number of low-rated books:", len(low_rated_books))


#4. هل الكتب الأعلى تقييمًا أغلى من الأقل تقييمًا؟
high_rated_avg_price = df[df["Rating"] >= 4]["Price"].mean()
low_rated_avg_price = df[df["Rating"] <= 2]["Price"].mean()

print("Avg price of high-rated books (4+):", high_rated_avg_price)
print("Avg price of low-rated books (1-2):", low_rated_avg_price)
#نسبة الكتب المناحة مقابل الغير متاحة
available = df["Availability"].str.contains("In stock").sum()
not_available = len(df) - available

print(f"Available: {available} ({available / len(df):.2%})")
print(f"Not Available: {not_available} ({not_available / len(df):.2%})")


#اغلى كتاب + اللينك بتاعه + سعره
most_expensive_book = df.sort_values(by="Price", ascending=False).iloc[0]
print("Most expensive book:")

# Access the values of 'Title' and 'Link' instead of printing the Series directly
print(f"Title: {most_expensive_book['Title']}")
print(f"Price : ${most_expensive_book['Price']}")
print(f"Link: {most_expensive_book['Link']}")

#books with rating=5
five_star_books = df[df["Rating"] == 5]
print(f"Number of 5-star books: {len(five_star_books)}")






#كتب سعرها قليل + تقييم عالي
average_price = df["Price"].astype(float).mean()
cheap_high_rated = df[(df["Price"] < average_price) & (df["Rating"] == 5)]
print("Cheap & highly rated books:")
print(cheap_high_rated[["Title", "Price", "Rating"]])


unique_titles = df["Title"].nunique()
print(f"Number of unique book titles: {unique_titles}")

# توزيع الأسعار إلى فئات (اغلب الكتب تتباع بكام)
bins = [0, 10, 20, 30, 40, 100]
labels = ["0-10£", "10-20£", "20-30£", "30-40£", "40£+"]
df["Price_Range"] = pd.cut(df["Price"], bins=bins, labels=labels)

price_range_counts = df["Price_Range"].value_counts().sort_index()
print("Books per price range:")
print(price_range_counts)

#اسماء الكتب الاكثر تكرارا
duplicate_titles = df["Title"].value_counts()
print("Most common titles (if any):")
print(duplicate_titles[duplicate_titles > 1])

#كلمات شائعة في عناوين الكتب
from collections import Counter
import re

# دمج كل العناوين في نص واحد وتحويلها لقائمة كلمات
words = re.findall(r'\b\w+\b', ' '.join(df["Title"]).lower())

# استبعاد الكلمات الشائعة عديمة المعنى
stopwords = {'the', 'and', 'of', 'in', 'to', 'a', 'for', 'on', 'with', 'at', 'by'}
filtered_words = [word for word in words if word not in stopwords]

common_words = Counter(filtered_words).most_common(10)
print("Most common words in book titles:")
print(common_words)


#ارخص كتب اللي بتقييم مرتفع
best_deals = df[df["Rating"] == 5].sort_values(by="Price").head(5)
print("Best deals (5-star books with lowest prices):")
print(best_deals[["Title", "Price", "Rating", "Link"]])

# تجميع الكتب حسب السعر والتقييم
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

# نجهز البيانات
features = df[["Price", "Numeric_Rating"]]
scaler = StandardScaler()
scaled_features = scaler.fit_transform(features)

# إنشاء نموذج KMeans
kmeans = KMeans(n_clusters=3, random_state=42)
df["Cluster"] = kmeans.fit_predict(scaled_features)

# عدد الكتب في كل مجموعة
print(df["Cluster"].value_counts())

# نظرة على كل مجموعة
for cluster_id in sorted(df["Cluster"].unique()):
    print(f"\nCluster {cluster_id}:")
    print(df[df["Cluster"] == cluster_id][["Title", "Price", "Numeric_Rating"]].head())


pivot_table = df.pivot_table(index="Rating", values="Price", aggfunc="mean")
sns.heatmap(pivot_table, annot=True, cmap="YlGnBu")
plt.title("Average Price by Rating (Heatmap)")
plt.show()
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
# Add any other imports from your notebook

st.title("Final Project Analysis")

# Load and process data
@st.cache_data
def load_data():
    # Load your dataset
    df = pd.read_csv(r"C:\fcds_projects\books_scraped.csv")  # or however you're loading it
    return df

df = load_data()

# Show raw data
if st.checkbox("Show Raw Data"):
    st.write(df)

# Add interactive analysis
st.subheader("Your Analysis")
# Example plot
fig, ax = plt.subplots()
df['Rating'].value_counts().plot(kind='bar', ax=ax)
st.pyplot(fig)

# Add more analysis/plots as needed
