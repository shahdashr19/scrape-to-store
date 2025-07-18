import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from pathlib import Path

# --- Page Configuration ---
st.set_page_config(page_title="Book Analysis Dashboard", layout="wide")

# --- Inject Custom CSS for Larger UI ---
st.markdown("""
    <style>
    /* Larger tabs */
    .stTabs [data-baseweb="tab"] {
        font-size: 30px;
        padding: 30px 30px;
        font-weight: bold;
    }

    .stTabs [aria-selected="true"] {
        background-color: #ffe4e1;
        color: black;
    }

    /* Larger sidebar text */
    section[data-testid="stSidebar"] .css-ng1t4o {
        font-size: 18px;
    }

    /* Larger headings */
    h1{
        font-size: 50px !important;
    }
    h2{
        font-size: 30px !important;
    }
    h3 {
        font-size: 20px !important;
    }

    /* Larger general layout */
    .block-container {
        padding: 2rem 4rem;
        font-size: 30px;
    }

    /* Larger tables */
    .stDataFrame, .stTable {
        font-size: 16px;
    }
    </style>
""", unsafe_allow_html=True)

# --- Load Data Function ---
@st.cache_data
def load_data():
    file_path = "books_scraped.csv"
    if Path(file_path).exists():
        df = pd.read_csv(file_path)
        st.write("Columns in the dataset:", df.columns.tolist())
        st.write("First five rows:", df.head())

        if df['Price'].dtype == 'object':
            df['Price'] = df['Price'].str.replace("\u00a3", "").astype(float)

        rating_map = {"One": 1, "Two": 2, "Three": 3, "Four": 4, "Five": 5}
        df['Numeric_Rating'] = df['Rating'].map(rating_map)

        if df.isnull().any().any():
            st.warning("Missing values found. Dropping rows with missing values.")
            df = df.dropna()
        return df
    else:
        st.error("books_scraped.csv not found! Please ensure the file is in the same directory.")
        return None

# --- Load Data ---
df = load_data()

# --- Title and Description ---
st.title("\U0001F4DA Book Analysis Dashboard")
st.image("Bookstore-806x440.jpg", width=1000)
st.markdown("Explore the scraped book dataset from books.toscrape.com with interactive visualizations and filters.")

if df is not None:
    # --- Sidebar Filters ---
    st.sidebar.header("Filters")
    rating_filter = st.sidebar.multiselect("Select Rating", options=sorted(df['Numeric_Rating'].unique()), default=sorted(df['Numeric_Rating'].unique()))
    availability_filter = st.sidebar.selectbox("Availability", options=["All", "In stock", "Out of stock"])
    price_range = st.sidebar.slider("Price Range (£)", float(df['Price'].min()), float(df['Price'].max()), (float(df['Price'].min()), float(df['Price'].max())))

    # --- Apply Filters ---
    filtered_df = df[df['Numeric_Rating'].isin(rating_filter)]
    if availability_filter != "All":
        filtered_df = filtered_df[filtered_df['Availability'].str.contains("In stock" if availability_filter == "In stock" else "Out of stock", case=False)]
    filtered_df = filtered_df[(filtered_df['Price'] >= price_range[0]) & (filtered_df['Price'] <= price_range[1])]

    # --- Tabs Layout ---
    if filtered_df.empty:
        st.warning("No books match the selected filters. Please adjust the filters.")
    else:
        tab1, tab2, tab3 = st.tabs(["\U0001F4CA Data Overview", "\U0001F4C8 Visualizations", "\U0001F50D Insights"])

        # --- Data Overview ---
        with tab1:
            st.header("Data Overview")
            st.write(f"Total Books: {len(filtered_df)}")
            st.write(f"Average Price: £{filtered_df['Price'].mean():.2f}")
            st.write(f"Average Rating: {filtered_df['Numeric_Rating'].mean():.2f}")
            st.dataframe(filtered_df[['Title', 'Price', 'Numeric_Rating', 'Availability', 'Link']], use_container_width=True)

        # --- Visualizations ---
        with tab2:
            import seaborn as sns
            import matplotlib.pyplot as plt

            st.header("Visualizations")

            # Price Distribution
            st.subheader("Price Distribution")
            fig, ax = plt.subplots(figsize=(6, 4))
            sns.histplot(data=filtered_df, x='Price', bins=20, kde=True, color='#ff69b4', ax=ax)
            ax.set_title("Distribution of Book Prices")
            ax.set_xlabel("Price (£)")
            ax.set_ylabel("Count")
            st.pyplot(fig)
            plt.close(fig)

            # Price vs Rating
            st.subheader("Price vs Rating")
            fig, ax = plt.subplots(figsize=(6, 4))
            sns.scatterplot(x='Numeric_Rating', y='Price', data=filtered_df, color='#6495ed', ax=ax)
            ax.set_title("Price vs. Rating")
            ax.set_xlabel('Rating')
            ax.set_ylabel('Price')
            st.pyplot(fig)
            plt.close(fig)

            # Books by Rating
            st.subheader("Books by Rating")
            fig, ax = plt.subplots(figsize=(6, 4))
            sns.countplot(data=filtered_df, x='Numeric_Rating', ax=ax, color='#98ff98')
            ax.set_title("Number of Books by Rating")
            ax.set_xlabel("Rating")
            ax.set_ylabel("Count")
            st.pyplot(fig)
            plt.close(fig)

            # Price vs Availability (Violin Plot)
            st.subheader("Price vs Availability (Violin Plot)")
            fig, ax = plt.subplots(figsize=(6, 4))
            sns.violinplot(x='Availability', y='Price', data=filtered_df, ax=ax, palette='Greens')
            ax.set_title("Does Price Affect Availability?")
            ax.set_xlabel("Availability")
            ax.set_ylabel("Price")
            st.pyplot(fig)
            plt.close(fig)

        # --- Insights ---
        with tab3:
            st.header("Insights")

            # Top 5 Expensive Books
            st.subheader("Top 5 Most Expensive Books")
            top_expensive = filtered_df.sort_values(by='Price', ascending=False).head(5)
            st.table(top_expensive[['Title', 'Price', 'Numeric_Rating', 'Availability']])

            # Best Deals (5-Star + Cheap)
            st.subheader("Best Deals (5-Star Books with Low Prices)")
            average_price = filtered_df['Price'].mean()
            best_deals = filtered_df[(filtered_df['Numeric_Rating'] == 5) & (filtered_df['Price'] < average_price)].sort_values(by='Price').head(5)
            st.table(best_deals[['Title', 'Price', 'Numeric_Rating', 'Availability']])

            # Most Common Words in Titles
            st.subheader("Most Common Words in Book Titles")
            from collections import Counter
            import re
            words = re.findall(r'\b\w+\b', ' '.join(filtered_df['Title']).lower())
            stopwords = {'the', 'and', 'of', 'in', 'to', 'a', 'for', 'on', 'with', 'at', 'by'}
            filtered_words = [word for word in words if word not in stopwords]
            common_words = Counter(filtered_words).most_common(10)
            common_words_df = pd.DataFrame(common_words, columns=['Word', 'Count'])
            st.table(common_words_df)

# --- Footer ---
st.markdown("---")
st.markdown("Built with Streamlit | Data scraped from books.toscrape.com")