import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from pathlib import Path

# Set page configuration
st.set_page_config(page_title="Book Analysis Dashboard", layout="wide")

# Function to load data
@st.cache_data
def load_data():
    file_path = "books_scraped.csv"
    if Path(file_path).exists():
        df = pd.read_csv(file_path)
        # Debug: Check data
        st.write("Columns in the dataset:", df.columns.tolist())
        st.write("First five rows:", df.head())
        # Clean Price column if needed
        if df['Price'].dtype == 'object':
            df['Price'] = df['Price'].str.replace("£", "").astype(float)
        # Map ratings to numeric
        rating_map = {"One": 1, "Two": 2, "Three": 3, "Four": 4, "Five": 5}
        df['Numeric_Rating'] = df['Rating'].map(rating_map)
        # Check for missing values
        if df['Price'].isnull().any() or df['Numeric_Rating'].isnull().any():
            st.warning("Some values in 'Price' or 'Numeric_Rating' are missing. Please check the data.")
        return df
    else:
        st.error("books_scraped.csv not found! Please ensure the file is in the same directory.")
        return None

# Load data
df = load_data()

# Title of the app
st.title("📚 Book Analysis Dashboard")
st.image("Bookstore-806x440.jpg",width=6000,)
st.markdown("Explore the scraped book dataset from books.toscrape.com with interactive visualizations and filters.")

if df is not None:
    # Sidebar for filters
    st.sidebar.header("Filters")
    rating_filter = st.sidebar.multiselect(
        "Select Rating", options=sorted(df['Numeric_Rating'].unique()), default=sorted(df['Numeric_Rating'].unique())
    )
    availability_filter = st.sidebar.selectbox("Availability", options=["All", "In stock", "Out of stock"])
    price_range = st.sidebar.slider("Price Range (£)", float(df['Price'].min()), float(df['Price'].max()), (float(df['Price'].min()), float(df['Price'].max())))

    # Filter data
    filtered_df = df[df['Numeric_Rating'].isin(rating_filter)]
    if availability_filter != "All":
        filtered_df = filtered_df[filtered_df['Availability'].str.contains("In stock" if availability_filter == "In stock" else "Out of stock", case=False)]
    filtered_df = filtered_df[(filtered_df['Price'] >= price_range[0]) & (filtered_df['Price'] <= price_range[1])]

    # Check if filtered data is empty
    if filtered_df.empty:
        st.warning("No books match the selected filters. Please adjust the filters.")
    else:
        # Tabs for different sections
        tab1, tab2, tab3 = st.tabs(["📊 Data Overview", "📈 Visualizations", "🔍 Insights"])

        with tab1:
            st.header("Data Overview")
            st.write(f"Total Books: {len(filtered_df)}")
            st.write(f"Average Price: £{filtered_df['Price'].mean():.2f}")
            st.write(f"Average Rating: {filtered_df['Numeric_Rating'].mean():.2f}")
            st.dataframe(filtered_df[['Title', 'Price', 'Numeric_Rating', 'Availability', 'Link']], use_container_width=True)

        with tab2:
            st.header("Visualizations")

            # Price Distribution
            st.subheader("Price Distribution")
            try:
                import seaborn as sns
                import matplotlib.pyplot as plt

                # Create a figure
                fig, ax = plt.subplots()
        
                # Plot histogram with KDE using Seaborn
                sns.histplot(data=filtered_df, x='Price', bins=20, kde=True, color='lightsteelblue', ax=ax)
        
                # Add title and labels
                ax.set_title("Distribution of Book Prices")
                ax.set_xlabel("Price (£)")
                ax.set_ylabel("Count")
        
                # Display the plot in Streamlit
                st.pyplot(fig)
        
                # Clear the figure to avoid overlapping in future plots
                plt.close(fig)
            except Exception as e:
                st.error(f"Error in Price Distribution plot: {str(e)}")

            # Price vs Rating
            st.subheader("Price vs Rating")
            try:
                fig, ax = plt.subplots(figsize=(8, 6))
                sns.scatterplot(x='Numeric_Rating', y='Price', data=filtered_df, palette='colorblind', ax=ax)
                ax.set_title("Price vs. Rating")
                ax.set_xlabel('Rating')
                ax.set_ylabel('Price')
                # Display the plot in Streamlit
                st.pyplot(fig)
                plt.close(fig)
            except Exception as e:
                st.error(f"Error in Price vs Rating plot: {str(e)}")

            # Books by Rating
            st.subheader("Books by Rating")
            try:
                fig, ax = plt.subplots(figsize=(8, 6))
    
                # Plot bar chart using Seaborn
                sns.countplot(data=filtered_df, x='Numeric_Rating', ax=ax, color='navajowhite')
    
                 # Add title and labels
                ax.set_title("Number of Books by Rating")
                ax.set_xlabel("Rating")
                ax.set_ylabel("Count")
    
             # Display the plot in Streamlit
                st.pyplot(fig)
                plt.close(fig)
            except Exception as e:
                st.error(f"Error in Books by Rating plot: {str(e)}")

            
      # 6. Price vs Availability (Violin Plot)
            st.subheader("Price vs Availability (Violin Plot)")
            try:
                fig, ax = plt.subplots(figsize=(8, 6))
                sns.violinplot(x='Availability', y='Price', data=filtered_df, ax=ax, palette='rocket')
                ax.set_title("Does Price Affect Availability?")
                ax.set_xlabel("Availability")    
                ax.set_ylabel("Price ")
                st.pyplot(fig)
                plt.close(fig)
            except Exception as e:
                st.error(f"Error in Price vs Availability plot: {str(e)}")

        with tab3:
            st.header("Insights")

            # Top 5 Expensive Books
            st.subheader("Top 5 Most Expensive Books")
            try:
                top_expensive = filtered_df.sort_values(by='Price', ascending=False).head(5)
                st.table(top_expensive[['Title', 'Price', 'Numeric_Rating', 'Availability']])
            except Exception as e:
                st.error(f"Error in Top 5 Expensive Books: {str(e)}")

            # Best Deals (Cheap + High Rating)
            st.subheader("Best Deals (5-Star Books with Low Prices)")
            try:
                average_price = filtered_df['Price'].mean()
                best_deals = filtered_df[(filtered_df['Numeric_Rating'] == 5) & (filtered_df['Price'] < average_price)].sort_values(by='Price').head(5)
                st.table(best_deals[['Title', 'Price', 'Numeric_Rating', 'Availability']])
            except Exception as e:
                st.error(f"Error in Best Deals: {str(e)}")

            # Most Common Words in Titles
            st.subheader("Most Common Words in Book Titles")
            try:
                from collections import Counter
                import re
                words = re.findall(r'\b\w+\b', ' '.join(filtered_df['Title']).lower())
                stopwords = {'the', 'and', 'of', 'in', 'to', 'a', 'for', 'on', 'with', 'at', 'by'}
                filtered_words = [word for word in words if word not in stopwords]
                common_words = Counter(filtered_words).most_common(10)
                common_words_df = pd.DataFrame(common_words, columns=['Word', 'Count'])
                st.table(common_words_df)
            except Exception as e:
                st.error(f"Error in Most Common Words analysis: {str(e)}")

# Footer
st.markdown("---")
st.markdown("Built with Streamlit | Data scraped from books.toscrape.com")