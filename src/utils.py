import pandas as pd
import re
import nltk
#nltk.download('punkt')
#nltk.download('stopwords')
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer

def load_dataset(filepath):
    """
    Load the dataset from a CSV file.
    """
    df = pd.read_csv(filepath, encoding='utf8')
    return df

'''	
#PROS TO PARON SE COMMENT LOGW TOU FORMAT TOU CSV

def preprocess_data(df):
    """
    Preprocess the dataset by handling missing values and filtering relevant data.
    """
    # Drop rows with missing values
    df = df.dropna()

    # Filter rows by year range (2019-2021)
    df = df[(df['review_date'] >= '2019-01-01') & (df['review_date'] <= '2021-12-31')]

    # Convert review_date to datetime
    df['review_date'] = pd.to_datetime(df['review_date'])

    return df
'''	

def extract_year(review_date):
    """
    Extract the year from the review_date column.
    :param review_date: Date string in format "Month Year" (e.g., "November 2017").
    :return: Year as an integer.
    """
    return int(review_date.split()[-1])

def filter_data(data, query):
    """
    Apply multiple filters to the dataset based on the query parameters.
    :param data: Pandas DataFrame.
    :param query: Dictionary with query parameters.
    :return: Filtered DataFrame.
    """
    filtered_data = data

    # Apply year range filter
    if "year_range" in query:
        start_year, end_year = query["year_range"]
        filtered_data = filtered_data[(filtered_data["review_date_year"] >= start_year) &
                                      (filtered_data["review_date_year"] <= end_year)]
    #print("After Year Range Filter:")
    #print(filtered_data)

    # Apply price range filter
    if "price_range" in query:
        min_price, max_price = query["price_range"]
        filtered_data = filtered_data[(filtered_data["100g_USD"] >= min_price) &
                                      (filtered_data["100g_USD"] <= max_price)]
    #print("After Price Range Filter:")
    #print(filtered_data)

    # Apply rating filter
    if "min_rating" in query:
        filtered_data = filtered_data[filtered_data["rating"] > query["min_rating"]]
    #print("After Rating Filter:")
    #print(filtered_data)

    # Apply categorical filter (e.g., country)
    if "loc_country" in query:
        filtered_data = filtered_data[filtered_data["loc_country"] == query["loc_country"]]
    #print("After Country Filter:")
    #print(filtered_data)

    return filtered_data
def get_top_n_results(data, n, sort_by="rating"):
    """
    Get the top-N results from the filtered dataset based on a sorting criterion.
    :param data: Pandas DataFrame.
    :param n: Number of top results to return.
    :param sort_by: Column to sort by.
    :return: DataFrame with top-N results.
    """
    return data.sort_values(by=sort_by, ascending=False).head(n)

def preprocess_text(text):
    """
    Preprocess the input text: clean, tokenize, and remove stop words.
    :param text: Input string (review text).
    :return: List of processed tokens.
    """
    # Lowercase the text
    text = text.lower()

    # Remove punctuation
    text = re.sub(r'[^\w\s]', '', text)

    # Tokenize the text
    tokens = word_tokenize(text)

    # Remove stop words
    stop_words = set(stopwords.words('english'))
    tokens = [word for word in tokens if word not in stop_words]

    # Optional: Stemming (reduce words to root form)
    stemmer = PorterStemmer()
    tokens = [stemmer.stem(word) for word in tokens]

    return tokens

