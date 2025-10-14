# --- Libraries ---
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud
from textblob import TextBlob

# --- Step 1: Load Dataset ---
csv_path = "Reviews.csv"  # make sure the file is in the same folder
df = pd.read_csv(csv_path)

print("✅ Dataset Loaded Successfully!")
print("Shape:", df.shape)
display(df.head())

# --- Step 2: Keep useful columns ---
df = df[['ProductId', 'UserId', 'ProfileName', 'HelpfulnessNumerator',
         'HelpfulnessDenominator', 'Score', 'Time', 'Summary', 'Text']]

# --- Step 3: Handle missing values ---
df['Text'] = df['Text'].fillna('')
df = df.dropna(subset=['Score'])

# --- Step 4: Sentiment Analysis ---
def get_sentiment(text):
    blob = TextBlob(str(text))
    polarity = blob.sentiment.polarity
    if polarity > 0:
        return 'Positive'
    elif polarity < 0:
        return 'Negative'
    else:
        return 'Neutral'

df['Sentiment'] = df['Text'].apply(get_sentiment)

# --- Step 5: Basic Insights ---
print("\nSentiment counts:")
print(df['Sentiment'].value_counts())

# --- Step 6: Visualizations ---

# Rating Distribution
plt.figure(figsize=(8,5))
sns.countplot(x='Score', data=df, palette='coolwarm')
plt.title('Rating Distribution')
plt.show()

# Sentiment Distribution
plt.figure(figsize=(8,5))
sns.countplot(x='Sentiment', data=df, palette='viridis')
plt.title('Sentiment Distribution')
plt.show()

# WordCloud for reviews
all_text = ' '.join(df['Text'].astype(str))
wordcloud = WordCloud(width=800, height=400, background_color='white').generate(all_text)
plt.figure(figsize=(15,7))
plt.imshow(wordcloud, interpolation='bilinear')
plt.axis('off')
plt.title('Most Common Words in Reviews')
plt.show()
