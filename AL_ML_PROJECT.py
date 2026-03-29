import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Step 1: Load data
df = pd.read_csv("movies.csv")
df.columns = df.columns.str.strip()
df['Genre'] = df['Genre'].fillna('')

# Step 2: Convert Genre text → numbers
cv = CountVectorizer()
vectors = cv.fit_transform(df['Genre'])

# Step 3: Create similarity matrix
similarity = cosine_similarity(vectors)

# Step 4: Function to recommend movies
def recommend(movie_name):
    # Find index of movie
    index = df[df['Movie Name'] == movie_name].index[0]
    
    # Get similarity scores
    distances = similarity[index]
    
    # Sort movies based on similarity
    movies_list = sorted(list(enumerate(distances)), reverse=True, key=lambda x: x[1])
    
    # Print top 5 similar movies
    for i in movies_list[1:10]:
        print(df.iloc[i[0]]['Movie Name'])

# Step 5: Test
recommend("John Wick")


movie = input("Enter movie name: ")

print("Showing results of simiar movies :-")


recommend(movie)

print("Hope so you find a Grate movie...")
print("Thanks for using our program . Have a grate day.")