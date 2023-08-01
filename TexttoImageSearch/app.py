import csv
import string
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import os

from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from nltk.stem import WordNetLemmatizer
import nltk

from flask import Flask, request, jsonify, render_template

nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')

app = Flask(__name__)

# Read image index file and return  image information
def build_image_index(csv_file):
    image_index = {}
    ids = []
    with open(csv_file, 'r') as file:
        csv_reader = csv.reader(file)
        for row in csv_reader:
            image_id, path, label = row
            image_index[image_id] = {'path': path, 'label': label}
            ids.append(image_id)
    return image_index, ids

# Function to preprocess text
def preprocess_text(text):

    # Step 1: Normalize to lowercase
    text = text.lower()

    # Step 2: Tokenize
    tokens = word_tokenize(text)

    # Step 3: Remove punctuation from each word
    punc = str.maketrans('', '', string.punctuation)
    non_punc = [w.translate(punc) for w in tokens]

    # Step 4: Remove non-alphabetic tokens
    words = [word for word in non_punc if word.isalpha()]

    # Step 5: Remove stop words from tokens
    stop_words = set(stopwords.words('english'))

    words = [w for w in words if not w in stop_words]

    # Step 6: Stemming of tokens
    stemmer = PorterStemmer()
    stemmed_text = [stemmer.stem(word) for word in words]

    return ' '.join(map(str, stemmed_text))
    
def preprocess_image(image_path, target_size=(256, 256)):
    # Read the image from the given path
    image = cv2.imread(image_path)

    # Resize the image to the target size
    image = cv2.resize(image, target_size)

    # Perform any other necessary preprocessing on the image here (e.g., normalization)

    return image

#Get image vector representation (surrogate)
def extract_image_surrogate(image):
    # Use a feature extractor to convert the image into a vector representation (surrogate)
    # For example, Use a Bag of Visual Words approach or deep learning-based features.
    # The output should be a numerical vector that represents the image.
    return image

#Get text vector representation (surrogate)
def extract_text_surrogate(text):
    vectorizer = TfidfVectorizer(tokenizer=lambda x: x, lowercase=False)
    text_surrogate = vectorizer.fit_transform([text])
    return text_surrogate

#return top 5 ranked images from the list  
def search_images_by_text(text_query, image_index, image_ids):
  # Preprocess the text query
  processed_text_query = preprocess_text(text_query)

  # Extract the text labels for all images
  image_labels = [image_index[image_id]['label'] for image_id in image_ids]

  # Initialize the TfidfVectorizer
  vectorizer = TfidfVectorizer(tokenizer=preprocess_text)

  # Fit and transform the image labels to get their text surrogates
  text_surrogates = vectorizer.fit_transform(image_labels)

  # Transform the text query to get its text surrogate
  text_query_surrogate = vectorizer.transform([text_query])

  similarities = {}
  
  for i, image_id in enumerate(image_ids):
    # Calculate the similarity between the query's text surrogate and the image's text surrogate
    similarity = cosine_similarity(text_query_surrogate, text_surrogates[i])
    similarities[image_id] = similarity[0][0]
  
  sorted_similarities = sorted(similarities.items(), key=lambda x: x[1], reverse=True)
  top_similarities = sorted_similarities[:5]
    
  return top_similarities

# Replace this with your actual image search and retrieval logic
def perform_image_search(query):

    csv_file = './reverse_image_search/reverse_image_search.csv'
    image_index, image_ids = build_image_index(csv_file)
    search_results = search_images_by_text(query, image_index, image_ids)
    results = []
    for image_id, similarity in search_results:
        result = {'url' : image_index[image_id]['path'], 'title': image_index[image_id]['path'].split('/')[-1].split('.')[0] }
        results.append(result)

    # Implement your image search and retrieval logic here
    # Return a list of image URLs or other relevant information
    # based on the given query
    #results = [
    #    {'url': 'https://example.com/image1.jpg', 'title': 'Image 1'},
    #    {'url': 'https://example.com/image2.jpg', 'title': 'Image 2'},
        # Add more search results here
    #]
    return results
@app.route('/')
def index():
    return (render_template('index.html'))
    
@app.route('/search', methods=['GET'])
def search_images():
    print("current path" + os.getcwd())
    query = request.args.get('q', '')
    results = perform_image_search(query)
    return jsonify(results)

if __name__ == '__main__':
    app.run(debug=True)
