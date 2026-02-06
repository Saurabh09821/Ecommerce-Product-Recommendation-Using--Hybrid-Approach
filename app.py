from flask import Flask, request, render_template , session , redirect , url_for
import pandas as pd
import random
from flask_sqlalchemy import SQLAlchemy
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import requests

app = Flask(__name__)

# load files===========================================================================================================
trending_products = pd.read_csv("models/trending_products.csv")
train_data = pd.read_csv("models/clean_data.csv")

# database configuration---------------------------------------
app.secret_key = "alskdjfwoeieikjjkurlskdjfslkdjf"
app.config['SQLALCHEMY_DATABASE_URI'] = "mysql://root:@localhost/ecomm"
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
db = SQLAlchemy(app)


# Define your model class for the 'signup' table
class Signup(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(100), nullable=False)
    email = db.Column(db.String(100), nullable=False)
    password = db.Column(db.String(100), nullable=False)

# Define your model class for the 'signup' table
class Signin(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(100), nullable=False)
    password = db.Column(db.String(100), nullable=False)


# Recommendations functions============================================================================================
# Function to truncate product name
def truncate(text, length):
    if len(text) > length:
        return text[:length] + "..."
    else:
        return text




#content based function
def content_based_recommendations(train_data, item_name, top_n=10):
    # Check if the item name exists in the training data
    if item_name not in train_data['Name'].values:
        print(f"Item '{item_name}' not found in the training data.")
        return pd.DataFrame()

    # Create a TF-IDF vectorizer for item descriptions
    tfidf_vectorizer = TfidfVectorizer(stop_words='english')

    # Apply TF-IDF vectorization to item descriptions
    tfidf_matrix_content = tfidf_vectorizer.fit_transform(train_data['Tags'])

    # Calculate cosine similarity between items based on descriptions
    cosine_similarities_content = cosine_similarity(tfidf_matrix_content, tfidf_matrix_content)

    # Find the index of the item
    item_index = train_data[train_data['Name'] == item_name].index[0]

    # Get the cosine similarity scores for the item
    similar_items = list(enumerate(cosine_similarities_content[item_index]))

    # Sort similar items by similarity score in descending order
    similar_items = sorted(similar_items, key=lambda x: x[1], reverse=True)

    # Get the top N most similar items (excluding the item itself)
    top_similar_items = similar_items[1:top_n+1]

    # Get the indices of the top similar items
    recommended_item_indices = [x[0] for x in top_similar_items]

    # Get the details of the top similar items
    recommended_items_details = train_data.iloc[recommended_item_indices][['Name', 'ReviewCount', 'Brand', 'ImageURL', 'Rating']]

    return recommended_items_details




def collaborative_filtering_recommendations(train_data, target_user_id, top_n=10):
    # Create the user-item matrix
    user_item_matrix = train_data.pivot_table(index='ID', columns='ProdID', values='Rating', aggfunc='mean').fillna(0)

    # Calculate the user similarity matrix using cosine similarity
    user_similarity = cosine_similarity(user_item_matrix)

    # Find the index of the target user in the matrix
    target_user_index = user_item_matrix.index.get_loc(target_user_id)

    # Get the similarity scores for the target user
    user_similarities = user_similarity[target_user_index]

    # Sort the users by similarity in descending order (excluding the target user)
    similar_users_indices = user_similarities.argsort()[::-1][1:]

    # Generate recommendations based on similar users
    recommended_items = []

    for user_index in similar_users_indices:
        # Get items rated by the similar user but not by the target user
        rated_by_similar_user = user_item_matrix.iloc[user_index]
        not_rated_by_target_user = (rated_by_similar_user == 0) & (user_item_matrix.iloc[target_user_index] == 0)

        # Extract the item IDs of recommended items
        recommended_items.extend(user_item_matrix.columns[not_rated_by_target_user][:top_n])

    # Get the details of recommended items
    recommended_items_details_collab = train_data[train_data['ProdID'].isin(recommended_items)][['Name', 'ReviewCount', 'Brand', 'ImageURL', 'Rating']]

    return recommended_items_details_collab.head(10)



# Hybrid Recommendations (Combine Content-Based and Collaborative Filtering)
def hybrid_recommendations(train_data,target_user_id, item_name, top_n=10):

    # Check if the item name exists in the training data
    if item_name not in train_data['Name'].values:
        print(f"Item '{item_name}' not found in the training data.")
        return pd.DataFrame()
    # Get content-based recommendations
    content_based_rec = content_based_recommendations(train_data,item_name, top_n)

    # Get collaborative filtering recommendations
    collaborative_filtering_rec = collaborative_filtering_recommendations(train_data,target_user_id, top_n)
    
    # Merge and deduplicate the recommendations
    hybrid_rec = pd.concat([content_based_rec, collaborative_filtering_rec]).drop_duplicates()
    # hybrid_rec['valid_image_url'] = hybrid_rec['ImageURL'].apply(get_valid_image_url)
    
    return hybrid_rec.head(top_n)
# routes===============================================================================
# List of predefined image URLs
random_image_urls = [
    "static/img_1.png",
    "static/img_2.png",
    "static/img_3.png",
    "static/img_4.png",
    "static/img_5.png",
    "static/img_6.png",
    "static/img_7.png",
    "static/img_8.png",
]


@app.route("/")
def index():
    # Create a list of random image URLs for each product
    random_product_image_urls = [random.choice(random_image_urls) for _ in range(len(trending_products))]
    price = [40, 50, 60, 70, 100, 122, 106, 50, 30, 50]
    return render_template('index.html',trending_products=trending_products.head(8),truncate = truncate,
                           random_product_image_urls=random_product_image_urls,
                           random_price = random.choice(price))

@app.route("/main")
def main():
    return render_template('main.html', hybrid_rec=None, truncate=truncate, is_empty=True, message=None)
    

# routes
@app.route("/index")
def indexredirect():
    # Create a list of random image URLs for each product
    random_product_image_urls = [random.choice(random_image_urls) for _ in range(len(trending_products))]
    price = [40, 50, 60, 70, 100, 122, 106, 50, 30, 50]
    return render_template('index.html', trending_products=trending_products.head(8), truncate=truncate,
                           random_product_image_urls=random_product_image_urls,
                           random_price=random.choice(price))

@app.route("/signup", methods=['POST','GET'])
def signup():
    if request.method=='POST':
        username = request.form['username']
        email = request.form['email']
        password = request.form['password']

        new_signup = Signup(username=username, email=email, password=password)
        db.session.add(new_signup)
        db.session.commit()

        # Create a list of random image URLs for each product
        random_product_image_urls = [random.choice(random_image_urls) for _ in range(len(trending_products))]
        price = [40, 50, 60, 70, 100, 122, 106, 50, 30, 50]
        return render_template('index.html', trending_products=trending_products.head(8), truncate=truncate,
                               random_product_image_urls=random_product_image_urls, random_price=random.choice(price),
                               signup_message='User signed up successfully!'
                               )

# Route for signin page
@app.route('/signin', methods=['POST', 'GET'])
def signin():
    if request.method == 'POST':
        username = request.form['signinUsername']
        password = request.form['signinPassword']
        user = Signin.query.filter_by(username=username, password=password).first()
        if user:
            user_id = user.id  #  Get user ID
            session['user_id'] = user_id  # Store user ID in session
        # new_signup = Signin(username=username,password=password)
        # db.session.add(new_signup)
        # db.session.commit()

        # Create a list of random image URLs for each product
            random_product_image_urls = [random.choice(random_image_urls) for _ in range(len(trending_products))]
            price = [40, 50, 60, 70, 100, 122, 106, 50, 30, 50]
            return render_template('index.html', trending_products=trending_products.head(8), truncate=truncate,
                               random_product_image_urls=random_product_image_urls, random_price=random.choice(price),
                               signup_message='User signed in successfully!'
                               )
        else:
            # Invalid credentials
            return render_template('signin.html', error='Invalid username or password')
    return render_template('signin.html', error=None)
    x



@app.route("/recommendations", methods=['POST', 'GET'])
def recommendations():
    if request.method == 'POST':
        prod = request.form.get('prod')
        nbr = int(request.form.get('nbr'))
        user_id = session.get('user_id')
        print(user_id)
        print(nbr)
        hybrid_rec=hybrid_recommendations(train_data,user_id, prod, top_n=nbr)

    
        # Check if the DataFrame is empty
        is_empty = hybrid_rec.empty
        print(f"Type of hybrid_rec: {type(hybrid_rec)}")
        print(f"Content of hybrid_rec: {hybrid_rec}")

        if hybrid_rec is None or hybrid_rec.empty :
            message = "No recommendations for this product."
            return render_template('main.html', message=message, hybrid_rec=None, is_empty=is_empty )
        else:
            # Create a list of random image URLs for each recommended product
            # random_product_image_urls = [random.choice(random_image_urls) for _ in range(len(hybrid_rec))]
            # Get actual image URLs with validity check
            valid_image_urls = hybrid_rec['ImageURL'].tolist()
            print(valid_image_urls)

            # Generate random prices for each recommended product
            random_prices = [random.choice([40, 50, 60, 70, 100, 122, 106, 50, 30, 50]) for _ in range(len(hybrid_rec))]

            return render_template('main.html', hybrid_rec=hybrid_rec, truncate=truncate,
                                   random_product_image_urls=valid_image_urls,
                                   random_prices=random_prices, is_empty=is_empty)


if __name__=='__main__':
    app.run(debug=True)