from flask import Flask, request, jsonify, send_from_directory
from flask_restful import Api, Resource
from flask_swagger_ui import get_swaggerui_blueprint
from flask_cors import CORS
import pandas as pd
import os
import ast
from typing import List, Dict
from recommender.recommender import RecommenderSystem

app = Flask(__name__, static_folder='static')
CORS(app)  # This will enable CORS for all routes
api = Api(app)

# Swagger UI setup
SWAGGER_URL = '/swagger'
API_URL = '/static/swagger.json'
swaggerui_blueprint = get_swaggerui_blueprint(
    SWAGGER_URL,
    API_URL,
    config={
        'app_name': "Item Recommendation API"
    }
)
app.register_blueprint(swaggerui_blueprint, url_prefix=SWAGGER_URL)

recommender_system = RecommenderSystem(
    article_path="./dataset/articles.csv",
    customer_path="./dataset/customers.csv",
    train_path="./dataset/split/fold_0/train.csv",
    test_path="./dataset/split/fold_0/test.csv",
    img_cluster_path="./resources/img_cluster_2000.csv",
    dev_mode=True,
    cache_dir="./cache/fold_0"
)

def get_items_by_ids(items: List[dict]):
    for item in items:
        item['liked'] = True
        item['image_url'] = get_image_path(item['article_id'])

    return items

def get_image_path(item_id):
    item_id_str = str(item_id)
    folder_number = '0' + item_id_str[:2]  # Ensure this logic matches your folder structure
    item_id_str = '0' + item_id_str
    image_url = f'http://localhost:5000/images/{folder_number}/{item_id_str}.jpg'
    return image_url

@app.route('/api/items', methods=['GET'])
def get_items():
    customer_id = request.args.get('customer_id')
    if not customer_id:
        return jsonify({'message': 'Customer ID is required'}), 400
    
    item_ids = recommender_system.recommend(customer_id)
    items = recommender_system.get_items_by_ids(item_ids)
    recommended_items = get_items_by_ids(items)

    return jsonify(recommended_items)

@app.route('/images/<path:filename>', methods=['GET'])
def send_image(filename):
    return send_from_directory(app.static_folder + "/images", filename)

@app.route('/api/purchases', methods=['GET'])
def get_purchases():
    customer_id = request.args.get('customer_id')
    if not customer_id:
        return jsonify({'error': 'Customer ID is required'}), 400

    # Filter the purchase DataFrame for the given customer ID
    customer_purchases = recommender_system.get_user_purchased(customer_id)

    if customer_purchases.empty:
        return jsonify({'message': 'No purchases found for this customer'}), 404

    # Parse the 'article_id' column which contains string representations of lists
    try:
        item_ids = [ast.literal_eval(ids) for ids in customer_purchases['article_id']]
    except ValueError:
        return jsonify({'error': 'Invalid data format for article IDs'}), 500

    # Flatten the list if it contains sublists
    flat_item_ids = [item for sublist in item_ids for item in sublist]

    detailed_items = recommender_system.get_items_by_ids(flat_item_ids)
    detailed_items = get_items_by_ids(detailed_items)

    return jsonify(detailed_items)

if __name__ == '__main__':
    # Example usage
    # item_ids = [108775015, 108775044]  # list of item IDs
    # items = get_items_by_ids(item_ids)
    # print(items)
    app.run(debug=True)
