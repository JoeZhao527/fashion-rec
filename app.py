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

path = r'C:/Users/Administrator/Desktop/h-and-m-personalized-fashion-recommendations'
recommender_system = RecommenderSystem(
    article_path= path + "/dataset/articles.csv",
    customer_path= path + "/dataset/customers.csv",
    train_path= path + "/dataset/split/fold_0/train.csv",
    test_path= path + "/dataset/split/fold_0/test.csv",
    img_cluster_path="./resources/img_cluster_2000.csv",
    dev_mode=True,
    cache_dir="./cache/fold_0"
)

# recommender_system = RecommenderSystem(
#     article_path="./dataset/articles.csv",
#     customer_path="./dataset/customers.csv",
#     train_path="./dataset/split/fold_0/train.csv",
#     test_path="./dataset/split/fold_0/test.csv",
#     img_cluster_path="./resources/img_cluster_2000.csv",
#     dev_mode=True,
#     cache_dir="./cache/fold_0"
# )

@app.route('/api/items', methods=['GET'])
def get_items():
    customer_id = request.args.get('customer_id')
    if not customer_id:
        return jsonify({'message': 'Customer ID is required'}), 400
    
    try:
        # Get the list of recommended article IDs
        item_ids = recommender_system.recommend(customer_id)
        print(f"Recommended item IDs: {item_ids}")
        recommended_items = recommender_system.get_items_by_ids(item_ids)
    except Exception as e:
        return jsonify({'error': str(e)}), 500

    return jsonify(recommended_items)

@app.route('/images/<path:filename>', methods=['GET'])
def send_image(filename):
    return send_from_directory(app.static_folder + "/images", filename)

@app.route('/api/purchases', methods=['GET'])
def get_purchases():
    customer_id = request.args.get('customer_id')
    if not customer_id:
        return jsonify({'error': 'Customer ID is required'}), 400

    # Get the list of article IDs purchased by the customer
    customer_purchases = recommender_system.get_user_purchased(customer_id)
    print(customer_purchases)
    if not customer_purchases:
        return jsonify({"error": "No purchases found for the given customer ID"}), 404

    # Get detailed information about the articles
    detailed_items = recommender_system.get_items_by_ids(customer_purchases)

    return jsonify(detailed_items)

if __name__ == '__main__':
    # Example usage
    # item_ids = [108775015, 108775044]  # list of item IDs
    # items = get_items_by_ids(item_ids)
    # print(items)
    app.run(debug=True)
