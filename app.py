from flask import Flask, request, jsonify, send_from_directory
from flask_restful import Api, Resource
from flask_swagger_ui import get_swaggerui_blueprint
from flask_cors import CORS
import pandas as pd
import os
import ast

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

# Load item data from CSV
items_df = pd.read_csv(r'C:\Users\Administrator\Desktop\h-and-m-personalized-fashion-recommendations\dataset\articles.csv')
purchase_df = pd.read_csv(r'C:\Users\Administrator\Desktop\h-and-m-personalized-fashion-recommendations\dataset\actual_purchases.csv')

def get_items_by_ids(item_ids):
    # Convert article_id to integer to handle scientific notation
    items_df['article_id'] = items_df['article_id'].apply(lambda x: int(float(x)))

    # Filter the DataFrame for the given item IDs
    filtered_items = items_df[items_df['article_id'].isin(item_ids)]

    # Define a helper function to update rows
    def update_item(row):
        row['liked'] = True
        row['image_url'] = get_image_path(row['article_id'])
        return row

    # Apply the helper function to each row in the filtered DataFrame
    updated_items = filtered_items.apply(update_item, axis=1)

    # Convert the updated DataFrame to a list of dictionaries
    item_details = updated_items.to_dict(orient='records')
    return item_details

def get_image_path(item_id):
    item_id_str = str(item_id)
    folder_number = '0' + item_id_str[:2]  # Ensure this logic matches your folder structure
    item_id_str = '0' + item_id_str
    image_url = f'http://localhost:5000/images/{folder_number}/{item_id_str}.jpg'
    print(image_url)
    return image_url

@app.route('/api/items', methods=['GET'])
def get_items():
    customer_id = request.args.get('customer_id')
    if not customer_id:
        return jsonify({'message': 'Customer ID is required'}), 400
    items = items_df.to_dict(orient='records')
    customer_purchases = purchase_df[purchase_df['customer_id'] == customer_id]['article_id']
    liked_items = eval(customer_purchases.iloc[0]) if not customer_purchases.empty else []

    # TODO make the real recommend in here
    
    recommended_items = items[:10]
    for item in recommended_items:
        item['liked'] = item['article_id'] in liked_items
        item['image_url'] = get_image_path(item['article_id'])
    return jsonify(recommended_items)

@app.route('/images/<path:filename>')
def send_image(filename):
    return send_from_directory(app.static_folder + "/images", filename)

@app.route('/api/purchases', methods=['GET'])
def get_purchases():
    customer_id = request.args.get('customer_id')
    if not customer_id:
        return jsonify({'error': 'Customer ID is required'}), 400

    # Filter the purchase DataFrame for the given customer ID
    customer_purchases = purchase_df[purchase_df['customer_id'] == customer_id]

    if customer_purchases.empty:
        return jsonify({'message': 'No purchases found for this customer'}), 404

    # Parse the 'article_id' column which contains string representations of lists
    try:
        item_ids = [ast.literal_eval(ids) for ids in customer_purchases['article_id']]
    except ValueError:
        return jsonify({'error': 'Invalid data format for article IDs'}), 500

    # Flatten the list if it contains sublists
    flat_item_ids = [item for sublist in item_ids for item in sublist]

    # Get detailed item info using the get_items_by_ids function
    detailed_items = get_items_by_ids(flat_item_ids)

    return jsonify(detailed_items)

if __name__ == '__main__':
    # Example usage
    # item_ids = [108775015, 108775044]  # list of item IDs
    # items = get_items_by_ids(item_ids)
    # print(items)
    app.run(debug=True)
