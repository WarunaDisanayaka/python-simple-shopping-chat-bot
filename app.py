from flask import Flask, render_template, request, jsonify
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
import re

app = Flask(__name__)

tokenizer = AutoTokenizer.from_pretrained("microsoft/DialoGPT-medium")
model = AutoModelForCausalLM.from_pretrained("microsoft/DialoGPT-medium")

added_products = {}


@app.route("/")
def index():
    return render_template('chat.html')

@app.route("/get", methods=["POST"])
def chat():
    msg = request.form["msg"]
    response = get_Chat_response(msg)
    return jsonify(response)

def get_Chat_response(text):
    intent = recognize_intent(text)
    if intent == "search_product":
        response = search_product(text)
    elif intent == "add_to_cart":
        response = add_to_cart(text)
    elif intent == "checkout":
        response = checkout(text)
    elif intent == "check_cart":
        response = check_cart(text)
    elif intent == "check_balance":
        response = check_balance()
    elif intent == "check_order_status":
        response = check_order_status()
    else:
        response = generate_response(text)
    return response

def recognize_intent(text):
    if re.search(r'\bbuy\b|\bsearch\b|\bfind\b|\bwant\b|\bwhat do you sell\b', text, re.IGNORECASE):
        return "search_product"
    elif re.search(r'\badd\b|\bcart\b', text, re.IGNORECASE):
        return "add_to_cart"
    elif re.search(r'\bcheckout\b', text, re.IGNORECASE):
        return "checkout"
    elif re.search(r'\bmycart\b', text, re.IGNORECASE):
        return "check_cart"
    elif re.search(r'\bbalance\b', text, re.IGNORECASE):
        return "check_balance"
    elif re.search(r'\border status\b|\bstatus\b', text, re.IGNORECASE):
        return "check_order_status"
    else:
        return "chat"


def search_product(text):
    products = call_shopping_backend_search(text)
    if products:
        return f"Found products: {', '.join(products)}"
    else:
        return "No products found."
    
def add_to_cart(text):
    product_id = extract_product_id(text)
    product_details = get_product_details(product_id)
    if call_shopping_backend_add_to_cart(product_id, product_details):
        write_to_cart_file(product_details)  # Write product details to file
        return "Product added to cart"
    else:
        return "Failed to add product to cart"

def write_to_cart_file(product_details):
    with open('cart.txt', 'a') as file:
        file.write(product_details + '\n')



def get_product_details(product_id):
    # This function can be expanded to fetch details from a database or another source
    return f"{product_id}"

def checkout(text):
    success = call_shopping_backend_checkout()
    return "Checkout successful" if success else "Checkout failed"


# Update the backend search function to read from a text file
def call_shopping_backend_search(query):
    try:
        with open('products.txt', 'r') as file:
            products = file.read().splitlines()
            print("Products from file:", products)  # Debug information
    except FileNotFoundError:
        return ["Error: products.txt not found"]

    # Debug information
    filtered_products = [product for product in products if query.lower() in product.lower()]
    print("Filtered products:", filtered_products)
    
    return products

def extract_product_id(text):
    # Read product IDs from file
    try:
        with open('products.txt', 'r') as file:
            product_ids = {}
            for line in file:
                name, product_id = line.strip().split(':')
                product_ids[name.strip()] = product_id.strip()
    except FileNotFoundError:
        print("Error: products.txt not found")
        return "unknown_product_id"

    # Search for the product name in the file
    for product_name, product_id in product_ids.items():
        if product_name.lower() in text.lower():
            return product_id
    
    return "unknown_product_id"


def call_shopping_backend_add_to_cart(product_id, product_details):
    if product_id != "unknown_product_id":
        added_products[product_id] = product_details
        return True
    else:
        return False
    
    
def call_shopping_backend_checkout():
    # Assuming added_products contains the details of products in the cart
    order_details = "\n".join([f"{product}: {details}" for product, details in added_products.items()])
    order_status = "processing"  # Default status
    
    # Save order details and status to a text file
    with open('order.txt', 'w') as file:
        file.write(order_details + '\n')
        file.write(f"Status: {order_status}\n")

    # Clear added_products after checkout
    added_products.clear()
    
    return True




def check_cart(text):
    cart_items = call_shopping_backend_cart(text)
    if cart_items:
        return f"Items in cart: {', '.join(cart_items)}"
    else:
        print ("Cart is empty")

def call_shopping_backend_cart(text):
    try:
        with open('cart.txt', 'r') as file:
            cart_items = file.read().splitlines()
            print("Cart items from file:", cart_items)  # Debug information
    except FileNotFoundError:
        print ("fdsfdsty")


    return cart_items

def check_balance():
    try:
        total_balance = 0
        with open('cart.txt', 'r') as file:
            for line in file:
                item, price = line.strip().split(' ')
                total_balance += int(price.replace('$', ''))
        return f"Total balance in cart: ${total_balance}"
    except FileNotFoundError:
        return "Cart is empty"

    

def get_total_balance():
    try:
        total_balance = 0
        with open('cart.txt', 'r') as file:
            for line in file:
                parts = line.strip().split()
                if len(parts) == 2 and parts[1].startswith('$'):
                    try:
                        balance = float(parts[1][1:])
                        total_balance += balance
                    except ValueError:
                        print("Invalid format:", line)
        return total_balance
    except FileNotFoundError:
        print("Cart file not found")
        return None


def check_order_status():
    try:
        with open('order.txt', 'r') as file:
            for line in file:
                if line.startswith("Status:"):
                    return line.strip().split(":")[1].strip()
        return "No orders found"
    except FileNotFoundError:
        return "No orders found"


def generate_response(text):
    new_user_input_ids = tokenizer.encode(str(text) + tokenizer.eos_token, return_tensors='pt')
    chat_history_ids = None
    if chat_history_ids is not None:
        bot_input_ids = torch.cat([chat_history_ids, new_user_input_ids], dim=-1)
    else:
        bot_input_ids = new_user_input_ids

    chat_history_ids = model.generate(bot_input_ids, max_length=1000, pad_token_id=tokenizer.eos_token_id)
    return tokenizer.decode(chat_history_ids[:, bot_input_ids.shape[-1]:][0], skip_special_tokens=True)

if __name__ == '__main__':
    app.run()
