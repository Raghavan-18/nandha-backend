import os
import random
import json
import logging
from datetime import datetime

from flask import Flask, request, jsonify
from flask_cors import CORS
from flask_limiter import Limiter
from flask_limiter.util import get_remote_address
from flask_talisman import Talisman
from marshmallow import Schema, fields, ValidationError
import requests
from dotenv import load_dotenv

# Load Environment Variables
load_dotenv()

# Setup Logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Initialize Flask App
app = Flask(__name__, static_folder='../frontend', static_url_path='')

is_production = os.environ.get('FLASK_ENV') == 'production'

# Global Configuration
app.config['MAX_CONTENT_LENGTH'] = 5 * 1024 * 1024  # 5 MB max payload

# Extensions (Only rate-limiter and Talisman security header management remains)
limiter = Limiter(
    get_remote_address,
    app=app,
    default_limits=["500 per day", "100 per hour"],
    storage_uri=os.environ.get("REDIS_URL", "memory://")
)

Talisman(app, content_security_policy=None, force_https=is_production)

frontend_url = os.environ.get('FRONTEND_URL', '*')
# Safely handle CORS
cors_origins = [frontend_url] if frontend_url != '*' else [r"https?://.*"]
CORS(app, supports_credentials=True, resources={r"/*": {"origins": cors_origins}})

# --- Validation Schemas ---
class RecommendSchema(Schema):
    soil_type = fields.Str(load_default="Black Soil")
    season = fields.Str(load_default="Rainy")
    water = fields.Str(load_default="Medium")
    health = fields.Str(load_default="Average")
    weather = fields.Str(load_default="Pleasant")
    location = fields.Str(load_default="")

class PredictSchema(Schema):
    crop = fields.Str(load_default="Soybean")
    area = fields.Float(load_default=1.0)

class LocationSchema(Schema):
    location = fields.Str(load_default="")

# --- Static Routes ---
@app.route('/')
def serve_index():
    return app.send_static_file('index.html')

@app.route('/<path:path>')
def serve_static(path):
    return app.send_static_file(path)

# --- Error Handlers ---
from werkzeug.exceptions import HTTPException

@app.errorhandler(HTTPException)
def handle_exception(e):
    # Return JSON instead of HTML for HTTP errors
    response = e.get_response()
    response.data = json.dumps({
        "status": "error",
        "message": e.description
    })
    response.content_type = "application/json"
    return response

@app.errorhandler(500)
def server_error(e):
    logger.error(f"Internal Server Error: {e}")
    return jsonify({"status": "error", "message": "Internal server error"}), 500

@app.errorhandler(429)
def ratelimit_handler(e):
    return jsonify({"status": "error", "message": f"Rate limit exceeded: {e.description}"}), 429

# --- Core API ---
def estimate_soil_from_location(loc):
    if not loc or len(loc.strip()) < 3:
        return "Unknown"
    loc = loc.lower().strip()
    invalid_keywords = ["mars", "moon", "jupiter", "saturn", "venus", "pluto", "planet", "galaxy", "space", "alien", "unknown", "test"]
    if any(k == loc or f" {k} " in f" {loc} " for k in invalid_keywords):
        return "Invalid"
    black_soil_regions = ["maharashtra", "mp", "gujarat", "coimbatore", "tiruppur"]
    if any(word in loc for word in black_soil_regions):
        return "Black Soil"
    return "Loamy Soil"

@app.route('/api/get-soil-info', methods=['POST'])
def get_soil_info():
    try:
        data = LocationSchema().load(request.get_json(silent=True) or {})
    except ValidationError as err:
        return jsonify({"status": "error", "message": "Invalid input"}), 400
        
    soil_type = estimate_soil_from_location(data.get('location', ''))
    if soil_type == "Invalid":
        return jsonify({"status": "error", "message": "Invalid location"}), 400
    return jsonify({"status": "success", "soil_type": soil_type})

def recommend_with_params(soil, season, water, health, weather, location=""):
    crop_profiles = {
        "Groundnut": {"soil": ["Red Soil", "Sandy Soil", "Loamy Soil"], "season": ["Rainy", "Summer"], "water": ["Low", "Medium"]},
        "Mustard": {"soil": ["Alluvial Soil", "Loamy Soil"], "season": ["Winter"], "water": ["Medium"]},
        "Soybean": {"soil": ["Black Soil", "Alluvial Soil"], "season": ["Rainy"], "water": ["High", "Medium"]},
    }
    scores = {crop: 0 for crop in crop_profiles}
    for crop, profile in crop_profiles.items():
        if soil in profile["soil"]: scores[crop] += 40
        if season in profile["season"]: scores[crop] += 35
        if water in profile["water"]: scores[crop] += 25
    
    sorted_crops = sorted(scores.items(), key=lambda x: x[1], reverse=True)
    top_crop = sorted_crops[0][0]
    
    # Fallback response if Groq fails or isn't configured
    fallback = {
        "status": "success", 
        "primary_crop": {
            "name": top_crop,
            "accuracy": "85",
            "reasoning": "Determined by optimal soil and season match.",
            "expected_yield": "1.5 Tons/Acre",
            "expert_tip": "Maintain good soil moisture.",
            "mandi_price": "₹ 6,500/q"
        },
        "alternatives": [{"name": sorted_crops[1][0], "accuracy": "80", "expected_yield": "1.2 Tons/Acre"}],
        "detected_params": {"soil": soil, "season": season, "weather": weather}
    }

    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        return jsonify(fallback)

    # Strictly allowlist inputs to prevent prompt injection
    safe_soils = ["Black Soil", "Red Soil", "Alluvial Soil", "Loamy Soil", "Sandy Soil", "Unknown"]
    safe_seasons = ["Rainy", "Summer", "Winter", "Spring", "Autumn"]
    clean_soil = soil if soil in safe_soils else "Standard Soil"
    clean_season = season if season in safe_seasons else "Standard Season"

    system_prompt = f"Explain why {top_crop} is best for {clean_soil} in {clean_season}. Return JSON: {{\"top_choice\":{{\"reason\":\"...\",\"advice\":\"...\",\"yield\":\"1.5\",\"price\":\"6000\"}},\"alternatives\":[{{\"crop\":\"{sorted_crops[1][0]}\",\"yield\":\"1.2\"}}]}}"
    
    try:
        response = requests.post(
            "https://api.groq.com/openai/v1/chat/completions",
            headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
            json={"model": "llama-3.1-8b-instant", "messages": [{"role": "user", "content": system_prompt}], "temperature": 0.2, "response_format": {"type": "json_object"}},
            timeout=10
        )
        response.raise_for_status()
        parsed = json.loads(response.json()["choices"][0]["message"]["content"])
        
        return jsonify({
            "status": "success",
            "primary_crop": {
                "name": top_crop,
                "accuracy": str(max(85, sorted_crops[0][1])),
                "reasoning": parsed["top_choice"]["reason"],
                "expected_yield": parsed["top_choice"]["yield"] + " Tons/Acre",
                "expert_tip": parsed["top_choice"]["advice"],
                "mandi_price": f"{parsed['top_choice']['price']}"
            },
            "alternatives": [{"name": alt["crop"], "accuracy": "80", "expected_yield": alt["yield"] + " Tons/Acre"} for alt in parsed.get("alternatives", [])],
            "detected_params": {"soil": soil, "season": season, "weather": weather}
        })
    except Exception as e:
        logger.error(f"LLM API Error: {e}")
        return jsonify(fallback)

@app.route('/api/recommend', methods=['POST'])
@limiter.limit("5 per minute")
def recommend():
    try:
        data = RecommendSchema().load(request.get_json(silent=True) or {})
    except ValidationError as err:
        return jsonify({"status": "error", "message": "Invalid input"}), 400
    return recommend_with_params(data['soil_type'], data['season'], data['water'], data['health'], data['weather'], data['location'])

@app.route('/api/smart-recommend', methods=['POST'])
@limiter.limit("5 per minute")
def smart_recommend():
    try:
        data = LocationSchema().load(request.get_json(silent=True) or {})
    except ValidationError as err:
        return jsonify({"status": "error", "message": "Invalid input"}), 400
        
    soil = estimate_soil_from_location(data.get('location', ''))
    if soil in ["Invalid", "Unknown"]:
        return jsonify({"status": "error", "message": "Valid location required"}), 400
    
    month = datetime.now().month
    season = "Rainy" if 7 <= month <= 9 else ("Winter" if 10 <= month <= 3 else "Summer")
    return recommend_with_params(soil, season, "Medium", "Average", "Pleasant", data.get('location', ''))

@app.route('/api/predict', methods=['POST'])
def predict():
    try:
        data = PredictSchema().load(request.get_json(silent=True) or {})
    except ValidationError as err:
        return jsonify({"status": "error", "message": "Invalid input"}), 400
        
    base_yield = 0.8 if "Groundnut" in data['crop'] else 0.5
    expected = base_yield * data['area'] * random.uniform(0.9, 1.1)
    return jsonify({"status": "success", "expected_yield": f"{expected:.2f} Tons", "profit_est": f"₹ {int(expected * 50000):,}", "harvest_time": "100-130 Days"})

@app.route('/api/detect', methods=['POST'])
@limiter.limit("5 per minute")
def detect():
    # Only allowed validation logic, no actual saving to disk to prevent RCE
    if 'image' not in request.files:
        return jsonify({"status": "error", "message": "No file uploaded"}), 400
    file = request.files['image']
    if not file.filename.lower().endswith(('.png', '.jpg', '.jpeg')):
        return jsonify({"status": "error", "message": "Invalid file type"}), 400
        
    DISEASE_DB = [{"name": "Leaf Spot", "desc": "Dark spots.", "remedy": ["Fungicide"]}]
    res = random.choice(DISEASE_DB)
    return jsonify({"status": "success", "detection": {"name": res['name'], "description": res['desc'], "remedy": res['remedy']}})

import hashlib

TN_DISTRICTS = [
    "Ariyalur", "Chengalpattu", "Chennai", "Coimbatore", "Cuddalore", "Dharmapuri", 
    "Dindigul", "Erode", "Kallakurichi", "Kanchipuram", "Kanniyakumari", "Karur", 
    "Krishnagiri", "Madurai", "Mayiladuthurai", "Nagapattinam", "Namakkal", "Nilgiris", 
    "Perambalur", "Pudukkottai", "Ramanathapuram", "Ranipet", "Salem", "Sivagangai", 
    "Tenkasi", "Thanjavur", "Theni", "Thoothukudi", "Tiruchirappalli", "Tirunelveli", 
    "Tirupathur", "Tiruppur", "Tiruvallur", "Tiruvannamalai", "Tiruvarur", "Vellore", 
    "Viluppuram", "Virudhunagar"
]

CROP_BASES = {
    "Groundnut": {"base": 6800, "volatility": 300},
    "Sesame": {"base": 12000, "volatility": 500},
    "Coconut": {"base": 3000, "volatility": 200},
    "Sunflower": {"base": 5500, "volatility": 400},
    "Castor": {"base": 4200, "volatility": 150},
    "Soybean": {"base": 4800, "volatility": 250},
    "Mustard": {"base": 6100, "volatility": 300}
}

def generate_stable_market_data():
    data = []
    for dist in TN_DISTRICTS:
        for crop, stats in CROP_BASES.items():
            seed_str = f"{dist}_{crop}_2026"
            hash_val = int(hashlib.md5(seed_str.encode()).hexdigest()[:8], 16)
            stable_variance = ((hash_val % 2000) / 1000.0) - 1.0 # -1.0 to 1.0
            
            price = int(stats["base"] + (stable_variance * stats["volatility"]))
            min_p = int(price * 0.95)
            max_p = int(price * 1.05)
            trend_val = stable_variance * 2.5 # -2.5% to 2.5%
            
            data.append({
                "name": crop,
                "mandi": f"{dist} Agri Market",
                "district": dist,
                "price": price,
                "min": min_p,
                "max": max_p,
                "trend": f"{'+' if trend_val >= 0 else ''}{trend_val:.1f}%",
                "status": "up" if trend_val >= 0 else "down"
            })
    return data

STABLE_MARKET_DATA = generate_stable_market_data()

@app.route('/api/market-prices', methods=['GET'])
def market_prices():
    return jsonify({"status": "success", "prices": STABLE_MARKET_DATA})

@app.route('/api/historical-trends', methods=['GET'])
def historical_trends():
    crop = request.args.get('crop', 'Groundnut')
    base = CROP_BASES.get(crop, {"base": 5000})["base"]
    # Generate stable pseudo-historical data
    data = [int(base * (1 + (i*0.02))) for i in range(-5, 1)]
    return jsonify({"status": "success", "labels": ["Oct", "Nov", "Dec", "Jan", "Feb", "Mar"], "data": data})

# Cache Prevention
@app.after_request
def add_header(response):
    response.headers['Cache-Control'] = 'no-store, no-cache, must-revalidate, max-age=0'
    response.headers['Pragma'] = 'no-cache'
    return response

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
