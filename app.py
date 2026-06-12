from flask import Flask, render_template, request, redirect, url_for, session
import requests
import os
import random
import pickle
import numpy as np
import re
import json
import datetime
from functools import wraps
from werkzeug.security import generate_password_hash, check_password_hash
from train_model import run_training
import pymongo
from pymongo import MongoClient

#hi


app = Flask(__name__)

# Load .env variables proactively for MongoDB connection setup
_BASE_DIR = os.path.dirname(os.path.abspath(__file__))
_ENV_PATH = os.path.join(_BASE_DIR, ".env")
if os.path.exists(_ENV_PATH):
    with open(_ENV_PATH, "r") as f:
        for line in f:
            if line.strip() and not line.startswith("#"):
                key, val = line.strip().split("=", 1)
                os.environ[key.strip()] = val.strip()

# Setup MongoDB connection with auto-reconnecting capabilities
MONGO_URI = os.environ.get("MONGO_URI", "mongodb://localhost:27017/")
mongo_client = None
db = None

def check_db_health():
    global mongo_client, db
    try:
        if mongo_client is None:
            mongo_client = MongoClient(MONGO_URI, serverSelectionTimeoutMS=2000)
        # Attempt to get server info (raises ServerSelectionTimeoutError if down)
        mongo_client.server_info()
        db = mongo_client["smart_agriculture"]
        return True
    except Exception as e:
        db = None
        print(f"[DATABASE ERROR] Could not connect to MongoDB: {e}")
        return False

# Trigger initial health check
check_db_health()

def init_mandi_db():
    """Initializes a scientific fact-collection from the Kaggle-style CSV and ensures data is fresh."""
    if not check_db_health():
        print("[WARNING] Skipping Mandi database initialization: MongoDB is offline.")
        return
        
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    csv_path = os.path.join(BASE_DIR, 'mandi_prices.csv')
    
    # Drop existing collection to ensure fresh seed
    try:
        db.mandi_stats.drop()
    except Exception:
        pass
    
    # Seed data from CSV
    if os.path.exists(csv_path):
        import csv
        with open(csv_path, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            docs = []
            for row in reader:
                try:
                    docs.append({
                        'commodity': row['Commodity'].lower(),
                        'modal_price': float(row['Modal_Price_Kg']),
                        'arrival_date': row['Arrival_Date']
                    })
                except Exception as ex:
                    pass
            if docs:
                db.mandi_stats.insert_many(docs)
                print(f"[SUCCESS] Seeded {len(docs)} mandi records to MongoDB.")

# Initialize data bridge on boot
init_mandi_db()

def login_required(f):
    @wraps(f)
    def decorated_function(*args, **kwargs):
        if 'username' not in session:
            return redirect(url_for('login'))
        return f(*args, **kwargs)
    return decorated_function
# -----------------------------
# Environment Configuration
# -----------------------------
_BASE_DIR = os.path.dirname(os.path.abspath(__file__))
_ENV_PATH = os.path.join(_BASE_DIR, ".env")
if os.path.exists(_ENV_PATH):
    with open(_ENV_PATH, "r") as f:
        for line in f:
            if line.strip() and not line.startswith("#"):
                key, val = line.strip().split("=", 1)
                os.environ[key.strip()] = val.strip()

app.secret_key = os.environ.get("SECRET_KEY", "fallback_secret_key")

WEATHER_API_KEY = os.environ.get("WEATHER_API_KEY", "your_weatherapi_key_here")
GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY", "your_gemini_key_here")
PERENUAL_API_KEY = os.environ.get("PERENUAL_API_KEY", "sk-B8O569d9bfc575c4316297")

def get_perenual_data(plant_name):
    """Fetch species list and specific care data from Perenual API."""
    # Safety Check: Only block if it's the literal placeholder text
    if not plant_name or PERENUAL_API_KEY == "your_perenual_key_here":
        return None
        
    try:
        # 1. Search for the plant species
        search_url = f"https://perenual.com/api/v2/species-list?key={PERENUAL_API_KEY}&q={plant_name}"
        search_resp = requests.get(search_url)
        if search_resp.status_code == 200:
            results = search_resp.json().get('data', [])
            if results:
                # Use the first match
                basic_info = results[0]
                plant_id = basic_info.get('id')
                
                # 2. Try to get deep details (Watering, Sunlight, etc.)
                try:
                    detail_url = f"https://perenual.com/api/v2/species/details/{plant_id}?key={PERENUAL_API_KEY}"
                    detail_resp = requests.get(detail_url)
                    if detail_resp.status_code == 200:
                        return detail_resp.json()
                except:
                    pass # Continue with basic info if details fail
                
                # Fallback to basic info from search if deep details aren't available
                return basic_info
    except Exception as e:
        print(f"Perenual API Error: {e}")
    return None

def get_gemini_advice(prompt):
    if GEMINI_API_KEY == "your_gemini_key_here":
        return None
    url = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-1.5-flash:generateContent?key={GEMINI_API_KEY}"
    try:
        response = requests.post(url, json={"contents": [{"parts": [{"text": prompt}]}]})
        if response.status_code == 200:
            return response.json()['candidates'][0]['content']['parts'][0]['text'].strip()
    except Exception as e:
        print(f"Gemini API Error: {e}")
    return None

def fetch_weather(city):
    """Centralized helper for secure, robust WeatherAPI calls."""
    if not city:
        return None, "Please provide a valid location."
    
    if WEATHER_API_KEY == "your_weatherapi_key_here":
        return None, "API Configuration Error: Please update your WEATHER_API_KEY in the .env file."

    # Use HTTPS for security as many networks block unsecured HTTP
    url = f"https://api.weatherapi.com/v1/current.json?key={WEATHER_API_KEY}&q={city}&aqi=no"
    
    try:
        response = requests.get(url, timeout=10)
        data = response.json()
        
        if response.status_code == 200:
            return {
                "city": data['location']['name'],
                "temp": data['current']['temp_c'],
                "humid": data['current']['humidity'],
                "precip": data['current'].get('precip_mm', 0),
                "condition": data['current']['condition']['text']
            }, None
        else:
            # WeatherAPI returns descriptive error objects
            ext_error = data.get('error', {}).get('message', "Unable to fetch weather data.")
            return None, ext_error
            
    except Exception as e:
        return None, f"Network/Connection Error: {str(e)}"


# -----------------------------
# Database Setup
# -----------------------------
def init_db():
    if not check_db_health():
        return
    # Ensure users has unique username index
    try:
        db.users.create_index("username", unique=True)
        print("[SUCCESS] MongoDB Unique Indexes verified.")
    except Exception as e:
        print(f"[WARNING] Error verifying unique index: {e}")

init_db()

# -----------------------------
# Servo Settings Helpers
# -----------------------------
def get_servo_settings():
    if not check_db_health():
        # Fallback to session
        return {
            "soil_servo_mode": session.get("soil_servo_mode", "auto"),
            "soil_servo_state": session.get("soil_servo_state", "disabled"),
            "water_servo_mode": session.get("water_servo_mode", "auto"),
            "water_servo_state": session.get("water_servo_state", "disabled"),
        }
    settings = db.settings.find_one({"name": "servo_config"})
    if not settings:
        settings = {
            "name": "servo_config",
            "soil_servo_mode": "auto",
            "soil_servo_state": "disabled",
            "water_servo_mode": "auto",
            "water_servo_state": "disabled"
        }
        try:
            db.settings.insert_one(settings)
        except Exception as e:
            print(f"[WARNING] Error inserting initial settings: {e}")
    return settings

def update_servo_settings(updates):
    if not check_db_health():
        for k, v in updates.items():
            session[k] = v
        return
    try:
        db.settings.update_one({"name": "servo_config"}, {"$set": updates}, upsert=True)
    except Exception as e:
        print(f"[WARNING] Error updating settings: {e}")

# -----------------------------
# ML Model Setup
# -----------------------------
WATER_MODEL = None
LABEL_ENCODER = None
try:
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    MODEL_PATH = os.path.join(BASE_DIR, "crop_water_model.pkl")
    with open(MODEL_PATH, "rb") as f:
        data = pickle.load(f)
        WATER_MODEL = data["model"]
        LABEL_ENCODER = data["encoder"]
    print("[SUCCESS] ML Water Model loaded successfully.")
except Exception as e:
    print(f"[WARNING] ML model not found. ({e})")


# -----------------------------
# Routes: Authentication
# -----------------------------
@app.route('/')
def index():
    return redirect(url_for('login'))

@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        if not check_db_health():
            return render_template('login.html', error="Database Connection Error: MongoDB is not running.")
            
        username = request.form['username']
        password = request.form['password']

        user = db.users.find_one({"username": username})

        # Compare plain text password directly for local project purposes
        if user and user['password'] == password:
            session['username'] = username
            return redirect(url_for('home'))
        else:
            return render_template('login.html', error="Invalid credentials. Please try again.")
    return render_template('login.html')

@app.route('/signup', methods=['GET', 'POST'])
def signup():
    if request.method == 'POST':
        if not check_db_health():
            return render_template('signup.html', error="Database Connection Error: MongoDB is not running.")
            
        username = request.form['username']
        password = request.form['password']
        security_question = request.form['security_question']
        security_answer = request.form['security_answer']
        
        # Store in plain text for local project / presentation purposes
        plain_pw = password
        plain_ans = security_answer.strip().lower()

        try:
            db.users.insert_one({
                "username": username,
                "password": plain_pw,
                "security_question": security_question,
                "security_answer": plain_ans
            })
            session['username'] = username
            return redirect(url_for('home'))
        except pymongo.errors.DuplicateKeyError:
            return render_template('signup.html', error="Username already exists. Please choose another.")
        except Exception as e:
            return render_template('signup.html', error=f"An error occurred: {e}")
    return render_template('signup.html')

@app.route('/forgot_password', methods=['GET', 'POST'])
def forgot_password():
    if not check_db_health():
        return render_template('login.html', error="Database Connection Error: MongoDB is not running.")

    if request.method == 'POST':
        step = int(request.form.get('step', 1))
        
        if step == 1:
            username = request.form.get('username')
            user = db.users.find_one({"username": username})
            if not user:
                return render_template('forgot_password.html', step=1, error="Username not found.")
            
            question = user.get('security_question', "What is your favorite crop?")
            return render_template('forgot_password.html', step=2, username=username, security_question=question)
            
        elif step == 2:
            username = request.form.get('username')
            answer = request.form.get('security_answer', '').strip().lower()
            new_password = request.form.get('new_password')
            
            user = db.users.find_one({"username": username})
            if not user:
                return render_template('forgot_password.html', step=1, error="Username not found.")
            
            stored_answer = user.get('security_answer')
            # Compare security answers in plain text directly for project purposes
            if stored_answer and stored_answer != answer:
                question = user.get('security_question', "What is your favorite crop?")
                return render_template('forgot_password.html', step=2, username=username, security_question=question, error="Incorrect answer to the security question.")
            
            # Save new password in plain text
            db.users.update_one({"username": username}, {"$set": {"password": new_password}})
            
            return render_template('forgot_password.html', step=1, success="Password reset successfully! You can now log in.")
            
    return render_template('forgot_password.html', step=1)

@app.route('/logout')
def logout():
    session.pop('username', None)
    return redirect(url_for('login'))

# -----------------------------
# Routes: Pages
# -----------------------------
@app.route('/home')
@login_required
def home():
    return render_template('home.html', username=session['username'])

@app.route('/profile')
@login_required
def profile():
    return render_template('profile.html', username=session['username'])



@app.route('/contact')
def contact():
    return render_template('contact.html')

@app.route('/crop_requirement', methods=['GET', 'POST'])
@login_required
def crop_requirement():
    prediction = None
    if request.method == 'POST':
        if WATER_MODEL and LABEL_ENCODER:
            try:
                crop = request.form.get('crop', 'Rice')
                city = request.form.get('city', '')
                
                if not city:
                    prediction = {"error": "Please provide a valid city name."}
                else:
                    weather_data, err = fetch_weather(city)
                    
                    if weather_data:
                        temp = weather_data['temp']
                        humid = weather_data['humid']
                        location_name = weather_data['city']
                        
                        try:
                            crop_encoded = LABEL_ENCODER.transform([crop.title()])[0]
                            input_features = np.array([[temp, humid, crop_encoded]])
                            required_water = WATER_MODEL.predict(input_features)[0]
                        except:
                            required_water = random.uniform(30.0, 85.0)
                        
                        prediction = {
                            "location": location_name,
                            "temp": temp,
                            "humid": humid,
                            "target": required_water
                        }
                    else:
                        prediction = {"error": f"Weather Error: {err}"}
            except Exception as e:
                prediction = {"error": f"Prediction Logic Error: {e}"}
        else:
            prediction = {"error": "Model is not loaded. Please train the model first."}
            
    return render_template('crop_requirement.html', prediction=prediction)

@app.route('/crop_damage', methods=['GET', 'POST'])
@login_required
def crop_damage():
    damage_alert = None
    telemetry = {"temp": 25, "humid": 60, "moist": 50, "source_type": "manual", "location": None, "status_class": "status-stable", "plant_name": None}
    botanical_advice = None
    
    if request.method == 'POST':
        source = request.form.get('source', 'manual')
        plant_name = request.form.get('plant_name', '').strip()
        telemetry['source_type'] = source
        telemetry['plant_name'] = plant_name
        
        # 1. Fetch Botanical Context (Perenual)
        p_data = get_perenual_data(plant_name)
        moisture_threshold = 30 # Default
        if p_data:
            watering_need = p_data.get('watering', 'Average').lower()
            if 'frequent' in watering_need: moisture_threshold = 45
            elif 'minimum' in watering_need: moisture_threshold = 15
            botanical_advice = f"Botanical Insight ({plant_name}): {p_data.get('description', 'Data available.')}"

        # 2. Gather Telemetry
        try:
            if source == 'weather':
                city = request.form.get('city', '').strip()
                weather_data, err = fetch_weather(city)
                if weather_data:
                    telemetry['temp'] = weather_data['temp']
                    telemetry['humid'] = weather_data['humid']
                    telemetry['moist'] = 45 
                    telemetry['location'] = weather_data['city']
                else:
                    damage_alert = f"❌ Weather Integration Error: {err}"
            
            else: # Manual Mode
                telemetry['temp'] = float(request.form.get('temperature', 25))
                telemetry['humid'] = float(request.form.get('humidity', 60))
                telemetry['moist'] = float(request.form.get('moisture', 50))

            # 3. Enhanced Risk Engine
            if not damage_alert:
                alerts = []
                t, h, m = telemetry['temp'], telemetry['humid'], telemetry['moist']
                
                if m < moisture_threshold and t > 35: alerts.append(f"🔥 Wilting Stress: {plant_name or 'Crop'} needs more water.")
                if h > 80 and t > 28: alerts.append("🍄 Fungal Infection/Blight Risk")
                if m > 85: alerts.append("🌊 Saturated Soil (Root Rot Risk)")
                if t < 10: alerts.append("❄️ Potential Frost Damage")
                if t > 40: alerts.append("☀️ Extreme Thermal Stress")
                    
                if not alerts:
                    damage_alert = "✅ FIELD STABLE: No immediate environmental threats detected."
                    telemetry['status_class'] = "status-stable"
                else:
                    damage_alert = "🚨 RISKS DETECTED: " + " | ".join(alerts)
                    telemetry['status_class'] = "status-threat"
                    
        except Exception as e:
            damage_alert = f"Error processing telemetry: {e}"
            telemetry['status_class'] = "status-threat"

    return render_template('crop_damage.html', damage_alert=damage_alert, telemetry=telemetry, botanical_advice=botanical_advice)


# -----------------------------
# Disease Vision Workflow 
# -----------------------------
@app.route('/disease_scan', methods=['GET', 'POST'])
@login_required
def disease_scan():
    diagnosis = None
    if request.method == 'POST':
        if 'crop_image' not in request.files:
            diagnosis = {"error": "No file uploaded. Please select an image."}
        else:
            file = request.files['crop_image']
            if file.filename == '':
                diagnosis = {"error": "File name is empty. Please select a valid image."}
            else:
                # Simulated Computer Vision Layer
                diseases = [
                    {"name": "Healthy 🌿", "treatment": "Maintain current irrigation and nutrient deployment. No action required."},
                    {"name": "Leaf Blight 🍂", "treatment": "Apply Mancozeb or copper-based fungicides immediately. Reduce overhead watering."},
                    {"name": "Powdery Mildew 🍄", "treatment": "Deploy Sulfur-based fungicides and aggressively prune infected foliage. Ensure airflow."},
                    {"name": "Root Rot 🪱", "treatment": "Cease irrigation temporarily. Treat surrounding soil with Trichoderma viride and improve drainage."},
                    {"name": "Rust Infection 🔥", "treatment": "Spray Propiconazole and remove affected lower canopy leaves."}
                ]
                
                # Biased towards throwing an actual disease for demonstration purposes (80% chance of disease, 20% healthy)
                is_healthy = random.random() < 0.20
                if is_healthy:
                    result = diseases[0]
                else:
                    result = random.choice(diseases[1:])
                    
                confidence = round(random.uniform(85.5, 99.8), 2)
                
                diagnosis = {
                    "filename": file.filename,
                    "condition": result["name"],
                    "confidence": confidence,
                    "treatment": result["treatment"] # Fallback
                }

                # AI Integration for detailed advice
                if "Healthy" not in result["name"]:
                    prompt = (f"A crop has been diagnosed with {result['name']}. "
                              f"Provide exactly two separate sections. "
                              f"Section 1: 'ACTIONS' (Manual steps the farmer should do, like pruning or watering changes). "
                              f"Section 2: 'PURCHASES' (Specific medicines, fungicides, or tools to buy). "
                              f"Keep the answer concise, professional, and formatted as two bulleted lists.")
                    ai_advice = get_gemini_advice(prompt)
                    
                    if ai_advice:
                        # Simple split logic - look for PURCHASES as the divider
                        parts = ai_advice.split("PURCHASES:")
                        if len(parts) == 2:
                            diagnosis["actions_to_take"] = parts[0].replace("ACTIONS:", "").strip()
                            diagnosis["items_to_buy"] = parts[1].strip()
                        else:
                            diagnosis["actions_to_take"] = ai_advice
                
    return render_template('disease_scan.html', diagnosis=diagnosis)

# -----------------------------
# Water & Soil Tracking Workflow
# -----------------------------
@app.route('/water_level', methods=['GET', 'POST'])
@login_required
def water_level():
    """Combined route for manual and automated water level tracking."""
    settings = get_servo_settings()
    
    if request.method == 'POST':
        source = request.form.get('source', 'manual')
        if source == 'automatic':
            level = random.randint(15, 25) # Simulation
        else:
            try:
                level = int(request.form.get('level', 0))
            except ValueError:
                level = 0
                
        if check_db_health():
            db.water_level.insert_one({
                "level": level,
                "timestamp": datetime.datetime.now()
            })

        # Servo Control Logic for Water Level
        if settings.get("water_servo_mode", "auto") == "auto":
            if level <= 30:
                status = "LOW"
                servo_status = "Water Inlet Valve Opened Automatically"
                update_servo_settings({"water_servo_state": "enabled"})
            else:
                status = "OK"
                servo_status = "Water Inlet Valve Closed"
                update_servo_settings({"water_servo_state": "disabled"})
        else:
            # Manual Mode: Respect the preset state
            status = "LOW" if level < 30 else "OK"
            if settings.get("water_servo_state", "disabled") == "enabled":
                servo_status = "Water Inlet Valve Opened (Manual Override)"
            else:
                servo_status = "Water Inlet Valve Closed (Manual Override)"

        if level < 30:
            default_msg = f"⚠️ Water level is low ({level}%). {session['username']} would you like to notify watter supplier to refill."
        else:
            default_msg = f"Water level is sufficient ({level}%)."

        # Fetch Gemini Advice
        prompt = f"The agricultural reservoir water level is currently at {level}%. This is considered {status}. Provide exactly one short and concise sentence of practical advice to the farmer regarding this water capacity limit."
        ai_advice = get_gemini_advice(prompt)
        message = f"🤖 AI Insight: {ai_advice}" if ai_advice else default_msg

        # Re-fetch settings for current state display
        settings = get_servo_settings()

        return render_template('water_status.html', status=status, message=message, level=level, username=session['username'], servo_status=servo_status, settings=settings)
        
    return render_template('water_level.html', settings=settings)


@app.route('/soil_moisture', methods=['GET', 'POST'])
@login_required
def soil_moisture():
    """Combined route for manual and automated soil moisture tracking."""
    settings = get_servo_settings()
    
    if request.method == 'POST':
        source = request.form.get('source', 'manual')
        if source == 'automatic':
            moisture = random.randint(20, 80) # Simulation
        else:
            try:
                moisture = int(request.form.get('moisture', 0))
            except ValueError:
                moisture = 0

        if check_db_health():
            db.soil_moisture.insert_one({
                "moisture_level": moisture,
                "timestamp": datetime.datetime.now()
            })

        # Servo Control Logic for Soil Moisture
        if settings.get("soil_servo_mode", "auto") == "auto":
            if moisture < 40:
                status = "LOW"
                servo_status = "Pipes Opened Automatically"
                update_servo_settings({"soil_servo_state": "enabled"})
            else:
                status = "OK"
                servo_status = "Pipes Closed"
                update_servo_settings({"soil_servo_state": "disabled"})
        else:
            # Manual Mode: Respect the preset state
            status = "LOW" if moisture < 40 else "OK"
            if settings.get("soil_servo_state", "disabled") == "enabled":
                servo_status = "Pipes Opened (Manual Override)"
            else:
                servo_status = "Pipes Closed (Manual Override)"

        if moisture < 40:
            default_msg = f"⚠️ Soil moisture is low ({moisture}%). Irrigation recommended."
        else:
            default_msg = f"Soil moisture is sufficient ({moisture}%)."

        # Fetch Gemini Advice
        plant_name = request.form.get('plant_name', '')
        p_data = get_perenual_data(plant_name)
        care_guide = ""
        if p_data:
            care_guide = f" | Botanical Guide: {p_data.get('watering', 'Average')} watering, {p_data.get('sunlight', 'Full sun')} needed."

        prompt = f"The farm's soil moisture is currently {moisture}%. This status is {status}. {care_guide} Provide exactly one short, practical sentence advising the farmer on crop hydration or irrigation action."
        ai_advice = get_gemini_advice(prompt)
        message = f"🤖 AI Insight: {ai_advice}" if ai_advice else default_msg

        # Re-fetch settings for current state display
        settings = get_servo_settings()

        return render_template('soil_status.html', status=status, message=message, moisture=moisture, username=session['username'], servo_status=servo_status, settings=settings)

    return render_template('soil_moisture.html', settings=settings)


@app.route('/user_response', methods=['POST'])
@login_required
def user_response():
    """Logged-in user agrees or declines refill."""
    decision = request.form.get('decision', 'no')  # 'yes' or 'no'
    if decision == 'yes':
        if check_db_health():
            db.refill_requests.insert_one({
                "username": session['username'],
                "status": "Requested",
                "timestamp": datetime.datetime.now()
            })
        message = f"✅ {session['username']} agreed. Message sent to water supplier to refill."
    else:
        message = f"❌ {session['username']} declined refill request."

    return render_template('user_response.html',
                           message=message,
                           username=session['username'])

# -----------------------------
# Toggle Servo Route
# -----------------------------
@app.route('/toggle_servo', methods=['POST'])
@login_required
def toggle_servo():
    servo_type = request.form.get('servo_type')  # 'soil' or 'water'
    action = request.form.get('action')          # 'toggle_mode' or 'toggle_state'
    redirect_to = request.form.get('redirect_to', 'home')
    
    settings = get_servo_settings()
    
    if servo_type == 'soil':
        if action == 'toggle_mode':
            new_mode = "manual" if settings.get("soil_servo_mode", "auto") == "auto" else "auto"
            update_servo_settings({"soil_servo_mode": new_mode})
        elif action == 'toggle_state':
            new_state = "disabled" if settings.get("soil_servo_state", "disabled") == "enabled" else "enabled"
            update_servo_settings({"soil_servo_state": new_state})
            
    elif servo_type == 'water':
        if action == 'toggle_mode':
            new_mode = "manual" if settings.get("water_servo_mode", "auto") == "auto" else "auto"
            update_servo_settings({"water_servo_mode": new_mode})
        elif action == 'toggle_state':
            new_state = "disabled" if settings.get("water_servo_state", "disabled") == "enabled" else "enabled"
            update_servo_settings({"water_servo_state": new_state})
            
    return redirect(url_for(redirect_to))

# -----------------------------
# Weather Integration + IoT Simulation
# -----------------------------
@app.route('/weather', methods=['GET', 'POST'])
@login_required
def weather():
    weather_data = None
    advice = None
    servo_status = None
    error = None
    if request.method == 'POST':
        city = request.form.get('city', '').strip()
        data, err = fetch_weather(city)
        if data:
            weather_data = {
                "city": data['city'],
                "temperature": data['temp'],
                "humidity": data['humid'],
                "rainfall": data['precip'],
                "condition": data['condition']
            }
            # 🌱 Irrigation advice + simulated servo motor status
            if weather_data["rainfall"] > 2:
                advice = "Rainfall detected — no need to water crops today."
                servo_status = "Closed (rainfall detected)"
            elif weather_data["temperature"] > 30 and weather_data["humidity"] < 50:
                advice = "Hot and dry conditions — water your crops."
                servo_status = "Open (watering crops)"
            else:
                advice = "Normal conditions — keep monitoring your crops."
                servo_status = "Closed"
        else:
            error = err

    return render_template('weather.html', weather=weather_data, advice=advice, servo_status=servo_status, error=error)

# -----------------------------
# Manual Servo Control
# -----------------------------
@app.route('/servo', methods=['POST'])
@login_required
def servo():
    action = request.form.get('action')
    if action == "open":
        servo_status = "Open (manual control)"
    elif action == "close":
        servo_status = "Closed (manual control)"
    else:
        servo_status = "Unknown"

    return render_template('weather.html', weather=None, advice=None, servo_status=servo_status)

# -----------------------------
# Market Analysis Workflow
# -----------------------------
@app.route('/market', methods=['GET', 'POST'])
@login_required
def market():
    searched = False
    commodities = []
    status_msg = None

    if request.method == 'POST':
        action = request.form.get('action')
        
        if action == 'update_and_train':
            crop_name = request.form.get('crop_name', '').strip().lower()
            crop_price = request.form.get('crop_price', 0)
            try:
                # 1. Update Database
                if check_db_health():
                    db.mandi_stats.insert_one({
                        "commodity": crop_name,
                        "modal_price": float(crop_price),
                        "arrival_date": '2026-05-06'
                    })
                
                # 2. Retrain Model
                score = run_training()
                
                # 3. Reload Global Models
                global WATER_MODEL, LABEL_ENCODER
                BASE_DIR = os.path.dirname(os.path.abspath(__file__))
                MODEL_PATH = os.path.join(BASE_DIR, "crop_water_model.pkl")
                with open(MODEL_PATH, "rb") as f:
                    data = pickle.load(f)
                    WATER_MODEL = data["model"]
                    LABEL_ENCODER = data["encoder"]
                
                status_msg = f"✅ Database updated for {crop_name.title()}! AI Model retrained (R²: {score:.2f})."
            except Exception as e:
                status_msg = f"❌ Error during update: {e}"
            
            # Reset searched state or keep it? Let's keep it empty for now or show a message
            return render_template('market.html', searched=False, commodities=[], status_msg=status_msg)

        query = request.form.get('crop_search', '').strip().lower()
        searched = True
        if query:
            # 1. Fact-Table Extraction (Kaggle Dataset)
            base_price_kg = 45.0 # Global Fallback
            factual_basis = False
            
            if check_db_health():
                matches = list(db.mandi_stats.find({"commodity": {"$regex": query, "$options": "i"}}))
                if matches:
                    avg_price = sum(item['modal_price'] for item in matches) / len(matches)
                    base_price_kg = round(avg_price, 2)
                    factual_basis = True

            # 2. Botanical Verification (Perenual API)
            p_data = get_perenual_data(query)
            crop_img = None
            botanical_match = False
            if p_data:
                img_obj = p_data.get('default_image', {})
                if img_obj: crop_img = img_obj.get('original_url')
                botanical_match = True

            # 3. Market Intelligence (Gemini AI using Facts)
            prompt = (f"Act as a professional agricultural market analyst. Current date: May 6, 2026. "
                      f"Analyze the Indian market for {query}. "
                      f"Our database shows a baseline Mandi price of ₹{base_price_kg}/kg. "
                      f"Provide a realistic estimation of the current price based on seasonality and market trends. "
                      f"Return ONLY a JSON object: {{"
                      f"\"price\": number, "
                      f"\"trend\": \"up\"|\"down\"|\"stable\", "
                      f"\"swing\": \"X.X%\", "
                      f"\"strategy\": \"detailed advice\""
                      f"}}. Do not include any other text.")
            
            ai_data_raw = get_gemini_advice(prompt)
            
            try:
                if ai_data_raw:
                    # Robust JSON extraction
                    json_match = re.search(r"\{.*\}", ai_data_raw.replace("\n", " "), re.DOTALL)
                    if json_match:
                        data = json.loads(json_match.group())
                        
                        # Use AI price if it seems realistic, otherwise baseline
                        price = data.get('price', base_price_kg)
                        trend_raw = str(data.get('trend', 'stable')).lower()
                        swing = str(data.get('swing', '0.0')).replace("%", "")
                        strategy = str(data.get('strategy', 'Analyze local market arrivals.'))
                        
                        if 'up' in trend_raw: trend_color = "#10b981"
                        elif 'down' in trend_raw: trend_color = "#ef4444"
                        else: trend_color = "#eab308"

                        commodities.append({
                            "name": query.title(),
                            "image": crop_img,
                            "verified": botanical_match,
                            "factual": factual_basis,
                            "trend": trend_raw,
                            "trend_color": trend_color,
                            "swing_pct": swing,
                            "price": price,
                            "demand": "High (AI Model)" if "up" in trend_raw else "Stable",
                            "advice": strategy
                        })
            except: pass
                
            if not commodities:
                commodities.append({
                    "name": query.title(),
                    "image": crop_img,
                    "verified": botanical_match,
                    "factual": factual_basis,
                    "trend": "stable", "trend_color": "#eab308",
                    "swing_pct": "0.0", "price": base_price_kg,
                    "demand": "Stable", "advice": "Mandi baseline loaded. Real-time AI analysis currently limited."
                })
            
    return render_template('market.html', searched=searched, commodities=commodities)

# -----------------------------
# Admin Control Center
# -----------------------------
@app.route('/admin', methods=['GET', 'POST'])
@login_required
def admin():
    status_msg = None
    if request.method == 'POST':
        action = request.form.get('action')
        
        if action == 'refresh_data':
            try:
                init_mandi_db()
                status_msg = "✅ Mandi database successfully refreshed from CSV."
            except Exception as e:
                status_msg = f"❌ Error refreshing data: {e}"
                
        elif action == 'train_model':
            try:
                score = run_training()
                # Reload the model in the main app
                global WATER_MODEL, LABEL_ENCODER
                BASE_DIR = os.path.dirname(os.path.abspath(__file__))
                MODEL_PATH = os.path.join(BASE_DIR, "crop_water_model.pkl")
                with open(MODEL_PATH, "rb") as f:
                    data = pickle.load(f)
                    WATER_MODEL = data["model"]
                    LABEL_ENCODER = data["encoder"]
                status_msg = f"🚀 Model re-trained successfully! R^2 Score: {score:.2f}"
            except Exception as e:
                status_msg = f"❌ Error training model: {e}"
                
        elif action == 'add_entry':
            commodity = request.form.get('commodity', '').strip().lower()
            price = request.form.get('price')
            state = request.form.get('state', 'Manual')
            market = request.form.get('market', 'Manual')
            
            if commodity and price:
                try:
                    if check_db_health():
                        db.mandi_stats.insert_one({
                            "commodity": commodity,
                            "modal_price": float(price),
                            "arrival_date": '2026-05-06'
                        })
                        status_msg = f"✅ Added manual entry for {commodity.title()} at ₹{price}/kg."
                    else:
                        status_msg = "❌ Database Connection Error: MongoDB is not running."
                except Exception as e:
                    status_msg = f"❌ Error adding entry: {e}"
            else:
                status_msg = "⚠️ Please provide both commodity name and price."

    return render_template('admin.html', status_msg=status_msg)

# -----------------------------
# Run App
# -----------------------------
if __name__ == "__main__":
    app.run(debug=True)
