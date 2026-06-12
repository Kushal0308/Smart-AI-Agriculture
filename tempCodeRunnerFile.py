from flask import Flask, render_template, request, redirect, url_for, session
import sqlite3
import requests
import os
from werkzeug.security import generate_password_hash, check_password_hash

app = Flask(__name__)
app.secret_key = os.getenv("SECRET_KEY", "fallback_secret_key")

# -----------------------------
# Database Setup
# -----------------------------
def init_db():
    conn = sqlite3.connect("users.db")
    c = conn.cursor()
    # Users table
    c.execute('''CREATE TABLE IF NOT EXISTS users (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    username TEXT UNIQUE NOT NULL,
                    password TEXT NOT NULL
                )''')
    # Water monitoring tables
    c.execute('''CREATE TABLE IF NOT EXISTS water_level (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    level INTEGER NOT NULL,
                    timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
                )''')
    c.execute('''CREATE TABLE IF NOT EXISTS refill_requests (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    username TEXT NOT NULL,
                    status TEXT NOT NULL,
                    timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
                )''')
    conn.commit()
    conn.close()

init_db()

# -----------------------------
# Routes: Authentication
# -----------------------------
@app.route('/')
def index():
    return redirect(url_for('login'))

@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']

        conn = sqlite3.connect("users.db")
        c = conn.cursor()
        c.execute("SELECT * FROM users WHERE username=?", (username,))
        user = c.fetchone()
        conn.close()

        if user and check_password_hash(user[2], password):
            session['username'] = username
            return redirect(url_for('home'))
        else:
            return render_template('login.html', error="Invalid credentials. Please try again.")
    return render_template('login.html')

@app.route('/signup', methods=['GET', 'POST'])
def signup():
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']
        hashed_pw = generate_password_hash(password)

        conn = sqlite3.connect("users.db")
        c = conn.cursor()
        try:
            c.execute("INSERT INTO users (username, password) VALUES (?, ?)", (username, hashed_pw))
            conn.commit()
            conn.close()
            session['username'] = username
            return redirect(url_for('home'))
        except sqlite3.IntegrityError:
            conn.close()
            return render_template('signup.html', error="Username already exists. Please choose another.")
    return render_template('signup.html')

@app.route('/logout')
def logout():
    session.pop('username', None)
    return redirect(url_for('login'))

# -----------------------------
# Routes: Pages
# -----------------------------
@app.route('/home')
def home():
    if 'username' in session:
        return render_template('home.html', username=session['username'])
    return redirect(url_for('login'))

@app.route('/about')
def about():
    return render_template('about.html')

@app.route('/how')
def how():
    return render_template('how.html')

@app.route('/training')
def training():
    return render_template('training.html')

@app.route('/future')
def future():
    return render_template('future.html')

@app.route('/contact')
def contact():
    return render_template('contact.html')

@app.route('/crop_requirement')
def crop_requirement():
    return render_template('crop_requirement.html')

@app.route('/crop_damage')
def crop_damage():
    return render_template('crop_damage.html')

# -----------------------------
# Water Level Monitoring Workflow
# -----------------------------
@app.route('/update_water', methods=['POST'])
def update_water():
    """User inputs water level manually."""
    if 'username' not in session:
        return redirect(url_for('login'))

    level = int(request.form['level'])
    conn = sqlite3.connect("users.db")
    c = conn.cursor()
    c.execute("INSERT INTO water_level (level) VALUES (?)", (level,))
    conn.commit()
    conn.close()

    if level < 30:
        status = "LOW"
        message = f"⚠️ Water level is low ({level}%). Notify {session['username']} to refill."
    else:
        status = "OK"
        message = f"Water level is sufficient ({level}%)."

    return render_template('water_status.html',
                           status=status,
                           message=message,
                           level=level,
                           username=session['username'])


@app.route('/user_response', methods=['POST'])
def user_response():
    """Logged-in user agrees or declines refill."""
    if 'username' not in session:
        return redirect(url_for('login'))

    decision = request.form['decision']  # 'yes' or 'no'
    if decision == 'yes':
        conn = sqlite3.connect("users.db")
        c = conn.cursor()
        c.execute("INSERT INTO refill_requests (username, status) VALUES (?, ?)",
                  (session['username'], "Requested"))
        conn.commit()
        conn.close()
        message = f"✅ {session['username']} agreed. Message sent to water supplier to refill."
    else:
        message = f"❌ {session['username']} declined refill request."

    return render_template('user_response.html',
                           message=message,
                           username=session['username'])

# -----------------------------
# Weather Integration + IoT Simulation
# -----------------------------
API_KEY = os.getenv("WEATHER_API_KEY", "your_weatherapi_key_here")

@app.route('/weather', methods=['GET', 'POST'])
def weather():
    weather_data = None
    advice = None
    servo_status = None
    if request.method == 'POST':
        city = request.form['city']
        url = f"http://api.weatherapi.com/v1/current.json?key={API_KEY}&q={city}&aqi=no"
        response = requests.get(url)
        if response.status_code == 200:
            data = response.json()
            weather_data = {
                "city": data['location']['name'],
                "temperature": data['current']['temp_c'],
                "humidity": data['current']['humidity'],
                "rainfall": data['current'].get('precip_mm', 0),
                "condition": data['current']['condition']['text']
            }

            # 🌱 Irrigation advice + simulated servo motor status
            if weather_data["rainfall"] > 2:
                advice = "Rainfall detected — no need to water crops today."
                servo_status = "Closed (rainfall detected)"
            elif weather_data["temperature"] > 30 and weather_data["humidity"] < 50:
                advice = "Hot and dry conditions — water your crops."
                servo_status = "Open (watering crops)"
            else:
                advice = "Moderate conditions — light irrigation may be sufficient."
                servo_status = "Partially Open (light irrigation)"
        else:
            weather_data = {"city": city, "temperature": "N/A", "humidity": "N/A", "rainfall": "N/A", "condition": "City not found"}
            advice = "Unable to provide irrigation advice."
            servo_status = "Unknown (no data)"

    return render_template('weather.html', weather=weather_data, advice=advice, servo_status=servo_status)

# -----------------------------
# Manual Servo Control
# -----------------------------
@app.route('/servo', methods=['POST'])
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
# Run App
# -----------------------------
if __name__ == "__main__":
    app.run(debug=True)
