import sqlite3
from datetime import datetime

DB_NAME = "water_quality.db"

def init_db():
    conn = sqlite3.connect(DB_NAME)
    cursor = conn.cursor()
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS predictions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp TEXT,
            ph REAL,
            hardness REAL,
            solids REAL,
            chloramines REAL,
            sulfate REAL,
            conductivity REAL,
            organic_carbon REAL,
            trihalomethanes REAL,
            turbidity REAL,
            prediction INTEGER,
            probability REAL
        )
    """)
    conn.commit()
    conn.close()

def save_prediction(data, prediction, probability):
    conn = sqlite3.connect(DB_NAME)
    cursor = conn.cursor()
    cursor.execute("""
        INSERT INTO predictions (
            timestamp, ph, hardness, solids, chloramines, sulfate,
            conductivity, organic_carbon, trihalomethanes, turbidity,
            prediction, probability
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    """, (
        datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        data['ph'], data['Hardness'], data['Solids'], data['Chloramines'],
        data['Sulfate'], data['Conductivity'], data['Organic_carbon'],
        data['Trihalomethanes'], data['Turbidity'],
        int(prediction), round(float(probability) * 100, 2)
    ))
    conn.commit()
    conn.close()

def get_all_predictions():
    conn = sqlite3.connect(DB_NAME)
    conn.row_factory = sqlite3.Row
    cursor = conn.cursor()
    cursor.execute("SELECT * FROM predictions ORDER BY id DESC")
    rows = cursor.fetchall()
    conn.close()
    return rows