import sqlite3
import os
from werkzeug.security import generate_password_hash, check_password_hash
from datetime import datetime
import json

DB_PATH = os.path.join(os.path.dirname(__file__), 'opticropt.db')

def get_db_connection():
    """Create and return a database connection with row factory"""
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn

def init_db():
    """Initialize database tables and seed default users if not present"""
    conn = get_db_connection()
    cursor = conn.cursor()

    # Users table
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS users (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            username TEXT UNIQUE NOT NULL,
            email TEXT UNIQUE NOT NULL,
            password_hash TEXT NOT NULL,
            full_name TEXT,
            role TEXT DEFAULT 'farmer',
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    ''')

    # Predictions table (Scenario 1 history)
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS predictions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id INTEGER,
            n REAL NOT NULL,
            p REAL NOT NULL,
            k REAL NOT NULL,
            temperature REAL NOT NULL,
            humidity REAL NOT NULL,
            ph REAL NOT NULL,
            rainfall REAL NOT NULL,
            recommended_crop TEXT NOT NULL,
            confidence REAL NOT NULL,
            top_recommendations TEXT,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (user_id) REFERENCES users(id)
        )
    ''')

    # Suitability assessments table (Scenario 2 history)
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS suitability_assessments (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id INTEGER,
            target_crop TEXT NOT NULL,
            n REAL NOT NULL,
            p REAL NOT NULL,
            k REAL NOT NULL,
            temperature REAL NOT NULL,
            humidity REAL NOT NULL,
            ph REAL NOT NULL,
            rainfall REAL NOT NULL,
            suitability_score REAL NOT NULL,
            suitability_level TEXT NOT NULL,
            recommendation TEXT,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (user_id) REFERENCES users(id)
        )
    ''')

    # Feedback table
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS feedback (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id INTEGER,
            name TEXT NOT NULL,
            email TEXT NOT NULL,
            rating INTEGER NOT NULL,
            comments TEXT NOT NULL,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (user_id) REFERENCES users(id)
        )
    ''')

    # Seed default users if table is empty
    cursor.execute('SELECT COUNT(*) FROM users')
    if cursor.fetchone()[0] == 0:
        demo_farmer_pw = generate_password_hash('password123')
        demo_admin_pw = generate_password_hash('admin123')
        cursor.execute('''
            INSERT INTO users (username, email, password_hash, full_name, role)
            VALUES (?, ?, ?, ?, ?)
        ''', ('farmer', 'farmer@opticrop.com', demo_farmer_pw, 'Kishan Rao', 'farmer'))
        cursor.execute('''
            INSERT INTO users (username, email, password_hash, full_name, role)
            VALUES (?, ?, ?, ?, ?)
        ''', ('admin', 'admin@opticrop.com', demo_admin_pw, 'OptiCrop Lead Admin', 'admin'))

    conn.commit()
    conn.close()

# User Management
def register_user(username, email, password, full_name='', role='farmer'):
    """Register a new user; returns (success: bool, message: str, user_id: int or None)"""
    conn = get_db_connection()
    cursor = conn.cursor()
    try:
        pw_hash = generate_password_hash(password)
        cursor.execute('''
            INSERT INTO users (username, email, password_hash, full_name, role)
            VALUES (?, ?, ?, ?, ?)
        ''', (username.strip(), email.strip().lower(), pw_hash, full_name.strip(), role))
        conn.commit()
        user_id = cursor.lastrowid
        return True, "Registration successful! You can now log in.", user_id
    except sqlite3.IntegrityError as e:
        err_msg = str(e).lower()
        if 'username' in err_msg:
            return False, "Username is already taken.", None
        elif 'email' in err_msg:
            return False, "Email address is already registered.", None
        return False, "User with this username or email already exists.", None
    except Exception as e:
        return False, f"Registration failed: {str(e)}", None
    finally:
        conn.close()

def authenticate_user(email_or_username, password):
    """Authenticate user with email or username and password"""
    conn = get_db_connection()
    cursor = conn.cursor()
    term = email_or_username.strip().lower()
    cursor.execute('''
        SELECT * FROM users 
        WHERE LOWER(email) = ? OR LOWER(username) = ?
    ''', (term, term))
    user = cursor.fetchone()
    conn.close()

    if user and check_password_hash(user['password_hash'], password):
        return dict(user)
    return None

def get_user_by_id(user_id):
    """Fetch user by id"""
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute('SELECT id, username, email, full_name, role, created_at FROM users WHERE id = ?', (user_id,))
    user = cursor.fetchone()
    conn.close()
    return dict(user) if user else None

# Prediction Management
def save_prediction(user_id, n, p, k, temperature, humidity, ph, rainfall, recommended_crop, confidence, top_recommendations):
    """Save a crop recommendation prediction"""
    conn = get_db_connection()
    cursor = conn.cursor()
    top_json = json.dumps(top_recommendations) if isinstance(top_recommendations, list) else str(top_recommendations)
    cursor.execute('''
        INSERT INTO predictions (user_id, n, p, k, temperature, humidity, ph, rainfall, recommended_crop, confidence, top_recommendations)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    ''', (user_id, n, p, k, temperature, humidity, ph, rainfall, recommended_crop, confidence, top_json))
    conn.commit()
    pred_id = cursor.lastrowid
    conn.close()
    return pred_id

def get_predictions_by_user(user_id=None, limit=50):
    """Retrieve prediction history for a user (or all if user_id is None)"""
    conn = get_db_connection()
    cursor = conn.cursor()
    if user_id:
        cursor.execute('''
            SELECT * FROM predictions 
            WHERE user_id = ? 
            ORDER BY created_at DESC 
            LIMIT ?
        ''', (user_id, limit))
    else:
        cursor.execute('''
            SELECT p.*, u.username, u.full_name 
            FROM predictions p
            LEFT JOIN users u ON p.user_id = u.id
            ORDER BY p.created_at DESC 
            LIMIT ?
        ''', (limit,))
    rows = cursor.fetchall()
    conn.close()
    return [dict(r) for r in rows]

def get_prediction_by_id(pred_id):
    """Get single prediction details"""
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute('''
        SELECT p.*, u.username, u.full_name, u.email 
        FROM predictions p
        LEFT JOIN users u ON p.user_id = u.id
        WHERE p.id = ?
    ''', (pred_id,))
    row = cursor.fetchone()
    conn.close()
    return dict(row) if row else None

# Suitability Management
def save_suitability_assessment(user_id, target_crop, n, p, k, temperature, humidity, ph, rainfall, suitability_score, suitability_level, recommendation):
    """Save crop suitability assessment"""
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute('''
        INSERT INTO suitability_assessments 
        (user_id, target_crop, n, p, k, temperature, humidity, ph, rainfall, suitability_score, suitability_level, recommendation)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    ''', (user_id, target_crop, n, p, k, temperature, humidity, ph, rainfall, suitability_score, suitability_level, recommendation))
    conn.commit()
    suit_id = cursor.lastrowid
    conn.close()
    return suit_id

def get_suitability_history(user_id=None, limit=50):
    """Get suitability assessments history"""
    conn = get_db_connection()
    cursor = conn.cursor()
    if user_id:
        cursor.execute('''
            SELECT * FROM suitability_assessments 
            WHERE user_id = ? 
            ORDER BY created_at DESC 
            LIMIT ?
        ''', (user_id, limit))
    else:
        cursor.execute('''
            SELECT s.*, u.username, u.full_name 
            FROM suitability_assessments s
            LEFT JOIN users u ON s.user_id = u.id
            ORDER BY s.created_at DESC 
            LIMIT ?
        ''', (limit,))
    rows = cursor.fetchall()
    conn.close()
    return [dict(r) for r in rows]

# Feedback
def save_feedback(user_id, name, email, rating, comments):
    """Save farmer feedback"""
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute('''
        INSERT INTO feedback (user_id, name, email, rating, comments)
        VALUES (?, ?, ?, ?, ?)
    ''', (user_id, name, email, rating, comments))
    conn.commit()
    conn.close()
    return True

def get_all_feedback(limit=20):
    """Fetch user feedback"""
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute('SELECT * FROM feedback ORDER BY created_at DESC LIMIT ?', (limit,))
    rows = cursor.fetchall()
    conn.close()
    return [dict(r) for r in rows]

# Summary statistics
def get_system_stats():
    """Retrieve high level system stats for dashboards"""
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute('SELECT COUNT(*) FROM users')
    total_users = cursor.fetchone()[0]
    cursor.execute('SELECT COUNT(*) FROM predictions')
    total_predictions = cursor.fetchone()[0]
    cursor.execute('SELECT COUNT(*) FROM suitability_assessments')
    total_assessments = cursor.fetchone()[0]
    cursor.execute('SELECT COUNT(*) FROM feedback')
    total_feedback = cursor.fetchone()[0]
    cursor.execute('''
        SELECT recommended_crop, COUNT(*) as count 
        FROM predictions 
        GROUP BY recommended_crop 
        ORDER BY count DESC 
        LIMIT 5
    ''')
    top_recommended = [dict(r) for r in cursor.fetchall()]
    conn.close()
    return {
        'total_users': total_users,
        'total_predictions': total_predictions,
        'total_assessments': total_assessments,
        'total_feedback': total_feedback,
        'top_recommended': top_recommended
    }

if __name__ == '__main__':
    init_db()
    print("Database initialized successfully at:", DB_PATH)
