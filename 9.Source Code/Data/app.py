import os
import sys
import json
import csv
import io
from flask import (
    Flask, render_template, request, redirect, url_for, 
    flash, session, jsonify, Response, send_file
)
import pandas as pd

# Add src to path
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(BASE_DIR, 'src'))

from crop_recommendation import CropRecommendationSystem
from database import (
    init_db, register_user, authenticate_user, get_user_by_id,
    save_prediction, get_predictions_by_user, get_prediction_by_id,
    save_suitability_assessment, get_suitability_history,
    save_feedback, get_all_feedback, get_system_stats
)

app = Flask(__name__)
app.secret_key = os.environ.get('SECRET_KEY', 'opticrop-super-secure-production-key-2026')

# Initialize DB
init_db()

# Initialize ML system
MODEL_DIR = os.path.join(BASE_DIR, 'models')
DATA_PATH = os.path.join(BASE_DIR, 'data', 'crop_recommendation.csv')

try:
    system = CropRecommendationSystem(MODEL_DIR)
except Exception as e:
    print(f"Warning: Model could not be loaded immediately: {e}")
    system = None

# Validation helper
def validate_crop_inputs(n, p, k, temperature, humidity, ph, rainfall):
    """Validate environmental input ranges (FT-03 / ST-02)"""
    errors = []
    try:
        n = float(n)
        if not (0 <= n <= 140):
            errors.append("Nitrogen (N) must be between 0 and 140.")
    except (ValueError, TypeError):
        errors.append("Nitrogen (N) must be a valid number.")

    try:
        p = float(p)
        if not (5 <= p <= 145):
            errors.append("Phosphorus (P) must be between 5 and 145.")
    except (ValueError, TypeError):
        errors.append("Phosphorus (P) must be a valid number.")

    try:
        k = float(k)
        if not (5 <= k <= 205):
            errors.append("Potassium (K) must be between 5 and 205.")
    except (ValueError, TypeError):
        errors.append("Potassium (K) must be a valid number.")

    try:
        temperature = float(temperature)
        if not (0 <= temperature <= 45):
            errors.append("Temperature must be between 0°C and 45°C.")
    except (ValueError, TypeError):
        errors.append("Temperature must be a valid number.")

    try:
        humidity = float(humidity)
        if not (10 <= humidity <= 100):
            errors.append("Humidity must be between 10% and 100%.")
    except (ValueError, TypeError):
        errors.append("Humidity must be a valid number.")

    try:
        ph = float(ph)
        if not (3.5 <= ph <= 9.5):
            errors.append("Soil pH must be between 3.5 and 9.5.")
    except (ValueError, TypeError):
        errors.append("Soil pH must be a valid number.")

    try:
        rainfall = float(rainfall)
        if not (20 <= rainfall <= 300):
            errors.append("Rainfall must be between 20 mm and 300 mm.")
    except (ValueError, TypeError):
        errors.append("Rainfall must be a valid number.")

    return len(errors) == 0, errors

# --- Routes ---

@app.route('/')
def index():
    """Home landing page"""
    stats = get_system_stats()
    return render_template('index.html', stats=stats)

# Scenario 1: Smart Crop Recommendation
@app.route('/recommend', methods=['GET', 'POST'])
def recommend():
    """Scenario 1: Smart Crop Recommendation for Farmers"""
    result = None
    analysis = None
    inputs = None
    pred_id = None

    if request.method == 'POST':
        n = request.form.get('n', 90)
        p = request.form.get('p', 42)
        k = request.form.get('k', 43)
        temperature = request.form.get('temperature', 20.88)
        humidity = request.form.get('humidity', 82.00)
        ph = request.form.get('ph', 6.50)
        rainfall = request.form.get('rainfall', 202.94)

        is_valid, errors = validate_crop_inputs(n, p, k, temperature, humidity, ph, rainfall)

        if not is_valid:
            for err in errors:
                flash(err, 'danger')
        else:
            inputs = {
                'n': float(n), 'p': float(p), 'k': float(k),
                'temperature': float(temperature), 'humidity': float(humidity),
                'ph': float(ph), 'rainfall': float(rainfall)
            }
            if system:
                result = system.predict_crop(
                    inputs['n'], inputs['p'], inputs['k'],
                    inputs['temperature'], inputs['humidity'], inputs['ph'], inputs['rainfall']
                )
                analysis = system.analyze_environmental_factors(
                    inputs['n'], inputs['p'], inputs['k'],
                    inputs['temperature'], inputs['humidity'], inputs['ph'], inputs['rainfall']
                )

                # Save prediction to DB
                user_id = session.get('user_id')
                pred_id = save_prediction(
                    user_id=user_id,
                    n=inputs['n'], p=inputs['p'], k=inputs['k'],
                    temperature=inputs['temperature'], humidity=inputs['humidity'],
                    ph=inputs['ph'], rainfall=inputs['rainfall'],
                    recommended_crop=result['recommended_crop'],
                    confidence=result['confidence'],
                    top_recommendations=result['top_3_recommendations']
                )
                flash(f"Crop recommendation generated: {result['recommended_crop'].upper()} ({result['confidence']:.1%})", 'success')
            else:
                flash("Model is not loaded. Please ensure models are trained.", 'danger')

    return render_template('recommend.html', result=result, analysis=analysis, inputs=inputs, pred_id=pred_id)

# Scenario 2: Crop Suitability Assessment
@app.route('/suitability', methods=['GET', 'POST'])
def suitability():
    """Scenario 2: Crop Suitability & Environmental Assessment"""
    all_crops = system.get_all_crops() if system else []
    selected_crop = 'rice'
    suitability_res = None
    inputs = None

    if request.method == 'POST':
        selected_crop = request.form.get('target_crop', 'rice').strip().lower()
        n = request.form.get('n', 70)
        p = request.form.get('p', 45)
        k = request.form.get('k', 40)
        temperature = request.form.get('temperature', 24.5)
        humidity = request.form.get('humidity', 75.0)
        ph = request.form.get('ph', 6.5)
        rainfall = request.form.get('rainfall', 120.0)

        is_valid, errors = validate_crop_inputs(n, p, k, temperature, humidity, ph, rainfall)

        if not is_valid:
            for err in errors:
                flash(err, 'danger')
        else:
            inputs = {
                'n': float(n), 'p': float(p), 'k': float(k),
                'temperature': float(temperature), 'humidity': float(humidity),
                'ph': float(ph), 'rainfall': float(rainfall)
            }
            if system:
                suitability_res = system.assess_crop_suitability(
                    inputs['n'], inputs['p'], inputs['k'],
                    inputs['temperature'], inputs['humidity'], inputs['ph'], inputs['rainfall'],
                    selected_crop
                )

                # Save assessment to DB
                user_id = session.get('user_id')
                save_suitability_assessment(
                    user_id=user_id,
                    target_crop=selected_crop,
                    n=inputs['n'], p=inputs['p'], k=inputs['k'],
                    temperature=inputs['temperature'], humidity=inputs['humidity'],
                    ph=inputs['ph'], rainfall=inputs['rainfall'],
                    suitability_score=suitability_res['suitability_score'],
                    suitability_level=suitability_res['suitability_level'],
                    recommendation=suitability_res['recommendation']
                )
                flash(f"Suitability assessed for {selected_crop.capitalize()}: {suitability_res['suitability_level']} ({suitability_res['suitability_score']:.1%})", 'info')

    return render_template('suitability.html', all_crops=all_crops, selected_crop=selected_crop, suitability=suitability_res, inputs=inputs)

# Scenario 3: Agricultural Research & Policy Dashboard
@app.route('/research')
def research():
    """Scenario 3: Agricultural Research and Policy Planning Dashboard"""
    stats = get_system_stats()
    return render_template('research.html', stats=stats)

# Prediction History
@app.route('/history')
def history():
    """View prediction and suitability audit logs"""
    user_id = session.get('user_id')
    # If admin, show all; if regular user, show user's; if guest, show recent
    role = session.get('role')
    filter_user_id = user_id if (role != 'admin' and user_id) else None
    
    predictions = get_predictions_by_user(filter_user_id, limit=100)
    assessments = get_suitability_history(filter_user_id, limit=100)
    return render_template('history.html', predictions=predictions, assessments=assessments)

# Printable Report
@app.route('/report/<int:pred_id>')
def view_report(pred_id):
    """View official printable / PDF recommendation report"""
    pred = get_prediction_by_id(pred_id)
    if not pred:
        flash("Report not found.", 'danger')
        return redirect(url_for('history'))
    
    top_recs = []
    if pred.get('top_recommendations'):
        try:
            top_recs = json.loads(pred['top_recommendations'])
        except:
            top_recs = []
    
    return render_template('report_view.html', pred=pred, top_recs=top_recs)

# Dataset Explorer
@app.route('/dataset')
def dataset_view():
    """Browse and filter dataset samples"""
    search_query = request.args.get('search', '').strip().lower()
    crop_filter = request.args.get('crop_filter', '').strip().lower()
    
    try:
        df = pd.read_csv(DATA_PATH)
        crops_list = sorted(df['label'].unique().tolist())
        
        filtered_df = df
        if crop_filter:
            filtered_df = filtered_df[filtered_df['label'] == crop_filter]
        if search_query:
            filtered_df = filtered_df[filtered_df['label'].str.contains(search_query)]
            
        rows = filtered_df.head(150).to_dict(orient='records')
        total_samples = len(df)
    except Exception as e:
        crops_list = []
        rows = []
        total_samples = 0
        flash(f"Error loading dataset: {e}", 'warning')

    return render_template('dataset.html', rows=rows, crops_list=crops_list, total_samples=total_samples, search_query=search_query, selected_filter=crop_filter)

# User Feedback
@app.route('/feedback', methods=['GET', 'POST'])
def feedback_view():
    """Farmer and user feedback form"""
    if request.method == 'POST':
        name = request.form.get('name', '').strip()
        email = request.form.get('email', '').strip()
        rating = int(request.form.get('rating', 5))
        comments = request.form.get('comments', '').strip()

        if name and email and comments:
            save_feedback(session.get('user_id'), name, email, rating, comments)
            flash("Thank you for your feedback! It helps improve OptiCrop recommendations.", 'success')
            return redirect(url_for('feedback_view'))
        else:
            flash("Please complete all required feedback fields.", 'danger')

    feedback_list = get_all_feedback(limit=25)
    return render_template('feedback.html', feedback_list=feedback_list)

# Export Predictions to CSV
@app.route('/export/predictions')
def export_predictions():
    """Export prediction records to CSV"""
    user_id = session.get('user_id')
    role = session.get('role')
    filter_user_id = user_id if (role != 'admin' and user_id) else None
    predictions = get_predictions_by_user(filter_user_id, limit=500)

    output = io.StringIO()
    writer = csv.writer(output)
    writer.writerow(['ID', 'Farmer', 'N', 'P', 'K', 'Temperature_C', 'Humidity_Pct', 'pH', 'Rainfall_mm', 'Recommended_Crop', 'Confidence', 'Date'])

    for p in predictions:
        writer.writerow([
            p['id'], p.get('full_name') or p.get('username') or 'Guest',
            p['n'], p['p'], p['k'], p['temperature'], p['humidity'], p['ph'], p['rainfall'],
            p['recommended_crop'], f"{p['confidence'] * 100:.2f}%", p['created_at']
        ])

    output.seek(0)
    return Response(
        output.getvalue(),
        mimetype="text/csv",
        headers={"Content-Disposition": "attachment;filename=opticropt_predictions.csv"}
    )

# Authentication: Register
@app.route('/register', methods=['GET', 'POST'])
def register():
    """User registration (FT-01)"""
    if request.method == 'POST':
        username = request.form.get('username', '').strip()
        email = request.form.get('email', '').strip()
        password = request.form.get('password', '')
        full_name = request.form.get('full_name', '').strip()
        role = request.form.get('role', 'farmer')

        if len(username) < 3:
            flash("Username must be at least 3 characters long.", 'danger')
        elif len(password) < 6:
            flash("Password must be at least 6 characters long.", 'danger')
        else:
            success, message, user_id = register_user(username, email, password, full_name, role)
            if success:
                flash(message, 'success')
                return redirect(url_for('login'))
            else:
                flash(message, 'danger')

    return render_template('register.html')

# Authentication: Login
@app.route('/login', methods=['GET', 'POST'])
def login():
    """User authentication (FT-02 / ST-01)"""
    if request.method == 'POST':
        username_or_email = request.form.get('username_or_email', '').strip()
        password = request.form.get('password', '')

        user = authenticate_user(username_or_email, password)
        if user:
            session['user_id'] = user['id']
            session['username'] = user['username']
            session['email'] = user['email']
            session['full_name'] = user['full_name']
            session['role'] = user['role']
            flash(f"Welcome back, {user['full_name'] or user['username']}!", 'success')
            return redirect(url_for('recommend'))
        else:
            flash("Invalid email/username or password. Please try again.", 'danger')

    return render_template('login.html')

# Authentication: Logout
@app.route('/logout')
def logout():
    """Terminate session (USN-20)"""
    session.clear()
    flash("You have successfully signed out.", 'info')
    return redirect(url_for('index'))

# User Profile
@app.route('/profile')
def profile():
    """User profile overview (USN-3)"""
    user_id = session.get('user_id')
    if not user_id:
        flash("Please log in to view your profile.", 'warning')
        return redirect(url_for('login'))

    user = get_user_by_id(user_id)
    preds = get_predictions_by_user(user_id)
    suits = get_suitability_history(user_id)
    return render_template('profile.html', user=user, user_preds_count=len(preds), user_suits_count=len(suits))

# --- REST APIs (For IoT sensors, mobile apps & automated testing) ---

@app.route('/api/predict', methods=['POST'])
def api_predict():
    """API endpoint for crop recommendation"""
    data = request.get_json(force=True, silent=True) or request.form
    try:
        n = float(data.get('n', 90))
        p = float(data.get('p', 42))
        k = float(data.get('k', 43))
        temperature = float(data.get('temperature', 20.88))
        humidity = float(data.get('humidity', 82.00))
        ph = float(data.get('ph', 6.50))
        rainfall = float(data.get('rainfall', 202.94))
    except (ValueError, TypeError) as e:
        return jsonify({'success': False, 'error': f"Invalid numeric input: {str(e)}"}), 400

    is_valid, errors = validate_crop_inputs(n, p, k, temperature, humidity, ph, rainfall)
    if not is_valid:
        return jsonify({'success': False, 'errors': errors}), 422

    if not system:
        return jsonify({'success': False, 'error': 'Model not loaded'}), 503

    result = system.predict_crop(n, p, k, temperature, humidity, ph, rainfall)
    analysis = system.analyze_environmental_factors(n, p, k, temperature, humidity, ph, rainfall)

    pred_id = save_prediction(
        user_id=session.get('user_id'),
        n=n, p=p, k=k, temperature=temperature, humidity=humidity, ph=ph, rainfall=rainfall,
        recommended_crop=result['recommended_crop'],
        confidence=result['confidence'],
        top_recommendations=result['top_3_recommendations']
    )

    return jsonify({
        'success': True,
        'prediction_id': pred_id,
        'recommended_crop': result['recommended_crop'],
        'confidence': result['confidence'],
        'top_3_recommendations': result['top_3_recommendations'],
        'analysis': analysis
    })

@app.route('/api/assess', methods=['POST'])
def api_assess():
    """API endpoint for crop suitability assessment"""
    data = request.get_json(force=True, silent=True) or request.form
    target_crop = data.get('target_crop', 'rice').strip().lower()

    try:
        n = float(data.get('n', 70))
        p = float(data.get('p', 45))
        k = float(data.get('k', 40))
        temperature = float(data.get('temperature', 24.5))
        humidity = float(data.get('humidity', 75.0))
        ph = float(data.get('ph', 6.5))
        rainfall = float(data.get('rainfall', 120.0))
    except (ValueError, TypeError) as e:
        return jsonify({'success': False, 'error': f"Invalid numeric input: {str(e)}"}), 400

    is_valid, errors = validate_crop_inputs(n, p, k, temperature, humidity, ph, rainfall)
    if not is_valid:
        return jsonify({'success': False, 'errors': errors}), 422

    if not system:
        return jsonify({'success': False, 'error': 'Model not loaded'}), 503

    res = system.assess_crop_suitability(n, p, k, temperature, humidity, ph, rainfall, target_crop)
    return jsonify({'success': True, 'suitability': res})

@app.route('/api/stats', methods=['GET'])
def api_stats():
    """System statistics API"""
    return jsonify(get_system_stats())

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    print(f"🌾 OptiCrop Flask Application starting at http://127.0.0.1:{port}")
    app.run(host='0.0.0.0', port=port, debug=True)
