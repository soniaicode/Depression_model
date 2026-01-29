"""
Flask Web Application for Depression Prediction
"""

# Suppress TensorFlow warnings BEFORE importing
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # Suppress TensorFlow logging
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'  # Disable oneDNN custom operations

# Suppress scikit-learn warnings
import warnings
warnings.filterwarnings('ignore', category=UserWarning)
warnings.filterwarnings('ignore', category=FutureWarning)
warnings.filterwarnings('ignore', message='.*InconsistentVersionWarning.*')

from flask import Flask, render_template, request, jsonify, session, redirect, url_for, flash
from flask_pymongo import PyMongo
import numpy as np
import joblib
from pathlib import Path
from datetime import datetime
import json
from flask_session import Session
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Import model configuration
from model_config import MODEL_PERFORMANCE, get_all_models, get_model_info

# Import authentication module
from auth import (
    login_required, admin_required, create_user, authenticate_user, 
    get_user_profile, get_user_demographic_data, 
    increment_assessment_count
)

# Import Gemini integration
try:
    from gemini_integration import GeminiAssistant
    gemini_assistant = GeminiAssistant(api_key=os.getenv('GEMINI_API_KEY'))
    # Temporarily disable due to quota limits
    GEMINI_ENABLED = True  # Set to True when quota resets
    print("⚠️ Gemini AI temporarily disabled (quota limits)")
except Exception as e:
    print(f"⚠ Gemini integration not available: {e}")
    gemini_assistant = None
    GEMINI_ENABLED = True

# Try to import TensorFlow, but make it optional
try:
    import tensorflow as tf
    # Suppress TensorFlow warnings
    tf.get_logger().setLevel('ERROR')
    from tensorflow import keras
    TENSORFLOW_AVAILABLE = True
except ImportError:
    TENSORFLOW_AVAILABLE = False
    print("⚠ TensorFlow not available - deep learning model will be disabled")

app = Flask(__name__)
app.secret_key = os.getenv('FLASK_SECRET_KEY', 'your-secret-key-change-this-in-production')

# Session Configuration - Server-side sessions (clears on restart)
app.config['SESSION_TYPE'] = 'filesystem'  # Store sessions on server filesystem
app.config['SESSION_FILE_DIR'] = './.flask_session/'  # Session storage directory
app.config['SESSION_PERMANENT'] = False  # Sessions expire when browser closes or server restarts
app.config['PERMANENT_SESSION_LIFETIME'] = 3600  # 1 hour (3600 seconds)
app.config['SESSION_COOKIE_SECURE'] = False  # Set to True in production with HTTPS
app.config['SESSION_COOKIE_HTTPONLY'] = True  # Prevent JavaScript access to session cookie
app.config['SESSION_COOKIE_SAMESITE'] = 'Lax'  # CSRF protection
app.config['SESSION_REFRESH_EACH_REQUEST'] = True  # Refresh session on each request

# Initialize Flask-Session (must be after config)
Session(app)

# Session timeout middleware
@app.before_request
def check_session_timeout():
    """Check if session has expired and logout user if needed"""
    from datetime import datetime, timedelta
    
    # Skip for static files and public routes
    if request.endpoint in ['static', 'landing', 'login', 'signup', 'admin_login', 'favicon']:
        return
    
    # Check if user is logged in
    if 'user_id' in session:
        # Check last activity time
        last_activity = session.get('last_activity')
        
        if last_activity:
            last_activity_time = datetime.fromisoformat(last_activity)
            current_time = datetime.now()
            
            # If inactive for more than session lifetime, logout
            if current_time - last_activity_time > timedelta(seconds=app.config['PERMANENT_SESSION_LIFETIME']):
                session.clear()
                flash('⏰ Your session has expired. Please login again.', 'warning')
                return redirect(url_for('login'))
        
        # Update last activity time
        session['last_activity'] = datetime.now().isoformat()
        session.modified = True

# Custom context processor to safely handle flash messages
@app.context_processor
def inject_safe_flash_messages():
    """Safely get flash messages, handling both old and new formats"""
    def safe_get_flashed_messages(with_categories=False):
        try:
            # Try to get messages with categories
            from flask import get_flashed_messages as original_get_flashed_messages
            messages = original_get_flashed_messages(with_categories=with_categories)
            return messages
        except (KeyError, IndexError, TypeError):
            # If that fails, clear the session and return empty
            if '_flashes' in session:
                session.pop('_flashes', None)
            return [] if not with_categories else []
    return dict(get_flashed_messages=safe_get_flashed_messages)

# MongoDB Configuration
mongo_uri = os.getenv('MONGO_URI', "mongodb://localhost:27017/depression_db")
app.config["MONGO_URI"] = mongo_uri

# Initialize MongoDB
try:
    mongo = PyMongo(app)
    # Test connection
    mongo.db.command('ping')
    print("✓ MongoDB connected successfully")
except Exception as e:
    mongo = None
    print(f"⚠ MongoDB connection failed: {e}")
    print("⚠ Please start MongoDB with: mongod")

# Load all trained models (only once, not during Flask reloader)
models_dir = Path('data/models')
models = {}
scaler = None

def load_models():
    """Load all ML models - called only once"""
    global models, scaler
    
    if models:  # Already loaded
        return
    
    print("\n" + "="*60)
    print("🔄 Loading ML Models...")
    print("="*60)
    
    try:
        # Load all models with version compatibility handling
        model_types = ['logistic_regression', 'random_forest', 'gradient_boosting']
        for model_type in model_types:
            try:
                model_files = sorted(models_dir.glob(f'{model_type}_*.pkl'))
                if model_files:
                    model_path = model_files[-1]
                    print(f"📦 Loading {model_type}...")
                    
                    # Load with warnings suppressed
                    with warnings.catch_warnings():
                        warnings.simplefilter("ignore")
                        models[model_type] = joblib.load(model_path)
                    
                    print(f"✅ {model_type} loaded successfully")
                else:
                    print(f"⚠️  No model file found for {model_type}")
            except Exception as e:
                print(f"❌ Could not load {model_type}: {str(e)}")
                if "sklearn" in str(e).lower() or "version" in str(e).lower():
                    print(f"   💡 Hint: Model was trained with different scikit-learn version")
                    print(f"   💡 Solution: Retrain models or match scikit-learn version")
        
        # Load deep learning model (only if TensorFlow is available)
        if TENSORFLOW_AVAILABLE:
            try:
                # Try .keras files first (new format), then .h5 (legacy)
                dl_files = sorted(models_dir.glob('deep_learning_*.keras'))
                if not dl_files:
                    dl_files = sorted(models_dir.glob('deep_learning_*.h5'))
                
                if dl_files:
                    print(f"📦 Loading deep_learning model...")
                    with warnings.catch_warnings():
                        warnings.simplefilter("ignore")
                        models['deep_learning'] = keras.models.load_model(
                            dl_files[-1], 
                            compile=False
                        )
                    print(f"✅ deep_learning loaded successfully")
            except Exception as e:
                print(f"❌ Could not load deep_learning: {str(e)}")
                print(f"   💡 Hint: Model may need to be retrained with current TensorFlow version")
        else:
            print("⚠️  Skipping deep learning model (TensorFlow not installed)")
        
        print(f"\n✅ Total models loaded: {len(models)}")
        print(f"📊 Available models: {list(models.keys())}")
        
        # If no models loaded, show helpful message
        if len(models) == 0:
            print("\n⚠️  WARNING: No models could be loaded!")
            print("💡 This is likely due to package version conflicts.")
            print("🔧 To fix this:")
            print("   1. Run: python quick_fix.py")
            print("   2. Then: python retrain_models_fixed.py")
            print("   3. Restart the app")
        elif len(models) == 1:
            print("\n⚠️  Only 1 model loaded (package version issues detected)")
            print("💡 For full functionality, run: python quick_fix.py")
        
    except Exception as e:
        print(f"❌ Error in model loading process: {e}")
    
    # Load scaler
    try:
        scaler_path = Path('data/processed/scaler.pkl')
        if scaler_path.exists():
            print(f"📦 Loading scaler...")
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                scaler = joblib.load(scaler_path)
            print("✅ Scaler loaded successfully")
        else:
            print("⚠️  Scaler file not found")
    except Exception as e:
        print(f"❌ Scaler loading failed: {e}")
    
    print("="*60 + "\n")

# Load models at startup (always)
print("\n🚀 Starting Flask app...")
print(f"   WERKZEUG_RUN_MAIN: {os.environ.get('WERKZEUG_RUN_MAIN')}")

# Load models with error handling
try:
    load_models()
except Exception as e:
    print(f"❌ Critical error loading models: {e}")
    import traceback
    traceback.print_exc()
    # Initialize empty models dict to prevent crashes
    models = {}
    scaler = None


@app.route('/')
def landing():
    """Landing page"""
    # If already logged in, redirect to home
    if 'user_id' in session:
        return redirect(url_for('index'))
    return render_template('landing.html')


@app.route('/favicon.ico')
def favicon():
    """Favicon route to prevent 404 errors"""
    from flask import send_from_directory
    import os
    # Return empty response if favicon doesn't exist
    return '', 204


@app.route('/api/health')
def health_check():
    """Simple health check endpoint"""
    return jsonify({
        'status': 'ok',
        'models_loaded': len(models) if models else 0,
        'models_list': list(models.keys()) if models else []
    })

@app.route('/home')
@login_required
def index():
    """Home page"""
    user = None
    if 'user_id' in session:
        user = get_user_profile(mongo, session['user_id'])
    return render_template('index.html', user=user)


@app.route('/signup', methods=['GET', 'POST'])
def signup():
    """User registration"""
    # If already logged in, redirect to home
    if 'user_id' in session:
        return redirect(url_for('index'))
    
    if request.method == 'POST':
        # Check if MongoDB is available BEFORE try block
        if mongo is None:
            flash('⚠️ Database not configured. User registration is currently unavailable. You can still use the prediction feature without login!', 'warning')
            return render_template('signup.html')
        
        try:
            user_data = {
                'email': request.form.get('email'),
                'password': request.form.get('password'),
                'full_name': request.form.get('full_name'),
                'age': request.form.get('age'),
                'gender': request.form.get('gender'),
                'phone': request.form.get('phone', ''),
                'education_years': request.form.get('education_years', 14),
                'income_level': request.form.get('income_level', 2),
                'marital_status': request.form.get('marital_status', 0),
                'employment_status': request.form.get('employment_status', 1),
                'residence_type': request.form.get('residence_type', 1),
                'family_history_depression': request.form.get('family_history_depression', 0),
            }
            
            success, message, user_id = create_user(mongo, user_data)
            
            if success:
                # Auto-login after successful signup
                from bson.objectid import ObjectId
                user = mongo.db.users.find_one({'_id': ObjectId(user_id)})
                
                if user:
                    from datetime import datetime
                    session['user_id'] = str(user['_id'])
                    session['user_email'] = user['email']
                    session['user_name'] = user['full_name']
                    session['is_admin'] = user.get('is_admin', False)
                    session['last_activity'] = datetime.now().isoformat()
                    session.permanent = False  # Session clears on server restart
                    
                    flash(f'🎉 Welcome {user["full_name"]}! Your account has been created successfully!', 'success')
                    # Redirect to dashboard after signup
                    return redirect(url_for('dashboard'))
                else:
                    flash(message, 'success')
                    return redirect(url_for('login'))
            else:
                flash(message, 'error')
                
        except Exception as e:
            flash(f'Error creating account: {str(e)}', 'error')
    
    return render_template('signup.html')


@app.route('/login', methods=['GET', 'POST'])
def login():
    """User login"""
    # If already logged in, redirect to home
    if 'user_id' in session:
        return redirect(url_for('index'))
    
    if request.method == 'POST':
        # Check if MongoDB is available BEFORE try block
        if mongo is None:
            flash('⚠️ Database not configured. User login is currently unavailable. You can still use the prediction feature without login!', 'warning')
            return render_template('login.html')
        
        try:
            email = request.form.get('email')
            password = request.form.get('password')
            
            success, message, user_data = authenticate_user(mongo, email, password)
            
            if success:
                from datetime import datetime
                session['user_id'] = user_data['_id']
                session['user_email'] = user_data['email']
                session['user_name'] = user_data['full_name']
                session['is_admin'] = user_data.get('is_admin', False)
                session['last_activity'] = datetime.now().isoformat()
                session.permanent = False  # Session clears on server restart
                
                flash(f'✨ Welcome back, {user_data["full_name"]}!', 'success')
                
                # Redirect admin to admin panel
                if session.get('is_admin'):
                    return redirect(url_for('admin_dashboard'))
                else:
                    # Redirect to dashboard after login
                    return redirect(url_for('dashboard'))
            else:
                # Check if user doesn't exist
                if "Invalid email or password" in message:
                    user_exists = mongo.db.users.find_one({'email': email})
                    if not user_exists:
                        flash('❌ Account not found! Please create your account first.', 'error')
                        return render_template('login.html', show_signup_prompt=True)
                
                flash(message, 'error')
                
        except Exception as e:
            flash(f'Error during login: {str(e)}', 'error')
    
    return render_template('login.html')


@app.route('/logout')
def logout():
    """User logout"""
    session.clear()
    return redirect(url_for('login'))


@app.route('/forgot-password', methods=['GET', 'POST'])
def forgot_password():
    """Forgot password - send reset code"""
    if request.method == 'POST':
        if mongo is None:
            flash('⚠️ Database not configured. Password reset is currently unavailable.', 'warning')
            return render_template('forgot_password.html')
        
        try:
            email = request.form.get('email')
            
            # Check if user exists
            user = mongo.db.users.find_one({'email': email})
            
            if user:
                # Generate 6-digit reset code
                import random
                reset_code = str(random.randint(100000, 999999))
                
                # Store reset code in database with expiration (10 minutes)
                from datetime import datetime, timedelta
                expiration = datetime.utcnow() + timedelta(minutes=10)
                
                mongo.db.users.update_one(
                    {'email': email},
                    {'$set': {
                        'reset_code': reset_code,
                        'reset_code_expiration': expiration
                    }}
                )
                
                # In production, send email here
                # For now, show the code in flash message (development only)
                flash(f'Reset code sent! For development: Your code is {reset_code}', 'success')
                flash('Please check your email for the reset code.', 'info')
                
                return redirect(url_for('reset_password'))
            else:
                flash('Email not found. Please check and try again.', 'error')
                
        except Exception as e:
            flash(f'Error: {str(e)}', 'error')
    
    return render_template('forgot_password.html')


@app.route('/reset-password', methods=['GET', 'POST'])
def reset_password():
    """Reset password with code"""
    if request.method == 'POST':
        if mongo is None:
            flash('⚠️ Database not configured. Password reset is currently unavailable.', 'warning')
            return render_template('reset_password.html')
        
        try:
            email = request.form.get('email')
            reset_code = request.form.get('reset_code')
            new_password = request.form.get('new_password')
            
            # Find user with matching email and reset code
            from datetime import datetime
            user = mongo.db.users.find_one({
                'email': email,
                'reset_code': reset_code,
                'reset_code_expiration': {'$gt': datetime.utcnow()}
            })
            
            if user:
                # Validate password
                from auth import validate_password
                from werkzeug.security import generate_password_hash
                
                is_valid, message = validate_password(new_password)
                if not is_valid:
                    flash(message, 'error')
                    return render_template('reset_password.html')
                
                # Update password and remove reset code
                hashed_password = generate_password_hash(new_password)
                mongo.db.users.update_one(
                    {'email': email},
                    {
                        '$set': {'password': hashed_password},
                        '$unset': {'reset_code': '', 'reset_code_expiration': ''}
                    }
                )
                
                flash('Password reset successful! Please login with your new password.', 'success')
                return redirect(url_for('login'))
            else:
                flash('Invalid or expired reset code. Please try again.', 'error')
                
        except Exception as e:
            flash(f'Error: {str(e)}', 'error')
    
    return render_template('reset_password.html')


# ==================== ADMIN ROUTES ====================

@app.route('/admin/login', methods=['GET', 'POST'])
def admin_login():
    """Admin login page"""
    # If already logged in as admin, redirect to dashboard
    if session.get('is_admin'):
        return redirect(url_for('admin_dashboard'))
    
    if request.method == 'POST':
        username = request.form.get('username')
        password = request.form.get('password')
        
        # Admin credentials
        ADMIN_USERNAME = os.getenv('ADMIN_USERNAME', 'admin')
        ADMIN_PASSWORD = os.getenv('ADMIN_PASSWORD', 'Admin@123')
        
        if username == ADMIN_USERNAME and password == ADMIN_PASSWORD:
            from datetime import datetime
            session['is_admin'] = True
            session['admin_username'] = username
            session['last_activity'] = datetime.now().isoformat()
            session.permanent = False  # Session clears on server restart
            flash('Admin login successful!', 'success')
            return redirect(url_for('admin_dashboard'))
        else:
            flash('Invalid admin credentials!', 'error')
    
    return render_template('admin_login.html')


@app.route('/admin/logout')
def admin_logout():
    """Admin logout"""
    session.pop('is_admin', None)
    session.pop('admin_username', None)
    flash('Admin logged out successfully', 'success')
    return redirect(url_for('admin_login'))


@app.route('/api/admin/model-usage')
@admin_required
def admin_model_usage():
    """API endpoint for model usage statistics"""
    if mongo is None:
        return jsonify({'labels': [], 'values': []})
    
    try:
        # Get model usage counts
        model_names = {
            'deep_learning': 'Enhanced Multimodal',
            'random_forest': 'Random Forest',
            'gradient_boosting': 'XGBoost',
            'logistic_regression': 'Logistic Regression'
        }
        
        labels = []
        values = []
        
        for model_key, model_name in model_names.items():
            count = mongo.db.predictions.count_documents({'model_type': model_key})
            if count > 0:  # Only include models that have been used
                labels.append(model_name)
                values.append(count)
        
        return jsonify({
            'labels': labels,
            'values': values
        })
    except Exception as e:
        print(f"Error getting model usage: {e}")
        return jsonify({'labels': [], 'values': []})


@app.route('/admin/dashboard')
@admin_required
def admin_dashboard():
    """Admin dashboard"""
    # Check if MongoDB is available
    if mongo is None:
        flash('⚠️ Database not configured. Admin dashboard is currently unavailable.', 'warning')
        return render_template('admin_dashboard.html', stats={}, recent_users=[])
    
    try:
        from datetime import datetime, timedelta
        from bson.objectid import ObjectId
        
        # Get statistics
        total_users = mongo.db.users.count_documents({})
        total_predictions = mongo.db.predictions.count_documents({})
        depression_cases = mongo.db.predictions.count_documents({'result.prediction': 1})
        
        # Get active users today
        today = datetime.now().replace(hour=0, minute=0, second=0, microsecond=0)
        active_today = mongo.db.users.count_documents({
            'last_login': {'$gte': today}
        })
        
        # Get recent users
        recent_users = list(mongo.db.users.find().sort('created_at', -1).limit(5))
        
        stats = {
            'total_users': total_users,
            'total_predictions': total_predictions,
            'depression_cases': depression_cases,
            'active_today': active_today
        }
        
        # Add current date and time
        now = datetime.now()
        current_date = now.strftime('%B %d, %Y')
        current_time = now.strftime('%I:%M %p')
        
        return render_template('admin_dashboard.html', 
                             stats=stats, 
                             recent_users=recent_users,
                             current_date=current_date,
                             current_time=current_time)
    except Exception as e:
        flash(f'Error loading dashboard: {str(e)}', 'error')
        from datetime import datetime
        now = datetime.now()
        return render_template('admin_dashboard.html', 
                             stats={}, 
                             recent_users=[],
                             current_date=now.strftime('%B %d, %Y'),
                             current_time=now.strftime('%I:%M %p'))


@app.route('/admin/users')
@admin_required
def admin_users():
    """Manage all users"""
    # Check if MongoDB is available
    if mongo is None:
        flash('⚠️ Database not configured. User management is currently unavailable.', 'warning')
        return render_template('admin/users.html', users=[])
    
    try:
        # Get all users
        users = list(mongo.db.users.find().sort('created_at', -1))
        for user in users:
            user['_id'] = str(user['_id'])
            # Get user's prediction count
            from bson.objectid import ObjectId
            user['prediction_count'] = mongo.db.predictions.count_documents({
                'user_id': ObjectId(user['_id'])
            })
        
        return render_template('admin/users.html', users=users)
    except Exception as e:
        flash(f'Error loading users: {str(e)}', 'error')
        return render_template('admin/users.html', users=[])


@app.route('/admin/user/<user_id>')
@admin_required
def admin_user_detail(user_id):
    """View user details"""
    try:
        from bson.objectid import ObjectId
        
        # Get user
        user = mongo.db.users.find_one({'_id': ObjectId(user_id)})
        if not user:
            flash('User not found', 'error')
            return redirect(url_for('admin_users'))
        
        user['_id'] = str(user['_id'])
        
        # Get user's predictions
        predictions = list(mongo.db.predictions.find({
            'user_id': ObjectId(user_id)
        }).sort('timestamp', -1))
        
        for pred in predictions:
            pred['_id'] = str(pred['_id'])
            pred['user_id'] = str(pred['user_id'])
        
        return render_template('admin/user_detail.html', user=user, predictions=predictions)
    except Exception as e:
        flash(f'Error loading user details: {str(e)}', 'error')
        return redirect(url_for('admin_users'))


@app.route('/admin/user/<user_id>/toggle-admin', methods=['POST'])
@admin_required
def admin_toggle_admin(user_id):
    """Toggle admin status for user"""
    try:
        from bson.objectid import ObjectId
        
        user = mongo.db.users.find_one({'_id': ObjectId(user_id)})
        if not user:
            flash('User not found', 'error')
            return redirect(url_for('admin_users'))
        
        # Toggle admin status
        new_status = not user.get('is_admin', False)
        mongo.db.users.update_one(
            {'_id': ObjectId(user_id)},
            {'$set': {'is_admin': new_status}}
        )
        
        action = 'granted' if new_status else 'revoked'
        flash(f'Admin privileges {action} for {user["full_name"]}', 'success')
        
    except Exception as e:
        flash(f'Error updating admin status: {str(e)}', 'error')
    
    return redirect(url_for('admin_user_detail', user_id=user_id))


@app.route('/admin/user/<user_id>/delete', methods=['POST'])
@admin_required
def admin_delete_user(user_id):
    """Delete user and their data"""
    try:
        from bson.objectid import ObjectId
        
        # Don't allow deleting yourself
        if str(session['user_id']) == user_id:
            flash('You cannot delete your own account', 'error')
            return redirect(url_for('admin_users'))
        
        user = mongo.db.users.find_one({'_id': ObjectId(user_id)})
        if not user:
            flash('User not found', 'error')
            return redirect(url_for('admin_users'))
        
        # Delete user's predictions
        mongo.db.predictions.delete_many({'user_id': ObjectId(user_id)})
        
        # Delete user
        mongo.db.users.delete_one({'_id': ObjectId(user_id)})
        
        flash(f'User {user["full_name"]} and all their data deleted successfully', 'success')
        
    except Exception as e:
        flash(f'Error deleting user: {str(e)}', 'error')
    
    return redirect(url_for('admin_users'))


@app.route('/admin/predictions')
@admin_required
def admin_predictions():
    """View all predictions"""
    # Check if MongoDB is available
    if mongo is None:
        flash('⚠️ Database not configured. Predictions view is currently unavailable.', 'warning')
        return render_template('admin/predictions.html', predictions=[])
    
    try:
        # Get all predictions with user info
        predictions = list(mongo.db.predictions.find().sort('timestamp', -1).limit(100))
        
        print(f"📊 Found {len(predictions)} predictions in database")
        
        for pred in predictions:
            pred['_id'] = str(pred['_id'])
            
            # Get user info if user_id exists
            if 'user_id' in pred and pred['user_id']:
                try:
                    from bson.objectid import ObjectId
                    user = mongo.db.users.find_one({'_id': ObjectId(pred['user_id'])})
                    if user:
                        pred['user_name'] = user.get('full_name', 'Unknown')
                        pred['user_email'] = user.get('email', 'Unknown')
                    else:
                        pred['user_name'] = 'Unknown User'
                        pred['user_email'] = 'N/A'
                    pred['user_id'] = str(pred['user_id'])
                except Exception as e:
                    print(f"Error fetching user for prediction: {e}")
                    pred['user_name'] = 'Unknown'
                    pred['user_email'] = 'N/A'
            else:
                pred['user_name'] = 'Guest User'
                pred['user_email'] = 'N/A'
            
            # Ensure result structure exists
            if 'result' not in pred:
                pred['result'] = {
                    'prediction': 0,
                    'probability': {'depression': 0, 'no_depression': 0},
                    'model_used': 'N/A',
                    'risk_level': 'Unknown'
                }
        
        print(f"✅ Successfully processed {len(predictions)} predictions")
        return render_template('admin/predictions.html', predictions=predictions)
        
    except Exception as e:
        print(f"❌ Error loading predictions: {str(e)}")
        import traceback
        traceback.print_exc()
        flash(f'Error loading predictions: {str(e)}', 'error')
        return render_template('admin/predictions.html', predictions=[])


@app.route('/admin/settings', methods=['GET', 'POST'])
@admin_required
def admin_settings():
    """System settings"""
    if request.method == 'POST':
        try:
            # Update system settings
            settings = {
                'site_name': request.form.get('site_name'),
                'maintenance_mode': request.form.get('maintenance_mode') == 'on',
                'allow_signups': request.form.get('allow_signups') == 'on',
                'gemini_enabled': request.form.get('gemini_enabled') == 'on',
            }
            
            # Save to database or config file
            mongo.db.settings.update_one(
                {'_id': 'system_settings'},
                {'$set': settings},
                upsert=True
            )
            
            flash('Settings updated successfully', 'success')
        except Exception as e:
            flash(f'Error updating settings: {str(e)}', 'error')
    
    # Load current settings
    try:
        settings = mongo.db.settings.find_one({'_id': 'system_settings'}) or {}
    except:
        settings = {}
    
    return render_template('admin/settings.html', settings=settings)


@app.route('/profile')
@login_required
def profile():
    """User profile page"""
    user = get_user_profile(mongo, session['user_id'])
    
    # Get user's prediction history
    from bson.objectid import ObjectId
    predictions = list(mongo.db.predictions.find(
        {'user_id': ObjectId(session['user_id'])}
    ).sort('timestamp', -1).limit(10))
    
    for pred in predictions:
        pred['_id'] = str(pred['_id'])
    
    return render_template('profile.html', user=user, predictions=predictions)


@app.route('/edit-profile', methods=['GET', 'POST'])
@login_required
def edit_profile():
    """Edit user profile"""
    from bson.objectid import ObjectId
    
    if request.method == 'POST':
        try:
            user_id = ObjectId(session['user_id'])
            
            # Get form data
            update_data = {
                'full_name': request.form.get('full_name'),
                'age': int(request.form.get('age', 0)),
                'gender': request.form.get('gender'),
                'phone': request.form.get('phone', ''),
                'education_years': int(request.form.get('education_years', 14)),
                'income_level': int(request.form.get('income_level', 2)),
                'marital_status': int(request.form.get('marital_status', 0)),
                'employment_status': int(request.form.get('employment_status', 1)),
                'residence_type': int(request.form.get('residence_type', 1)),
                'family_history_depression': int(request.form.get('family_history_depression', 0)),
            }
            
            # Update user in database
            result = mongo.db.users.update_one(
                {'_id': user_id},
                {'$set': update_data}
            )
            
            if result.modified_count > 0:
                flash('Profile updated successfully!', 'success')
            else:
                flash('No changes made to profile.', 'info')
            
            return redirect(url_for('profile'))
            
        except Exception as e:
            flash(f'Error updating profile: {str(e)}', 'error')
    
    # GET request - show edit form
    user = get_user_profile(mongo, session['user_id'])
    return render_template('edit_profile.html', user=user)


def map_questionnaire_to_model_features(questionnaire_data, user_demographic_data=None):
    """
    Map 31 questionnaire features to 24 model features (matching scaler)
    
    Questionnaire structure (31 features):
    - 0-8: PHQ-9 questions
    - 9-14: Women-specific factors
    - 15-24: Physiological indicators
    - 25-30: Social & environmental factors
    
    Model expects (24 features):
    - 6 women-specific features
    - 10 physiological features
    - 6 social/environmental features
    - 2 derived features (PHQ-9 total and flag)
    """
    
    # Extract questionnaire responses
    q = questionnaire_data
    
    # Use user demographic data if available, otherwise use defaults
    if user_demographic_data:
        age = user_demographic_data.get('age', 35.0)
        education_years = user_demographic_data.get('education_years', 14.0)
        income_level = user_demographic_data.get('income_level', 2)
        marital_status = user_demographic_data.get('marital_status', 1)
        employment = user_demographic_data.get('employment_status', 1)
        residence = user_demographic_data.get('residence_type', 1)
        family_history = user_demographic_data.get('family_history_depression', 0)
    else:
        age = 35.0
        education_years = 14.0
        income_level = 2
        marital_status = 1
        employment = 1
        residence = 1
        family_history = 0
    
    # Derive model features from questionnaire (24 features to match scaler)
    model_features = [
        # Women-specific factors (Q10-Q15) - 6 features
        q[9],   # hormonal changes Q10
        q[10],  # postpartum mood changes Q11
        q[11],  # body image Q12
        q[12],  # relationship stress Q13
        q[13],  # work-life balance Q14
        q[14],  # caregiving burden Q15
        
        # Physiological indicators (Q16-Q25) - 10 features
        q[15],  # heart_rate Q16
        q[16],  # hrv Q17
        q[17],  # sleep_duration Q18
        q[18],  # sleep_quality Q19
        q[19],  # physical_activity Q20
        q[20],  # stress_level Q21
        q[21],  # bp_systolic Q22
        q[22],  # bp_diastolic Q23
        q[23],  # bmi Q24
        q[24],  # vitamin_d Q25
        
        # Social & Environmental (Q26-Q31) - 6 features
        q[25],  # social_support Q26
        q[26],  # financial_stress Q27
        q[27],  # traumatic_events Q28
        q[28],  # substance_use Q29
        q[29],  # chronic_illness Q30
        q[30],  # family_history Q31
        
        # Derived from PHQ-9 (Q1-Q9) - 2 features
        sum([q[i] for i in range(9)]),  # phq9_total_score
        1 if sum([q[i] for i in range(9)]) >= 10 else 0,  # phq9_depression_flag
    ]
    
    # Total: 6 + 10 + 6 + 2 = 24 features (matches scaler)
    return np.array(model_features).reshape(1, -1)


@app.route('/api/available-models')
def available_models():
    """API endpoint to get available models"""
    # Debug logging
    print(f"\n🔍 API /api/available-models called")
    
    # Check if models variable exists and is accessible
    try:
        print(f"   Checking models variable...")
        global models
        print(f"   Models type: {type(models)}")
        print(f"   Models is None: {models is None}")
        print(f"   Models keys: {list(models.keys()) if models else 'None'}")
        print(f"   Models count: {len(models) if models else 0}")
    except Exception as e:
        print(f"   ❌ Error accessing models: {e}")
    
    # Check if models are loaded
    if not models or len(models) == 0:
        print(f"   ⚠️ No models loaded - returning error response")
        return jsonify({
            'success': False,
            'models': {},
            'total': 0,
            'loaded_models': [],
            'error': 'No models loaded',
            'message': 'Models need to be trained. Please run: python scripts/train_models.py'
        })
    
    # Get available models from central config
    available = {}
    for key in models.keys():
        if key in MODEL_PERFORMANCE:
            available[key] = {
                **MODEL_PERFORMANCE[key],
                'loaded': True,
                'available': True
            }
    
    print(f"   ✅ Returning {len(available)} models")
    
    return jsonify({
        'success': True,
        'models': available,
        'total': len(available),
        'loaded_models': list(models.keys())
    })


@app.route('/api/get-ai-insights', methods=['POST'])
def get_ai_insights():
    """API endpoint to get AI-powered insights on demand"""
    try:
        # Get request data
        data = request.get_json()
        
        prediction = data.get('prediction')
        probability = data.get('probability')
        risk_level = data.get('risk_level')
        questionnaire_data = data.get('questionnaire_data', {})
        
        print(f"🧠 AI Insights requested for prediction: {prediction}, risk: {risk_level}")
        
        # Check if Gemini is enabled
        if not GEMINI_ENABLED or not gemini_assistant:
            print("⚠️ Gemini AI not available - returning fallback")
            return jsonify({
                'success': False,
                'error': 'AI insights temporarily unavailable',
                'message': 'AI service is currently disabled or unavailable'
            })
        
        # Generate AI insights
        try:
            ai_explanation = gemini_assistant.generate_prediction_explanation(
                prediction=prediction,
                probability=probability,
                risk_level=risk_level,
                questionnaire_data=questionnaire_data
            )
            
            wellness_tips = gemini_assistant.generate_wellness_tips(
                risk_level=risk_level,
                questionnaire_data=questionnaire_data
            )
            
            spiritual_guidance = gemini_assistant.generate_spiritual_guidance(
                risk_level=risk_level,
                questionnaire_data=questionnaire_data
            )
            
            print("✅ AI insights generated successfully")
            
            return jsonify({
                'success': True,
                'ai_explanation': ai_explanation,
                'wellness_tips': wellness_tips,
                'spiritual_guidance': spiritual_guidance
            })
            
        except Exception as e:
            print(f"❌ Error generating AI insights: {e}")
            import traceback
            traceback.print_exc()
            
            return jsonify({
                'success': False,
                'error': 'Failed to generate AI insights',
                'message': str(e)
            })
            
    except Exception as e:
        print(f"❌ Error in get_ai_insights endpoint: {e}")
        import traceback
        traceback.print_exc()
        
        return jsonify({
            'success': False,
            'error': 'Internal server error',
            'message': str(e)
        }), 500


@app.route('/model-status')
def model_status():
    """Model status page"""
    return render_template('model_status.html')


@app.route('/train-guide')
def train_guide():
    """Model training guide page"""
    return render_template('train_models_guide.html')


@app.route('/test-predict')
def test_predict():
    """Test predict endpoint (for debugging)"""
    return render_template('test_predict.html')


@app.route('/predict-enhanced')
@login_required
def predict_enhanced():
    """Enhanced prediction page with better UI"""
    return render_template('predict_enhanced.html')


@app.route('/predict', methods=['GET', 'POST'])
@login_required
def predict():
    """Prediction page"""
    if request.method == 'POST':
        try:
            # Get form data
            data = request.get_json()
            
            # Get selected model (default to first available model)
            default_model = list(models.keys())[0] if models else 'logistic_regression'
            model_type = data.get('model_type', default_model)
            print(f"Selected model type: {model_type}")
            print(f"Available models: {list(models.keys())}")
            
            if model_type not in models:
                # If requested model not available, use default
                available_models = list(models.keys())
                if not available_models:
                    return jsonify({
                        'success': False,
                        'error': 'No models available. Please contact administrator.'
                    }), 500
                
                # Use first available model as fallback
                model_type = available_models[0]
                print(f"⚠ Requested model not available, using fallback: {model_type}")
                
                return jsonify({
                    'success': False,
                    'error': f'Model not available. Please select from: {", ".join(available_models)}',
                    'available_models': available_models
                }), 400
            
            # Extract questionnaire features (31 features)
            questionnaire_features = []
            for i in range(31):
                key = f'feature_{i}'
                value = float(data.get(key, 0))
                questionnaire_features.append(value)
            
            print(f"Extracted {len(questionnaire_features)} questionnaire features")
            
            # Get user demographic data for enhanced predictions
            user_demographic_data = get_user_demographic_data(mongo, session.get('user_id'))
            print(f"User demographic data: {user_demographic_data}")
            
            # Map questionnaire features to model features (28 features)
            features_array = map_questionnaire_to_model_features(questionnaire_features, user_demographic_data)
            print(f"Mapped to model features shape: {features_array.shape}")
            
            # Scale features
            if scaler:
                features_scaled = scaler.transform(features_array)
            else:
                features_scaled = features_array
            
            # Make prediction with selected model
            selected_model = models[model_type]
            
            if model_type == 'deep_learning':
                # Deep learning model expects 2 inputs: [questionnaire, physiological]
                # Model was trained with:
                # - Questionnaire: 12 features (women-specific + social/environmental)
                # - Physiological: 11 features (10 physiological + Age)
                
                # But our current features (24 total) don't include Age separately
                # So we need to match the training structure
                
                # Questionnaire features (12):
                # - Women-specific (0-5): 6 features
                # - Social/environmental (16-21): 6 features
                questionnaire_input = np.concatenate([
                    features_scaled[:, 0:6],   # Women-specific (Q10-Q15) - 6 features
                    features_scaled[:, 16:22]  # Social/environmental (Q26-Q31) - 6 features
                ], axis=1)  # Total: 12 features
                
                # Physiological features (11):
                # - Physiological indicators (6-15): 10 features
                # - PHQ-9 total score (22): 1 feature (used as proxy for Age/severity)
                physiological_input = np.concatenate([
                    features_scaled[:, 6:16],   # Physiological (Q16-Q25) - 10 features
                    features_scaled[:, 22:23]   # PHQ-9 total score - 1 feature
                ], axis=1)  # Total: 11 features
                
                print(f"Questionnaire input shape: {questionnaire_input.shape}")
                print(f"Physiological input shape: {physiological_input.shape}")
                
                # Predict with both inputs
                prediction_proba = selected_model.predict([questionnaire_input, physiological_input], verbose=0).flatten()[0]
                prediction = 1 if prediction_proba >= 0.5 else 0
                probability = [1 - prediction_proba, prediction_proba]
            else:
                prediction = selected_model.predict(features_scaled)[0]
                probability = selected_model.predict_proba(features_scaled)[0]
            
            # Apply probability smoothing to prevent extreme 0% or 100%
            # This is important for medical predictions - never show absolute certainty
            epsilon = 0.01  # Minimum probability (1%)
            prob_no_depression = float(probability[0])
            prob_depression = float(probability[1])
            
            # Smooth extreme probabilities
            if prob_no_depression > 0.99:
                prob_no_depression = 0.99
                prob_depression = 0.01
            elif prob_depression > 0.99:
                prob_depression = 0.99
                prob_no_depression = 0.01
            elif prob_no_depression < 0.01:
                prob_no_depression = 0.01
                prob_depression = 0.99
            elif prob_depression < 0.01:
                prob_depression = 0.01
                prob_no_depression = 0.99
            
            # Ensure probabilities sum to 1.0
            total = prob_no_depression + prob_depression
            prob_no_depression = prob_no_depression / total
            prob_depression = prob_depression / total
            
            result = {
                'prediction': int(prediction),
                'probability': {
                    'no_depression': prob_no_depression,
                    'depression': prob_depression
                },
                'risk_level': get_risk_level(prob_depression),
                'model_used': model_type,
                'confidence_note': 'Probabilities are calibrated for medical safety (never showing 0% or 100%)'
            }
            
            # Don't generate AI insights automatically - let user request them via button
            # This saves API quota and gives users control
            
            # Save to database with user_id
            save_prediction(data, result, session.get('user_id'))
            
            # Increment user's assessment count
            if session.get('user_id'):
                increment_assessment_count(mongo, session['user_id'])
            
            return jsonify({
                'success': True,
                'result': result
            })
                
        except Exception as e:
            import traceback
            error_details = traceback.format_exc()
            print(f"Error: {str(e)}")
            print(f"Traceback: {error_details}")
            return jsonify({
                'success': False,
                'error': str(e),
                'details': error_details
            }), 400
    
    return render_template('predict.html')


@app.route('/history')
@login_required
def history():
    """View prediction history for current user"""
    try:
        from bson.objectid import ObjectId
        user_id = ObjectId(session['user_id'])
        
        # Get only current user's predictions
        predictions = list(mongo.db.predictions.find(
            {'user_id': user_id}
        ).sort('timestamp', -1).limit(50))
        
        # Convert ObjectId to string
        for pred in predictions:
            pred['_id'] = str(pred['_id'])
            if 'user_id' in pred:
                pred['user_id'] = str(pred['user_id'])
        
        # Get user info
        user = get_user_profile(mongo, session['user_id'])
        
        return render_template('history.html', predictions=predictions, user=user)
    except Exception as e:
        return render_template('history.html', predictions=[], error=str(e), user=None)


@app.route('/dashboard')
@login_required
def dashboard():
    """Analytics dashboard for current user"""
    try:
        from bson.objectid import ObjectId
        user_id = ObjectId(session['user_id'])
        
        # Get only current user's predictions
        total_predictions = mongo.db.predictions.count_documents({'user_id': user_id})
        depression_cases = mongo.db.predictions.count_documents({
            'user_id': user_id,
            'result.prediction': 1
        })
        
        stats = {
            'total': total_predictions,
            'depression': depression_cases,
            'no_depression': total_predictions - depression_cases,
            'percentage': (depression_cases / total_predictions * 100) if total_predictions > 0 else 0
        }
        
        # Get user info
        user = get_user_profile(mongo, session['user_id'])
        
        return render_template('dashboard.html', stats=stats, user=user)
    except Exception as e:
        return render_template('dashboard.html', stats={}, error=str(e), user=None)


@app.route('/api/stats')
def api_stats():
    """API endpoint for statistics"""
    try:
        total = mongo.db.predictions.count_documents({})
        depression = mongo.db.predictions.count_documents({'result.prediction': 1})
        
        return jsonify({
            'total_predictions': total,
            'depression_cases': depression,
            'no_depression_cases': total - depression,
            'depression_percentage': (depression / total * 100) if total > 0 else 0
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/predict-simple', methods=['POST'])
def predict_simple():
    """Simple prediction API for testing"""
    try:
        data = request.get_json()
        
        # Get selected model
        model_type = data.get('model_type', 'logistic_regression')
        
        if model_type not in models:
            available_models = list(models.keys())
            if not available_models:
                return jsonify({
                    'success': False,
                    'error': 'No models available'
                }), 500
            model_type = available_models[0]
        
        # Extract features
        questionnaire_features = []
        for i in range(31):
            key = f'feature_{i}'
            value = float(data.get(key, 0))
            questionnaire_features.append(value)
        
        # Map to model features
        features_array = map_questionnaire_to_model_features(questionnaire_features)
        
        # Scale features
        if scaler:
            features_scaled = scaler.transform(features_array)
        else:
            features_scaled = features_array
        
        # Make prediction
        selected_model = models[model_type]
        
        if model_type == 'deep_learning':
            prediction_proba = selected_model.predict(features_scaled, verbose=0).flatten()[0]
            prediction = 1 if prediction_proba >= 0.5 else 0
            probability = [1 - prediction_proba, prediction_proba]
        else:
            prediction = selected_model.predict(features_scaled)[0]
            probability = selected_model.predict_proba(features_scaled)[0]
        
        # Format result
        result = {
            'prediction': int(prediction),
            'probability': {
                'no_depression': float(probability[0]),
                'depression': float(probability[1])
            },
            'risk_level': get_risk_level(probability[1]),
            'model_used': model_type
        }
        
        return jsonify({
            'success': True,
            'result': result
        })
        
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 400


@app.route('/test')
def test_page():
    """Simple test page"""
    return render_template('test_prediction.html')


def get_risk_level(probability):
    """Determine risk level based on probability"""
    if probability < 0.3:
        return 'Low'
    elif probability < 0.6:
        return 'Moderate'
    elif probability < 0.8:
        return 'High'
    else:
        return 'Very High'


def save_prediction(input_data, result, user_id=None):
    """Save prediction to MongoDB"""
    if not mongo:
        print("⚠ MongoDB not available - skipping save")
        return
        
    try:
        from bson.objectid import ObjectId
        
        prediction_doc = {
            'input_data': input_data,
            'result': result,
            'timestamp': datetime.now(),
            'model_version': 'v1.0'
        }
        
        # Add user_id if available
        if user_id:
            prediction_doc['user_id'] = ObjectId(user_id)
        
        mongo.db.predictions.insert_one(prediction_doc)
    except Exception as e:
        print(f"Error saving to database: {e}")


# ==================== VOICE-BASED PREDICTION ====================

@app.route('/predict-voice')
@login_required
def predict_voice_page():
    """Voice-based depression prediction page"""
    user = None
    if 'user_id' in session:
        user = get_user_profile(mongo, session['user_id'])
    return render_template('voice_predict.html', user=user)


@app.route('/api/predict-voice', methods=['POST'])
@login_required
def predict_voice_api():
    """API endpoint for voice-based prediction"""
    try:
        # Check if audio file is present
        if 'audio' not in request.files:
            return jsonify({
                'success': False,
                'error': 'No audio file provided'
            }), 400
        
        audio_file = request.files['audio']
        
        if audio_file.filename == '':
            return jsonify({
                'success': False,
                'error': 'No audio file selected'
            }), 400
        
        # Save audio file temporarily
        import tempfile
        import os
        from voice_prediction_api import VoicePredictor
        
        # Create temp file
        temp_dir = tempfile.gettempdir()
        timestamp = datetime.now().timestamp()
        
        # Save uploaded file first
        original_ext = os.path.splitext(audio_file.filename)[1]
        temp_original = os.path.join(temp_dir, f'voice_original_{session["user_id"]}_{timestamp}{original_ext}')
        audio_file.save(temp_original)
        
        # Convert to WAV if needed
        temp_path = os.path.join(temp_dir, f'voice_{session["user_id"]}_{timestamp}.wav')
        
        try:
            # Try to convert using pydub
            from pydub import AudioSegment
            audio = AudioSegment.from_file(temp_original)
            audio = audio.set_channels(1)  # Convert to mono
            audio = audio.set_frame_rate(22050)  # Set sample rate
            audio.export(temp_path, format='wav')
            os.remove(temp_original)  # Clean up original
        except Exception as conv_error:
            print(f"Conversion error: {conv_error}")
            # If conversion fails, try using original file
            if temp_original.endswith('.wav'):
                temp_path = temp_original
            else:
                # Try direct rename
                os.rename(temp_original, temp_path)
        
        try:
            # Predict
            predictor = VoicePredictor()
            result = predictor.predict(temp_path)
            
            # Save to database if MongoDB is available
            if mongo and result['success']:
                try:
                    from bson.objectid import ObjectId
                    
                    prediction_data = {
                        'user_id': ObjectId(session['user_id']),
                        'timestamp': datetime.utcnow(),
                        'type': 'voice',
                        'prediction': result['prediction'],
                        'prediction_label': result['prediction_label'],
                        'confidence': result['confidence'],
                        'risk_level': result['risk_level'],
                        'voice_features': result['voice_features'],
                        'indicators': result['indicators']
                    }
                    
                    mongo.db.predictions.insert_one(prediction_data)
                    
                    # Increment assessment count
                    increment_assessment_count(mongo, session['user_id'])
                    
                except Exception as e:
                    print(f"Error saving to database: {e}")
            
            # Add success message if result is valid
            if result and result.get('success'):
                return jsonify(result)
            else:
                return jsonify({
                    'success': False,
                    'error': 'Failed to analyze audio. Please try recording again or upload a different file.'
                }), 400
            
        finally:
            # Clean up temp files
            if os.path.exists(temp_path):
                try:
                    os.remove(temp_path)
                except:
                    pass
            if 'temp_original' in locals() and os.path.exists(temp_original):
                try:
                    os.remove(temp_original)
                except:
                    pass
    
    except Exception as e:
        print(f"Error in voice prediction: {e}")
        import traceback
        traceback.print_exc()
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500


if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)

