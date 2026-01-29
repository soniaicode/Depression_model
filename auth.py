"""
Authentication and User Management Module
"""

from flask import session, redirect, url_for, flash
from werkzeug.security import generate_password_hash, check_password_hash
from functools import wraps
from datetime import datetime
import re
from flask_session import Session

def login_required(f):
    """Decorator to require login for routes"""
    @wraps(f)
    def decorated_function(*args, **kwargs):
        if 'user_id' not in session:
            flash('Please login to access this page.', 'warning')
            return redirect(url_for('login'))
        return f(*args, **kwargs)
    return decorated_function

def admin_required(f):
    """Decorator to require admin role for routes"""
    @wraps(f)
    def decorated_function(*args, **kwargs):
        # Check if admin is logged in (either through admin login or user login with admin privileges)
        if not session.get('is_admin', False):
            flash('Please login as admin to access this page.', 'warning')
            return redirect(url_for('admin_login'))
        
        return f(*args, **kwargs)
    return decorated_function

def validate_email(email):
    """Validate email format"""
    pattern = r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$'
    return re.match(pattern, email) is not None

def validate_password(password):
    """
    Validate password strength
    - At least 8 characters
    - Contains uppercase and lowercase
    - Contains number
    """
    if len(password) < 8:
        return False, "Password must be at least 8 characters"
    if not re.search(r'[A-Z]', password):
        return False, "Password must contain uppercase letter"
    if not re.search(r'[a-z]', password):
        return False, "Password must contain lowercase letter"
    if not re.search(r'\d', password):
        return False, "Password must contain number"
    return True, "Password is valid"

def create_user(mongo, user_data):
    """
    Create new user in database
    
    Args:
        mongo: MongoDB instance
        user_data: Dictionary with user information
        
    Returns:
        tuple: (success, message, user_id)
    """
    try:
        # Check if MongoDB is available
        if not mongo:
            return False, "Database not available. Please contact administrator.", None
        
        # Check if email already exists
        existing_user = mongo.db.users.find_one({'email': user_data['email']})
        if existing_user:
            return False, "Email already registered", None
        
        # Validate email
        if not validate_email(user_data['email']):
            return False, "Invalid email format", None
        
        # Validate password
        is_valid, message = validate_password(user_data['password'])
        if not is_valid:
            return False, message, None
        
        # Hash password
        hashed_password = generate_password_hash(user_data['password'])
        
        # Create user document
        user_doc = {
            'email': user_data['email'],
            'password': hashed_password,
            'full_name': user_data['full_name'],
            'age': int(user_data['age']),
            'gender': user_data['gender'],
            'phone': user_data.get('phone', ''),
            
            # Demographic data for model enhancement
            'education_years': int(user_data.get('education_years', 14)),
            'income_level': int(user_data.get('income_level', 2)),
            'marital_status': int(user_data.get('marital_status', 0)),
            'employment_status': int(user_data.get('employment_status', 1)),
            'residence_type': int(user_data.get('residence_type', 1)),
            
            # Health information
            'chronic_conditions': user_data.get('chronic_conditions', []),
            'medications': user_data.get('medications', []),
            'family_history_depression': int(user_data.get('family_history_depression', 0)),
            
            # Account metadata
            'created_at': datetime.now(),
            'last_login': datetime.now(),
            'is_active': True,
            'profile_complete': True,
            'total_assessments': 0
        }
        
        # Insert user
        result = mongo.db.users.insert_one(user_doc)
        
        return True, "Account created successfully", str(result.inserted_id)
        
    except Exception as e:
        return False, f"Error creating account: {str(e)}", None

def authenticate_user(mongo, email, password):
    """
    Authenticate user login
    
    Args:
        mongo: MongoDB instance
        email: User email
        password: User password
        
    Returns:
        tuple: (success, message, user_data)
    """
    try:
        # Check if MongoDB is available
        if not mongo:
            return False, "Database not available. Please contact administrator.", None
        
        # Find user
        user = mongo.db.users.find_one({'email': email})
        
        if not user:
            return False, "Invalid email or password", None
        
        # Check password
        if not check_password_hash(user['password'], password):
            return False, "Invalid email or password", None
        
        # Check if account is active
        if not user.get('is_active', True):
            return False, "Account is deactivated", None
        
        # Update last login
        mongo.db.users.update_one(
            {'_id': user['_id']},
            {'$set': {'last_login': datetime.now()}}
        )
        
        # Remove password from user data
        user.pop('password', None)
        user['_id'] = str(user['_id'])
        
        return True, "Login successful", user
        
    except Exception as e:
        return False, f"Error during login: {str(e)}", None

def get_user_profile(mongo, user_id):
    """Get user profile data"""
    if not mongo:
        return None
    try:
        from bson.objectid import ObjectId
        user = mongo.db.users.find_one({'_id': ObjectId(user_id)})
        if user:
            user.pop('password', None)
            user['_id'] = str(user['_id'])
            return user
        return None
    except Exception as e:
        print(f"Error getting user profile: {e}")
        return None

def update_user_profile(mongo, user_id, update_data):
    """Update user profile"""
    try:
        from bson.objectid import ObjectId
        
        # Remove sensitive fields
        update_data.pop('password', None)
        update_data.pop('email', None)
        update_data.pop('_id', None)
        
        # Update user
        result = mongo.db.users.update_one(
            {'_id': ObjectId(user_id)},
            {'$set': update_data}
        )
        
        return result.modified_count > 0
        
    except Exception as e:
        print(f"Error updating profile: {e}")
        return False

def get_user_demographic_data(mongo, user_id):
    """
    Get user demographic data for model enhancement
    
    Returns:
        dict: Demographic data to enhance predictions
    """
    if not mongo:
        return {}
    try:
        user = get_user_profile(mongo, user_id)
        if not user:
            return {}
        
        return {
            'age': user.get('age', 35),
            'education_years': user.get('education_years', 14),
            'income_level': user.get('income_level', 2),
            'marital_status': user.get('marital_status', 0),
            'employment_status': user.get('employment_status', 1),
            'residence_type': user.get('residence_type', 1),
            'family_history_depression': user.get('family_history_depression', 0),
            'chronic_conditions': user.get('chronic_conditions', []),
        }
        
    except Exception as e:
        print(f"Error getting demographic data: {e}")
        return {}

def increment_assessment_count(mongo, user_id):
    """Increment user's total assessment count"""
    if not mongo:
        return
    try:
        from bson.objectid import ObjectId
        mongo.db.users.update_one(
            {'_id': ObjectId(user_id)},
            {'$inc': {'total_assessments': 1}}
        )
    except Exception as e:
        print(f"Error incrementing assessment count: {e}")
