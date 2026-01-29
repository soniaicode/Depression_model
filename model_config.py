"""
Central Model Configuration
Single source of truth for all model performance metrics
"""

# Model Performance Metrics (Actual Trained Models)
MODEL_PERFORMANCE = {
    'deep_learning': {
        'name': 'Enhanced Multimodal',
        'short_name': 'Enhanced Multimodal',
        'accuracy': 88.9,
        'precision': 85.7,
        'recall': 84.2,
        'f1_score': 84.9,
        'auc_roc': 0.921,
        'specificity': 93.6,
        'icon': '🧠',
        'category': 'Proposed Models',
        'badge': '🏆 Best Performance',
        'description': 'Advanced multimodal deep learning with attention mechanism combining questionnaire and physiological data',
        'highlight': True
    },
    'random_forest': {
        'name': 'Random Forest',
        'short_name': 'Random Forest',
        'accuracy': 85.3,
        'precision': 81.1,
        'recall': 78.9,
        'f1_score': 80.0,
        'auc_roc': 0.892,
        'specificity': 91.7,
        'icon': '🌲',
        'category': 'Conventional ML',
        'badge': 'High Accuracy',
        'description': 'Ensemble learning with decision trees - reliable and interpretable',
        'highlight': False
    },
    'logistic_regression': {
        'name': 'Logistic Regression',
        'short_name': 'Logistic Regression',
        'accuracy': 78.0,
        'precision': 75.2,
        'recall': 73.5,
        'f1_score': 74.3,
        'auc_roc': 0.832,
        'specificity': 82.5,
        'icon': '📊',
        'category': 'Traditional Statistical',
        'badge': 'Baseline Model',
        'description': 'Traditional statistical baseline model',
        'highlight': False
    },
    'gradient_boosting': {
        'name': 'Gradient Boosting',
        'short_name': 'Gradient Boosting',
        'accuracy': 74.2,
        'precision': 77.3,
        'recall': 80.7,
        'f1_score': 79.0,
        'auc_roc': 0.817,
        'specificity': 67.7,
        'icon': '⚡',
        'category': 'Conventional ML',
        'badge': 'Fast & Efficient',
        'description': 'Gradient boosting framework - fast and efficient',
        'highlight': False
    }
}

# Model display order (best first)
MODEL_ORDER = ['deep_learning', 'random_forest', 'logistic_regression', 'gradient_boosting']

def get_model_info(model_key):
    """Get model information by key"""
    return MODEL_PERFORMANCE.get(model_key, {})

def get_all_models():
    """Get all models in display order"""
    return {key: MODEL_PERFORMANCE[key] for key in MODEL_ORDER if key in MODEL_PERFORMANCE}

def get_model_count():
    """Get total number of models"""
    return len(MODEL_PERFORMANCE)

def get_best_model():
    """Get the best performing model"""
    return 'deep_learning', MODEL_PERFORMANCE['deep_learning']
