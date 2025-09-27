"""
Configuration settings for MoSPI AI-Enhanced Survey Analysis
"""

# Missing value indicators that should be recognized as NaN
MISSING_VALUES = [
    '?', 'NA', 'N/A', 'null', 'NULL', 'none', 'None', 'NONE',
    '--', '-', '', ' ', 'missing', 'Missing', 'MISSING',
    'unknown', 'Unknown', 'UNKNOWN', 'n/a', 'N/a',
    '#N/A', '#NA', '#NULL!', '#DIV/0!', '#VALUE!', '#REF!'
]

# Survey types supported
SURVEY_TYPES = [
    "Household Survey",
    "Enterprise Survey", 
    "Agriculture Survey",
    "Labour Force Survey",
    "Consumer Expenditure Survey",
    "Annual Survey of Industries",
    "Other"
]

# Sampling methods
SAMPLING_METHODS = [
    "Simple Random",
    "Stratified",
    "Cluster", 
    "Systematic",
    "Multi-stage",
    "Probability Proportional to Size"
]

# AI processing configuration
AI_CONFIG = {
    'text_generation': {
        'min_confidence': 0.6,
        'max_iterations': 100,
        'context_window': 5
    },
    'numerical_processing': {
        'outlier_threshold': 3.0,
        'missing_threshold': 0.3,
        'correlation_threshold': 0.8
    },
    'model_performance': {
        'min_r2_score': 0.5,
        'cv_folds': 5,
        'ensemble_weights': {
            'random_forest': 0.6,
            'gradient_boosting': 0.4
        }
    }
}

# Report generation settings
REPORT_CONFIG = {
    'max_insights': 10,
    'confidence_threshold': 0.7,
    'quality_thresholds': {
        'excellent': 0.9,
        'good': 0.8,
        'acceptable': 0.6,
        'poor': 0.4
    }
}

# File upload settings
FILE_CONFIG = {
    'max_file_size_mb': 100,
    'allowed_extensions': ['.csv', '.xlsx', '.xls'],
    'chunk_size': 10000
}

# Visualization settings
VIZ_CONFIG = {
    'color_palette': ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd'],
    'default_height': 400,
    'default_width': 600
}
