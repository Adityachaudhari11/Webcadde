import pandas as pd
import numpy as np
from typing import Dict, List, Any
import config

def validate_data(df: pd.DataFrame) -> Dict[str, Any]:
    """Comprehensive data validation with AI insights"""
    validation_report = {
        'total_records': len(df),
        'total_variables': len(df.columns),
        'missing_data': {
            'total_missing': df.isnull().sum().sum(),
            'missing_percentage': (df.isnull().sum().sum() / (len(df) * len(df.columns))) * 100,
            'columns_with_missing': df.columns[df.isnull().any()].tolist()
        },
        'data_types': {
            'numeric': len(df.select_dtypes(include=[np.number]).columns),
            'categorical': len(df.select_dtypes(exclude=[np.number]).columns),
            'datetime': len(df.select_dtypes(include=['datetime']).columns)
        },
        'duplicates': df.duplicated().sum(),
        'memory_usage': f"{df.memory_usage(deep=True).sum() / 1024**2:.2f} MB"
    }
    
    # Enhanced validation checks
    issues = []
    recommendations = []
    
    # Missing data analysis
    if validation_report['missing_data']['missing_percentage'] > 30:
        issues.append("Very high percentage of missing data (>30%)")
        recommendations.append("Consider AI-powered missing value imputation")
    elif validation_report['missing_data']['missing_percentage'] > 10:
        issues.append("High percentage of missing data (>10%)")
        recommendations.append("Use hybrid AI processing for optimal results")
    
    # Duplicate analysis
    if validation_report['duplicates'] > 0:
        issues.append(f"{validation_report['duplicates']} duplicate records found")
        recommendations.append("Remove duplicates before processing")
    
    # Data type balance
    if validation_report['data_types']['numeric'] == 0:
        issues.append("No numeric variables for statistical analysis")
    elif validation_report['data_types']['categorical'] == 0:
        issues.append("No categorical variables for stratification")
    
    # Sample size assessment
    if validation_report['total_records'] < 30:
        issues.append("Very small sample size (<30)")
        recommendations.append("Results may have limited statistical power")
    elif validation_report['total_records'] < 100:
        issues.append("Small sample size (<100)")
        recommendations.append("Interpret confidence intervals carefully")
    
    # Memory usage warning
    memory_mb = df.memory_usage(deep=True).sum() / 1024**2
    if memory_mb > config.FILE_CONFIG['max_file_size_mb']:
        issues.append(f"Large file size ({memory_mb:.1f} MB)")
        recommendations.append("Consider data chunking for processing")
    
    # AI processing recommendations
    if validation_report['missing_data']['missing_percentage'] > 5:
        recommendations.append("AI hybrid processing recommended for missing data")
    
    if validation_report['data_types']['categorical'] > 0:
        recommendations.append("AI text generation available for categorical variables")
    
    validation_report['issues'] = issues
    validation_report['recommendations'] = recommendations
    validation_report['quality_score'] = calculate_quality_score(validation_report)
    validation_report['ai_processing_suitable'] = is_suitable_for_ai_processing(validation_report)
    
    return validation_report

def calculate_quality_score(validation_report: Dict) -> float:
    """Calculate enhanced data quality score (0-100)"""
    score = 100
    
    # Deduct for missing data (more sophisticated)
    missing_pct = validation_report['missing_data']['missing_percentage']
    if missing_pct > 0:
        # Non-linear penalty for missing data
        missing_penalty = min(40, missing_pct * 1.5 + (missing_pct ** 1.5) * 0.1)
        score -= missing_penalty
    
    # Deduct for duplicates
    total_records = validation_report['total_records']
    if validation_report['duplicates'] > 0:
        duplicate_penalty = min(20, (validation_report['duplicates'] / total_records) * 100)
        score -= duplicate_penalty
    
    # Deduct for small sample size
    if total_records < 30:
        score -= 20
    elif total_records < 100:
        score -= 10
    
    # Deduct for data type imbalance
    numeric_count = validation_report['data_types']['numeric']
    categorical_count = validation_report['data_types']['categorical']
    
    if numeric_count == 0 or categorical_count == 0:
        score -= 15
    
    # Bonus for good data characteristics
    if missing_pct == 0:
        score += 5  # Perfect completeness bonus
    
    if validation_report['duplicates'] == 0:
        score += 3  # No duplicates bonus
    
    return max(0, min(100, score))

def is_suitable_for_ai_processing(validation_report: Dict) -> bool:
    """Determine if data is suitable for AI processing"""
    criteria = []
    
    # Sample size criterion
    criteria.append(validation_report['total_records'] >= 10)
    
    # Missing data criterion (AI can handle missing data)
    criteria.append(validation_report['missing_data']['missing_percentage'] <= 90)
    
    # Data type diversity
    has_numeric = validation_report['data_types']['numeric'] > 0
    has_categorical = validation_report['data_types']['categorical'] > 0
    criteria.append(has_numeric or has_categorical)
    
    # Memory usage criterion
    memory_mb = float(validation_report['memory_usage'].replace(' MB', ''))
    criteria.append(memory_mb <= config.FILE_CONFIG['max_file_size_mb'] * 2)  # Allow 2x limit for AI processing
    
    return all(criteria)

def get_survey_metadata(df: pd.DataFrame) -> Dict[str, Any]:
    """Extract comprehensive survey metadata from dataframe"""
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    categorical_cols = df.select_dtypes(exclude=[np.number]).columns
    
    metadata = {
        'basic_info': {
            'sample_size': len(df),
            'total_variables': len(df.columns),
            'numeric_variables': len(numeric_cols),
            'categorical_variables': len(categorical_cols)
        },
        'variables': {
            'numeric': list(numeric_cols),
            'categorical': list(categorical_cols)
        },
        'data_characteristics': {
            'memory_usage_mb': df.memory_usage(deep=True).sum() / 1024**2,
            'completeness_rate': (1 - df.isnull().sum().sum() / (len(df) * len(df.columns))),
            'duplicate_rate': df.duplicated().sum() / len(df) if len(df) > 0 else 0
        },
        'statistical_summary': {},
        'categorical_summary': {}
    }
    
    # Statistical summary for numeric variables
    if len(numeric_cols) > 0:
        metadata['statistical_summary'] = df[numeric_cols].describe().to_dict()
    
    # Categorical summary
    if len(categorical_cols) > 0:
        categorical_summary = {}
        for col in categorical_cols:
            categorical_summary[col] = {
                'unique_values': df[col].nunique(),
                'most_frequent': df[col].mode().iloc[0] if len(df[col].mode()) > 0 else None,
                'missing_count': df[col].isnull().sum()
            }
        metadata['categorical_summary'] = categorical_summary
    
    # Date range analysis (if datetime columns exist)
    datetime_cols = df.select_dtypes(include=['datetime']).columns
    if len(datetime_cols) > 0:
        metadata['date_range'] = {}
        for col in datetime_cols:
            metadata['date_range'][col] = {
                'start': df[col].min(),
                'end': df[col].max(),
                'span_days': (df[col].max() - df[col].min()).days if pd.notna(df[col].min()) else None
            }
    
    return metadata

def format_number(number: float, decimal_places: int = 2) -> str:
    """Format numbers for display with intelligent rounding"""
    if pd.isna(number):
        return "N/A"
    
    abs_number = abs(number)
    
    if abs_number >= 1e9:
        return f"{number/1e9:.{decimal_places}f}B"
    elif abs_number >= 1e6:
        return f"{number/1e6:.{decimal_places}f}M"
    elif abs_number >= 1e3:
        return f"{number/1e3:.{decimal_places}f}K"
    else:
        return f"{number:.{decimal_places}f}"

def calculate_sampling_error(estimate: float, se: float, confidence_level: float = 0.95, 
                           sample_size: int = None, design_effect: float = 1.0) -> Dict[str, float]:
    """Calculate comprehensive sampling error metrics - FIXED VERSION"""
    from scipy import stats
    
    alpha = 1 - confidence_level
    z_score = stats.norm.ppf(1 - alpha/2)
    
    margin_of_error = z_score * se
    cv = (se / estimate) * 100 if estimate != 0 else float('inf')
    rse = (se / estimate) * 100 if estimate != 0 else float('inf')
    
    # Quality classification based on CV
    if cv < 10:
        quality = "Excellent"
    elif cv < 20:
        quality = "Good"
    elif cv < 30:
        quality = "Acceptable"
    else:
        quality = "Poor"
    
    # Calculate effective sample size if sample_size is provided
    effective_sample_size = None
    if sample_size is not None:
        effective_sample_size = sample_size / design_effect
    
    return {
        'margin_of_error': margin_of_error,
        'coefficient_of_variation': cv,
        'relative_standard_error': rse,
        'quality_rating': quality,
        'confidence_interval': {
            'lower': estimate - margin_of_error,
            'upper': estimate + margin_of_error
        },
        'design_effect': design_effect,
        'effective_sample_size': effective_sample_size
    }

def generate_data_profile(df: pd.DataFrame) -> Dict[str, Any]:
    """Generate comprehensive data profile for AI processing"""
    profile = {
        'data_shape': {
            'rows': len(df),
            'columns': len(df.columns)
        },
        'column_analysis': {},
        'data_quality_indicators': {},
        'ai_processing_recommendations': []
    }
    
    # Analyze each column
    for col in df.columns:
        col_analysis = {
            'data_type': str(df[col].dtype),
            'missing_count': df[col].isnull().sum(),
            'missing_percentage': (df[col].isnull().sum() / len(df)) * 100,
            'unique_values': df[col].nunique(),
            'memory_usage_bytes': df[col].memory_usage(deep=True)
        }
        
        if df[col].dtype in ['int64', 'float64']:
            col_analysis.update({
                'mean': df[col].mean(),
                'median': df[col].median(),
                'std': df[col].std(),
                'min': df[col].min(),
                'max': df[col].max(),
                'skewness': df[col].skew(),
                'kurtosis': df[col].kurtosis()
            })
            
            # Recommend processing method
            if col_analysis['missing_percentage'] > 10:
                profile['ai_processing_recommendations'].append(f"Use ML-based imputation for {col}")
            
        else:  # Categorical
            col_analysis.update({
                'most_frequent': df[col].mode().iloc[0] if len(df[col].mode()) > 0 else None,
                'frequency_of_most_frequent': df[col].value_counts().iloc[0] if len(df[col].value_counts()) > 0 else 0
            })
            
            # Recommend processing method
            if col_analysis['missing_percentage'] > 5:
                profile['ai_processing_recommendations'].append(f"Use AI text generation for {col}")
        
        profile['column_analysis'][col] = col_analysis
    
    # Overall data quality indicators
    total_missing = df.isnull().sum().sum()
    total_cells = len(df) * len(df.columns)
    
    profile['data_quality_indicators'] = {
        'overall_completeness': 1 - (total_missing / total_cells),
        'columns_with_missing': len([col for col in df.columns if df[col].isnull().any()]),
        'high_missing_columns': len([col for col in df.columns if (df[col].isnull().sum() / len(df)) > 0.3]),
        'duplicate_rows': df.duplicated().sum(),
        'memory_efficiency_score': calculate_memory_efficiency(df)
    }
    
    return profile

def calculate_memory_efficiency(df: pd.DataFrame) -> float:
    """Calculate memory efficiency score (0-1)"""
    current_memory = df.memory_usage(deep=True).sum()
    
    # Estimate optimal memory usage
    optimal_memory = 0
    for col in df.columns:
        if df[col].dtype == 'object':
            # String columns - estimate based on average length
            avg_length = df[col].astype(str).str.len().mean()
            optimal_memory += len(df) * min(avg_length, 50)  # Cap at 50 chars
        elif df[col].dtype == 'int64':
            # Check if can be downcasted
            if df[col].max() <= 127 and df[col].min() >= -128:
                optimal_memory += len(df) * 1  # int8
            elif df[col].max() <= 32767 and df[col].min() >= -32768:
                optimal_memory += len(df) * 2  # int16
            else:
                optimal_memory += len(df) * 8  # int64
        else:
            optimal_memory += len(df) * 8  # Default 8 bytes
    
    efficiency = optimal_memory / current_memory if current_memory > 0 else 1.0
    return min(1.0, efficiency)

def detect_outliers_iqr(series: pd.Series, multiplier: float = 1.5) -> pd.Series:
    """Detect outliers using IQR method"""
    if series.dtype not in ['int64', 'float64']:
        return pd.Series([False] * len(series), index=series.index)
    
    Q1 = series.quantile(0.25)
    Q3 = series.quantile(0.75)
    IQR = Q3 - Q1
    
    lower_bound = Q1 - multiplier * IQR
    upper_bound = Q3 + multiplier * IQR
    
    return (series < lower_bound) | (series > upper_bound)

def suggest_ai_processing_strategy(df: pd.DataFrame) -> Dict[str, str]:
    """Suggest AI processing strategy based on data characteristics"""
    strategy = {
        'overall_approach': 'hybrid',
        'missing_value_strategy': {},
        'outlier_detection': 'ai_enhanced_iqr',
        'feature_engineering': [],
        'model_recommendations': []
    }
    
    # Analyze missing value strategy for each column
    for col in df.columns:
        missing_pct = (df[col].isnull().sum() / len(df)) * 100
        
        if missing_pct == 0:
            strategy['missing_value_strategy'][col] = 'no_action_needed'
        elif df[col].dtype in ['int64', 'float64']:
            if missing_pct < 10:
                strategy['missing_value_strategy'][col] = 'smart_mean_median'
            elif missing_pct < 30:
                strategy['missing_value_strategy'][col] = 'ml_based_prediction'
            else:
                strategy['missing_value_strategy'][col] = 'hybrid_statistical'
        else:  # Categorical
            if missing_pct < 15:
                strategy['missing_value_strategy'][col] = 'context_aware_mode'
            else:
                strategy['missing_value_strategy'][col] = 'ai_text_generation'
    
    # Feature engineering recommendations
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    if len(numeric_cols) > 1:
        strategy['feature_engineering'].append('correlation_analysis')
        strategy['feature_engineering'].append('interaction_terms')
    
    if len(df) > 100:
        strategy['feature_engineering'].append('polynomial_features')
    
    # Model recommendations
    if len(df) < 100:
        strategy['model_recommendations'].append('simple_models_recommended')
    elif len(df) < 1000:
        strategy['model_recommendations'].append('ensemble_methods_suitable')
    else:
        strategy['model_recommendations'].append('deep_learning_feasible')
    
    return strategy

def validate_survey_weights(weights: np.ndarray) -> Dict[str, Any]:
    """Validate survey weights and provide diagnostics"""
    validation = {
        'is_valid': True,
        'issues': [],
        'statistics': {},
        'recommendations': []
    }
    
    if len(weights) == 0:
        validation['is_valid'] = False
        validation['issues'].append("No weights provided")
        return validation
    
    # Calculate statistics
    validation['statistics'] = {
        'count': len(weights),
        'mean': np.mean(weights),
        'median': np.median(weights),
        'std': np.std(weights),
        'min': np.min(weights),
        'max': np.max(weights),
        'cv': np.std(weights) / np.mean(weights) if np.mean(weights) != 0 else float('inf')
    }
    
    # Validation checks
    if np.any(weights <= 0):
        validation['is_valid'] = False
        validation['issues'].append("Negative or zero weights found")
    
    if np.any(np.isnan(weights)):
        validation['is_valid'] = False
        validation['issues'].append("NaN values in weights")
    
    if np.any(np.isinf(weights)):
        validation['is_valid'] = False
        validation['issues'].append("Infinite values in weights")
    
    # Quality checks
    cv = validation['statistics']['cv']
    if cv > 2.0:
        validation['issues'].append(f"High coefficient of variation in weights ({cv:.2f})")
        validation['recommendations'].append("Consider weight trimming or capping")
    
    if validation['statistics']['max'] / validation['statistics']['min'] > 10:
        validation['issues'].append("Extreme weight ratio detected")
        validation['recommendations'].append("Review sampling design and weight calculation")
    
    return validation

def calculate_design_effects(df: pd.DataFrame, target_vars: List[str], 
                           weight_col: str = 'survey_weight') -> Dict[str, float]:
    """Calculate design effects for target variables"""
    design_effects = {}
    
    if weight_col not in df.columns:
        # If no weights, assume simple random sampling (DEFF = 1)
        return {var: 1.0 for var in target_vars}
    
    weights = df[weight_col]
    
    for var in target_vars:
        if var not in df.columns:
            continue
            
        # Calculate design effect using the formula:
        # DEFF = 1 + CV²(weights) * ICC
        # Simplified version: DEFF ≈ 1 + CV²(weights)
        
        if weights.std() != 0:
            cv_weights = weights.std() / weights.mean()
            deff = 1 + cv_weights**2
        else:
            deff = 1.0
        
        design_effects[var] = deff
    
    return design_effects

def generate_quality_report(df: pd.DataFrame) -> str:
    """Generate a comprehensive data quality report"""
    report = []
    
    # Header
    report.append("=" * 60)
    report.append("DATA QUALITY ASSESSMENT REPORT")
    report.append("=" * 60)
    report.append("")
    
    # Basic info
    report.append(f"Dataset Shape: {df.shape[0]:,} rows × {df.shape[1]} columns")
    report.append(f"Memory Usage: {df.memory_usage(deep=True).sum() / 1024**2:.2f} MB")
    report.append("")
    
    # Missing data analysis
    missing_data = df.isnull().sum()
    total_missing = missing_data.sum()
    missing_pct = (total_missing / (len(df) * len(df.columns))) * 100
    
    report.append("MISSING DATA ANALYSIS:")
    report.append("-" * 25)
    report.append(f"Total missing values: {total_missing:,} ({missing_pct:.2f}%)")
    
    if total_missing > 0:
        report.append("\nColumns with missing data:")
        for col, count in missing_data[missing_data > 0].items():
            pct = (count / len(df)) * 100
            report.append(f"  • {col}: {count:,} ({pct:.1f}%)")
    else:
        report.append("✓ No missing data detected")
    
    report.append("")
    
    # Data types
    report.append("DATA TYPE ANALYSIS:")
    report.append("-" * 20)
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    categorical_cols = df.select_dtypes(exclude=[np.number]).columns
    
    report.append(f"Numeric variables: {len(numeric_cols)}")
    report.append(f"Categorical variables: {len(categorical_cols)}")
    report.append("")
    
    # Duplicates
    duplicates = df.duplicated().sum()
    report.append(f"DUPLICATE RECORDS: {duplicates:,}")
    if duplicates > 0:
        report.append("⚠ Consider removing duplicate records")
    else:
        report.append("✓ No duplicate records found")
    
    report.append("")
    
    # Quality score
    validation_report = validate_data(df)
    quality_score = validation_report['quality_score']
    
    report.append("OVERALL QUALITY ASSESSMENT:")
    report.append("-" * 30)
    report.append(f"Quality Score: {quality_score:.1f}/100")
    
    if quality_score >= 90:
        report.append("✓ EXCELLENT - Data is ready for analysis")
    elif quality_score >= 80:
        report.append("✓ GOOD - Minor issues, suitable for analysis")
    elif quality_score >= 60:
        report.append("⚠ ACCEPTABLE - Some data quality issues present")
    else:
        report.append("⚠ POOR - Significant data quality issues require attention")
    
    report.append("")
    report.append("=" * 60)
    
    return "\n".join(report)
