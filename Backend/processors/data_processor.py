import pandas as pd
import numpy as np
from sklearn.impute import SimpleImputer, KNNImputer
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler
from sklearn.ensemble import IsolationForest, RandomForestClassifier, RandomForestRegressor
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

class DataProcessor:
    def __init__(self):
        self.scaler = None
        self.imputer = None
        
    def clean_data_with_report(self, df, missing_strategy='drop', outlier_method='iqr', normalization='none'):
        """Comprehensive data cleaning pipeline with detailed reporting"""
        df_clean = df.copy()
        
        # Track original state
        original_shape = df_clean.shape
        original_missing = df_clean.isnull().sum().to_dict()
        
        # Handle missing values and track changes
        df_clean, missing_report = self._handle_missing_values_with_report(df_clean, missing_strategy)
        
        # Handle outliers
        if outlier_method != 'none':
            df_clean = self._handle_outliers(df_clean, outlier_method)
        
        # Normalization
        if normalization != 'none':
            df_clean = self._normalize_data(df_clean, normalization)
        
        # Create comprehensive report
        final_missing = df_clean.isnull().sum().to_dict()
        
        detailed_report = {
            'original_shape': original_shape,
            'final_shape': df_clean.shape,
            'original_missing': original_missing,
            'final_missing': final_missing,
            'missing_strategy': missing_strategy,
            'outlier_method': outlier_method,
            'normalization': normalization,
            'missing_details': missing_report
        }
        
        return df_clean, detailed_report
    
    def clean_data(self, df, missing_strategy='drop', outlier_method='iqr', normalization='none'):
        """Original clean_data method for backward compatibility"""
        df_clean, _ = self.clean_data_with_report(df, missing_strategy, outlier_method, normalization)
        return df_clean
    
    def _handle_missing_values_with_report(self, df, strategy):
        """Handle missing values with detailed reporting"""
        missing_report = {}
        
        if strategy == 'drop':
            original_len = len(df)
            df_clean = df.dropna()
            missing_report['rows_dropped'] = original_len - len(df_clean)
            missing_report['method'] = 'Dropped rows with missing values'
            return df_clean, missing_report
        
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        categorical_cols = df.select_dtypes(exclude=[np.number]).columns
        
        # Handle numeric columns
        for col in numeric_cols:
            if df[col].isnull().any():
                missing_count = df[col].isnull().sum()
                
                if strategy == 'impute_mean':
                    fill_value = df[col].mean()
                    df[col] = df[col].fillna(fill_value)
                    missing_report[col] = {
                        'count_filled': missing_count,
                        'method': 'Mean imputation',
                        'fill_value': fill_value,
                        'data_type': 'numeric'
                    }
                elif strategy == 'impute_median':
                    fill_value = df[col].median()
                    df[col] = df[col].fillna(fill_value)
                    missing_report[col] = {
                        'count_filled': missing_count,
                        'method': 'Median imputation',
                        'fill_value': fill_value,
                        'data_type': 'numeric'
                    }
                elif strategy == 'forward_fill':
                    df[col] = df[col].fillna(method='ffill')
                    missing_report[col] = {
                        'count_filled': missing_count,
                        'method': 'Forward fill',
                        'fill_value': 'Previous valid value',
                        'data_type': 'numeric'
                    }
        
        # Handle categorical columns
        for col in categorical_cols:
            if df[col].isnull().any():
                missing_count = df[col].isnull().sum()
                mode_values = df[col].mode()
                
                if len(mode_values) > 0:
                    fill_value = mode_values.iloc[0]
                    df[col] = df[col].fillna(fill_value)
                    missing_report[col] = {
                        'count_filled': missing_count,
                        'method': 'Mode imputation',
                        'fill_value': fill_value,
                        'data_type': 'categorical'
                    }
                else:
                    fill_value = 'Unknown'
                    df[col] = df[col].fillna(fill_value)
                    missing_report[col] = {
                        'count_filled': missing_count,
                        'method': 'Default value',
                        'fill_value': fill_value,
                        'data_type': 'categorical'
                    }
        
        return df, missing_report
    
    def _handle_missing_values(self, df, strategy):
        """Original method for backward compatibility"""
        df_clean, _ = self._handle_missing_values_with_report(df, strategy)
        return df_clean
    
    def _handle_outliers(self, df, method):
        """Detect and handle outliers with enhanced methods"""
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        
        if len(numeric_cols) == 0:
            return df
        
        if method == 'iqr':
            for col in numeric_cols:
                if df[col].isnull().all():
                    continue
                Q1 = df[col].quantile(0.25)
                Q3 = df[col].quantile(0.75)
                IQR = Q3 - Q1
                lower = Q1 - 1.5 * IQR
                upper = Q3 + 1.5 * IQR
                df[col] = df[col].clip(lower, upper)
        
        elif method == 'z_score':
            for col in numeric_cols:
                if df[col].isnull().all():
                    continue
                z_scores = np.abs(stats.zscore(df[col], nan_policy='omit'))
                mask = (z_scores < 3) | np.isnan(z_scores)
                df = df[mask]
        
        elif method == 'isolation_forest':
            non_null_data = df[numeric_cols].dropna()
            if len(non_null_data) > 10:
                iso_forest = IsolationForest(contamination=0.1, random_state=42)
                outlier_predictions = iso_forest.fit_predict(non_null_data)
                non_outlier_indices = non_null_data.index[outlier_predictions == 1]
                df = df.loc[non_outlier_indices]
        
        return df
    
    def _normalize_data(self, df, method):
        """Normalize numeric data with enhanced options"""
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        
        if len(numeric_cols) == 0:
            return df
        
        valid_numeric_cols = []
        for col in numeric_cols:
            if not df[col].isnull().all():
                valid_numeric_cols.append(col)
        
        if len(valid_numeric_cols) == 0:
            return df
        
        if method == 'min_max':
            self.scaler = MinMaxScaler()
        elif method == 'standard':
            self.scaler = StandardScaler()
        elif method == 'robust':
            self.scaler = RobustScaler()
        
        df[valid_numeric_cols] = self.scaler.fit_transform(df[valid_numeric_cols])
        return df
    
    def calculate_weights(self, df, survey_config):
        """Calculate survey weights based on sampling design"""
        n_sample = len(df)
        N_population = survey_config.get('population', 10000)
        
        basic_weight = N_population / n_sample
        
        if 'sampling' in survey_config and survey_config['sampling'] == 'Stratified':
            weights = np.full(n_sample, basic_weight)
            np.random.seed(42)
            weights += np.random.normal(0, basic_weight * 0.1, n_sample)
            weights = np.maximum(weights, basic_weight * 0.5)
        else:
            weights = np.full(n_sample, basic_weight)
        
        return weights
    
    def validate_weights(self, weights):
        """Enhanced weight validation"""
        if len(weights) == 0:
            return {
                'mean_weight': 0,
                'weight_variance': 0,
                'min_weight': 0,
                'max_weight': 0,
                'cv_weights': 0
            }
        
        return {
            'mean_weight': np.mean(weights),
            'weight_variance': np.var(weights),
            'min_weight': np.min(weights),
            'max_weight': np.max(weights),
            'cv_weights': np.std(weights) / np.mean(weights) if np.mean(weights) != 0 else 0
        }
