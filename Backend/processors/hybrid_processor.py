import pandas as pd
import numpy as np
from .data_processor import DataProcessor
from .ai_text_processor import AITextProcessor
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
import warnings
warnings.filterwarnings('ignore')

class HybridProcessor:
    def __init__(self):
        self.data_processor = DataProcessor()
        self.ai_text_processor = AITextProcessor()
        
    def process_with_ai(self, df, config):
        """Main AI hybrid processing pipeline"""
        df_processed = df.copy()
        processing_report = {
            'ai_insights': [],
            'processing_details': [],
            'column_improvements': {},
            'confidence_scores': {},
            'text_generation_results': [],
            'overall_confidence': 0.0
        }
        
        # Separate numeric and text columns
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        text_cols = df.select_dtypes(exclude=[np.number]).columns.tolist()
        
        # Process numeric columns with traditional + ML methods
        for col in numeric_cols:
            if df[col].isnull().any():
                original_missing = df[col].isnull().sum()
                df_processed[col], col_report = self._process_numeric_column(
                    df_processed, col, config['numerical_method']
                )
                final_missing = df_processed[col].isnull().sum()
                
                processing_report['column_improvements'][col] = original_missing - final_missing
                processing_report['confidence_scores'][col] = col_report.get('confidence', 0.8)
                processing_report['processing_details'].append({
                    'Column': col,
                    'Type': 'Numeric',
                    'Method': config['numerical_method'],
                    'Values_Filled': original_missing - final_missing,
                    'Confidence': col_report.get('confidence', 0.8)
                })
        
        # Process text columns with AI generation
        context_columns = numeric_cols  # Use numeric columns as context for text generation
        
        for col in text_cols:
            if df[col].isnull().any():
                original_missing = df[col].isnull().sum()
                df_processed[col], col_report = self.ai_text_processor.process_text_missing_values(
                    df_processed, col, 
                    method=self._map_text_method(config['text_method']),
                    context_columns=context_columns
                )
                final_missing = df_processed[col].isnull().sum()
                
                processing_report['column_improvements'][col] = original_missing - final_missing
                processing_report['confidence_scores'][col] = col_report.get('confidence', 0.7)
                processing_report['processing_details'].append({
                    'Column': col,
                    'Type': 'Text',
                    'Method': config['text_method'],
                    'Values_Filled': original_missing - final_missing,
                    'Confidence': col_report.get('confidence', 0.7)
                })
                
                # Add text generation insights
                text_insights = self.ai_text_processor.generate_text_insights(df, col, col_report)
                processing_report['text_generation_results'].extend(text_insights)
        
        # Handle outliers with AI-enhanced methods
        df_processed = self._handle_ai_outliers(df_processed, config['outlier_method'])
        
        # Generate AI insights
        processing_report['ai_insights'] = self._generate_ai_insights(df, df_processed, config)
        
        # Calculate overall confidence
        if processing_report['confidence_scores']:
            processing_report['overall_confidence'] = np.mean(list(processing_report['confidence_scores'].values()))
        
        return df_processed, processing_report
    
    def _process_numeric_column(self, df, column, method):
        """Process numeric column with hybrid methods"""
        missing_mask = df[column].isnull()
        
        if method == "Smart Mean/Median":
            return self._smart_mean_median_imputation(df, column)
        elif method == "ML-Based Prediction":
            return self._ml_based_numeric_imputation(df, column)
        elif method == "Hybrid Statistical":
            return self._hybrid_numeric_imputation(df, column)
        else:
            # Fallback to traditional mean
            filled = df[column].fillna(df[column].mean())
            return filled, {'method': 'fallback_mean', 'confidence': 0.6}
    
    def _smart_mean_median_imputation(self, df, column):
        """Smart selection between mean and median based on distribution"""
        values = df[column].dropna()
        
        if len(values) == 0:
            return df[column], {'method': 'no_data', 'confidence': 0.0}
        
        # Analyze distribution
        skewness = values.skew()
        
        if abs(skewness) > 1.0:  # Highly skewed - use median
            fill_value = values.median()
            method_used = f"Smart Median (skew: {skewness:.2f})"
            confidence = 0.85
        else:  # Normal-ish distribution - use mean
            fill_value = values.mean()
            method_used = f"Smart Mean (skew: {skewness:.2f})"
            confidence = 0.90
        
        filled = df[column].fillna(fill_value)
        
        report = {
            'method': method_used,
            'fill_value': fill_value,
            'confidence': confidence,
            'skewness': skewness
        }
        
        return filled, report
    
    def _ml_based_numeric_imputation(self, df, column):
        """ML-based numeric imputation using RandomForest"""
        # Prepare feature columns
        feature_cols = [col for col in df.columns 
                       if col != column and df[col].dtype in ['int64', 'float64']]
        
        if len(feature_cols) == 0:
            # Fallback to mean if no features available
            fill_value = df[column].mean()
            filled = df[column].fillna(fill_value)
            return filled, {'method': 'fallback_mean', 'confidence': 0.6}
        
        # Prepare training data
        missing_mask = df[column].isnull()
        train_mask = ~missing_mask
        
        if train_mask.sum() == 0:
            fill_value = 0
            filled = df[column].fillna(fill_value)
            return filled, {'method': 'zero_fill', 'confidence': 0.3}
        
        X_train = df.loc[train_mask, feature_cols].fillna(0)  # Simple fill for features
        y_train = df.loc[train_mask, column]
        X_pred = df.loc[missing_mask, feature_cols].fillna(0)
        
        try:
            # Train RandomForest
            rf = RandomForestRegressor(n_estimators=50, random_state=42, max_depth=10)
            rf.fit(X_train, y_train)
            
            # Predict missing values
            predictions = rf.predict(X_pred)
            
            # Fill missing values
            filled = df[column].copy()
            filled.loc[missing_mask] = predictions
            
            # Calculate confidence based on model score
            confidence = min(0.95, max(0.5, rf.score(X_train, y_train)))
            
            report = {
                'method': 'RandomForest ML',
                'model_score': rf.score(X_train, y_train),
                'confidence': confidence,
                'features_used': len(feature_cols)
            }
            
            return filled, report
            
        except Exception as e:
            # Fallback to mean on error
            fill_value = df[column].mean()
            filled = df[column].fillna(fill_value)
            return filled, {'method': 'error_fallback_mean', 'confidence': 0.5, 'error': str(e)}
    
    def _hybrid_numeric_imputation(self, df, column):
        """Hybrid numeric imputation combining multiple methods"""
        # Get results from different methods
        smart_result, smart_report = self._smart_mean_median_imputation(df, column)
        ml_result, ml_report = self._ml_based_numeric_imputation(df, column)
        
        # Combine based on confidence
        missing_mask = df[column].isnull()
        final_result = df[column].copy()
        
        smart_confidence = smart_report.get('confidence', 0.7)
        ml_confidence = ml_report.get('confidence', 0.7)
        
        if ml_confidence > smart_confidence + 0.1:  # ML significantly better
            final_result = ml_result
            chosen_method = "ML-based (higher confidence)"
            final_confidence = ml_confidence
        elif smart_confidence > ml_confidence + 0.1:  # Smart method significantly better
            final_result = smart_result
            chosen_method = "Smart statistical (higher confidence)"
            final_confidence = smart_confidence
        else:  # Similar confidence - blend
            # Simple average of predictions for missing values
            for idx in df[missing_mask].index:
                smart_val = smart_result.loc[idx]
                ml_val = ml_result.loc[idx]
                final_result.loc[idx] = (smart_val + ml_val) / 2
            chosen_method = "Blended hybrid"
            final_confidence = (smart_confidence + ml_confidence) / 2
        
        report = {
            'method': chosen_method,
            'confidence': final_confidence,
            'smart_confidence': smart_confidence,
            'ml_confidence': ml_confidence
        }
        
        return final_result, report
    
    def _map_text_method(self, method):
        """Map text method names to AI processor methods"""
        mapping = {
            "AI Text Generation": "ai_generation",
            "Context-Aware Mode": "context_mode",
            "Hybrid AI+Statistical": "hybrid_ai_statistical"
        }
        return mapping.get(method, "ai_generation")
    
    def _handle_ai_outliers(self, df, method):
        """AI-enhanced outlier handling"""
        if method == "None":
            return df
        
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        
        if method == "AI-Enhanced IQR":
            return self._ai_enhanced_iqr(df, numeric_cols)
        elif method == "ML Isolation Forest":
            return self.data_processor._handle_outliers(df, "isolation_forest")
        elif method == "Hybrid Detection":
            return self._hybrid_outlier_detection(df, numeric_cols)
        
        return df
    
    def _ai_enhanced_iqr(self, df, numeric_cols):
        """AI-enhanced IQR with adaptive thresholds"""
        for col in numeric_cols:
            if df[col].isnull().all():
                continue
            
            # Calculate dynamic IQR multiplier based on data distribution
            values = df[col].dropna()
            skewness = abs(values.skew()) if len(values) > 0 else 0
            
            # Adaptive multiplier: more skewed data gets higher tolerance
            multiplier = 1.5 + (skewness * 0.5)  # 1.5 to 2.5 range
            
            Q1 = df[col].quantile(0.25)
            Q3 = df[col].quantile(0.75)
            IQR = Q3 - Q1
            lower = Q1 - multiplier * IQR
            upper = Q3 + multiplier * IQR
            df[col] = df[col].clip(lower, upper)
        
        return df
    
    def _hybrid_outlier_detection(self, df, numeric_cols):
        """Hybrid outlier detection combining multiple methods"""
        # First apply gentle IQR
        df_iqr = self._ai_enhanced_iqr(df.copy(), numeric_cols)
        
        # Then apply isolation forest to remaining data
        df_final = self.data_processor._handle_outliers(df_iqr, "isolation_forest")
        
        return df_final
    
    def _generate_ai_insights(self, original_df, processed_df, config):
        """Generate AI insights about the processing"""
        insights = []
        
        # Data improvement insights
        original_missing = original_df.isnull().sum().sum()
        processed_missing = processed_df.isnull().sum().sum()
        improvement = original_missing - processed_missing
        
        if improvement > 0:
            improvement_pct = (improvement / original_missing) * 100
            insights.append(f"AI processing improved data completeness by {improvement_pct:.1f}% ({improvement} values filled)")
        
        # Method insights
        if config['text_method'] == "AI Text Generation":
            insights.append("Advanced AI text generation was used to create contextually appropriate values for missing text data")
        
        if config['numerical_method'] == "ML-Based Prediction":
            insights.append("Machine learning models were trained to predict missing numerical values based on relationships in your data")
        
        # Distribution insights
        numeric_cols = original_df.select_dtypes(include=[np.number]).columns
        for col in numeric_cols:
            if col in processed_df.columns:
                original_skew = original_df[col].skew() if not original_df[col].isnull().all() else 0
                processed_skew = processed_df[col].skew() if not processed_df[col].isnull().all() else 0
                
                if abs(original_skew - processed_skew) > 0.5:
                    insights.append(f"Data distribution for {col} was adjusted during processing (skewness change: {original_skew:.2f} → {processed_skew:.2f})")
        
        return insights
    
    def calculate_intelligent_weights(self, df, survey_config):
        """Calculate weights with AI enhancement"""
        # Start with basic weights
        basic_weights = self.data_processor.calculate_weights(df, survey_config)
        
        # AI enhancement: adjust weights based on data quality and completeness
        quality_adjustment = self._calculate_quality_adjustments(df)
        
        # Apply adjustments
        enhanced_weights = basic_weights * quality_adjustment
        
        # Ensure weights are reasonable (between 0.5 and 3.0 times basic weight)
        enhanced_weights = np.clip(enhanced_weights, basic_weights * 0.5, basic_weights * 3.0)
        
        return enhanced_weights
    
    def _calculate_quality_adjustments(self, df):
        """Calculate quality-based weight adjustments"""
        n_records = len(df)
        adjustments = np.ones(n_records)
        
        # Adjust based on completeness of each record
        for idx in range(n_records):
            record = df.iloc[idx]
            completeness = (record.notna().sum()) / len(record)
            
            # Higher weight for more complete records (up to 1.2x)
            adjustments[idx] = 0.8 + (completeness * 0.4)
        
        return adjustments
