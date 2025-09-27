import numpy as np
import pandas as pd
from scipy import stats
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.model_selection import cross_val_score
from sklearn.metrics import mean_squared_error
import warnings
warnings.filterwarnings('ignore')

class AIEstimator:
    def __init__(self):
        self.models = {}
        
    def generate_estimates(self, df, target_var, method='design_based', 
                          confidence_level=0.95, stratification_vars=None):
        """Generate population estimates using various methods"""
        
        if method == 'design_based':
            return self._design_based_estimation(df, target_var, confidence_level, stratification_vars)
        elif method == 'model_based':
            return self._model_based_estimation(df, target_var, confidence_level)
        elif method == 'hybrid':
            return self._hybrid_estimation(df, target_var, confidence_level, stratification_vars)
    
    def generate_ai_enhanced_estimates(self, df, target_var, method='ai_enhanced_hybrid',
                                     confidence_level=0.95, stratification_vars=None):
        """Generate AI-enhanced estimates with advanced techniques"""
        
        if method == 'ai_enhanced_hybrid':
            return self._ai_enhanced_hybrid_estimation(df, target_var, confidence_level, stratification_vars)
        else:
            # Fall back to regular methods
            return self.generate_estimates(df, target_var, method, confidence_level, stratification_vars)
    
    def _ai_enhanced_hybrid_estimation(self, df, target_var, confidence_level, stratification_vars):
        """AI-Enhanced hybrid estimation with ensemble methods"""
        # Get results from multiple methods
        design_result = self._design_based_estimation(df, target_var, confidence_level, stratification_vars)
        model_result = self._enhanced_model_based_estimation(df, target_var, confidence_level)
        
        # AI-enhanced weighting based on data quality and model performance
        design_weight, model_weight = self._calculate_intelligent_weights(df, target_var, design_result, model_result)
        
        # Combine estimates
        hybrid_total = (design_weight * design_result['total'] + 
                       model_weight * model_result['total'])
        
        # Enhanced standard error calculation
        hybrid_se = self._calculate_enhanced_standard_error(
            design_result, model_result, design_weight, model_weight
        )
        
        # AI uncertainty quantification
        ai_uncertainty = self._quantify_ai_uncertainty(df, target_var)
        hybrid_se = hybrid_se * (1 + ai_uncertainty)
        
        # Confidence interval
        alpha = 1 - confidence_level
        t_critical = stats.t.ppf(1 - alpha/2, df=len(df)-1)
        ci_lower = hybrid_total - t_critical * hybrid_se
        ci_upper = hybrid_total + t_critical * hybrid_se
        
        cv = (hybrid_se / hybrid_total) * 100 if hybrid_total != 0 else 0
        
        # Generate AI insights
        ai_insights = self._generate_estimation_insights(df, target_var, design_result, model_result)
        
        # Calculate AI confidence
        ai_confidence = self._calculate_ai_confidence(df, target_var, model_result)
        
        return {
            'total': hybrid_total,
            'mean': hybrid_total / df.get('survey_weight', np.ones(len(df))).sum(),
            'se': hybrid_se,
            'cv': cv,
            'ci_lower': ci_lower,
            'ci_upper': ci_upper,
            'n': len(df),
            'design_estimate': design_result['total'],
            'model_estimate': model_result['total'],
            'design_weight': design_weight,
            'model_weight': model_weight,
            'ai_confidence': ai_confidence,
            'ai_insights': ai_insights,
            'strata_estimates': design_result.get('strata_estimates', {})
        }
    
    def _enhanced_model_based_estimation(self, df, target_var, confidence_level):
        """Enhanced model-based estimation with ensemble methods"""
        # Prepare features
        feature_cols = [col for col in df.columns 
                       if col not in [target_var, 'survey_weight'] 
                       and df[col].dtype in ['int64', 'float64']]
        
        if len(feature_cols) == 0:
            return self._model_based_estimation(df, target_var, confidence_level)
        
        X = df[feature_cols].fillna(df[feature_cols].mean())
        y = df[target_var].fillna(df[target_var].mean())
        
        # Ensemble of models
        models = {
            'rf': RandomForestRegressor(n_estimators=100, random_state=42),
            'gb': GradientBoostingRegressor(n_estimators=100, random_state=42, max_depth=6)
        }
        
        predictions = {}
        model_scores = {}
        
        for name, model in models.items():
            # Train model
            model.fit(X, y)
            self.models[f"{target_var}_{name}"] = model
            
            # Generate predictions
            pred = model.predict(X)
            predictions[name] = pred
            
            # Calculate score
            model_scores[name] = model.score(X, y)
        
        # Weighted ensemble prediction
        total_score = sum(model_scores.values())
        if total_score > 0:
            weights = {name: score/total_score for name, score in model_scores.items()}
        else:
            weights = {name: 1/len(models) for name in models.keys()}
        
        ensemble_predictions = np.zeros(len(df))
        for name, pred in predictions.items():
            ensemble_predictions += weights[name] * pred
        
        # Calculate estimates
        survey_weights = df.get('survey_weight', np.ones(len(df)))
        total_estimate = np.sum(ensemble_predictions * survey_weights)
        
        # Enhanced standard error using ensemble variance
        ensemble_variance = np.var([pred for pred in predictions.values()], axis=0).mean()
        se = np.sqrt(ensemble_variance * np.sum(survey_weights**2))
        
        # Confidence interval
        alpha = 1 - confidence_level
        z_critical = stats.norm.ppf(1 - alpha/2)
        ci_lower = total_estimate - z_critical * se
        ci_upper = total_estimate + z_critical * se
        
        cv = (se / total_estimate) * 100 if total_estimate != 0 else 0
        
        return {
            'total': total_estimate,
            'mean': total_estimate / np.sum(survey_weights),
            'se': se,
            'cv': cv,
            'ci_lower': ci_lower,
            'ci_upper': ci_upper,
            'n': len(df),
            'ensemble_r2': np.mean(list(model_scores.values())),
            'model_weights': weights,
            'strata_estimates': {}
        }
    
    def _calculate_intelligent_weights(self, df, target_var, design_result, model_result):
        """Calculate intelligent weights for hybrid estimation"""
        # Base weights
        design_base = 0.6
        model_base = 0.4
        
        # Adjust based on data quality
        completeness = df[target_var].notna().mean()
        if completeness < 0.8:  # Poor completeness favors design-based
            design_weight = design_base + 0.2
            model_weight = model_base - 0.2
        elif completeness > 0.95:  # High completeness favors model-based
            design_weight = design_base - 0.1
            model_weight = model_base + 0.1
        else:
            design_weight = design_base
            model_weight = model_base
        
        # Adjust based on model performance
        if 'ensemble_r2' in model_result and model_result['ensemble_r2'] > 0.8:
            # High R² favors model-based
            model_weight += 0.1
            design_weight -= 0.1
        elif 'ensemble_r2' in model_result and model_result['ensemble_r2'] < 0.5:
            # Low R² favors design-based
            design_weight += 0.1
            model_weight -= 0.1
        
        # Ensure weights sum to 1
        total = design_weight + model_weight
        design_weight = design_weight / total
        model_weight = model_weight / total
        
        return design_weight, model_weight
    
    def _calculate_enhanced_standard_error(self, design_result, model_result, design_weight, model_weight):
        """Calculate enhanced standard error for hybrid estimates"""
        # Combine standard errors with correlation adjustment
        design_se = design_result['se']
        model_se = model_result['se']
        
        # Assume moderate positive correlation between methods
        correlation = 0.3
        
        combined_se = np.sqrt(
            design_weight**2 * design_se**2 + 
            model_weight**2 * model_se**2 + 
            2 * design_weight * model_weight * correlation * design_se * model_se
        )
        
        return combined_se
    
    def _quantify_ai_uncertainty(self, df, target_var):
        """Quantify additional uncertainty from AI processing"""
        # Simple heuristic based on data characteristics
        missing_rate = df[target_var].isnull().mean()
        uncertainty_factor = missing_rate * 0.1  # 0-10% additional uncertainty
        
        return uncertainty_factor
    
    def _calculate_ai_confidence(self, df, target_var, model_result):
        """Calculate overall AI confidence in the estimates"""
        factors = []
        
        # Data completeness factor
        completeness = df[target_var].notna().mean()
        factors.append(completeness)
        
        # Model performance factor
        if 'ensemble_r2' in model_result:
            factors.append(min(1.0, model_result['ensemble_r2']))
        
        # Sample size factor
        sample_size_factor = min(1.0, len(df) / 100)  # Normalize by 100
        factors.append(sample_size_factor)
        
        # Overall confidence as geometric mean
        confidence = np.prod(factors) ** (1/len(factors))
        return confidence
    
    def _generate_estimation_insights(self, df, target_var, design_result, model_result):
        """Generate AI insights about the estimation process"""
        insights = []
        
        # Compare estimates
        design_est = design_result['total']
        model_est = model_result['total']
        difference = abs(design_est - model_est) / design_est * 100
        
        if difference < 5:
            insights.append("Design-based and model-based estimates are very similar, indicating robust results")
        elif difference > 20:
            insights.append(f"Significant difference ({difference:.1f}%) between estimation methods suggests need for careful interpretation")
        
        # Precision insights
        design_cv = design_result['cv']
        model_cv = model_result.get('cv', design_cv)
        
        if min(design_cv, model_cv) < 10:
            insights.append("High precision estimates achieved with low coefficient of variation")
        elif max(design_cv, model_cv) > 30:
            insights.append("Lower precision estimates - consider increasing sample size for future surveys")
        
        # Model performance insights
        if 'ensemble_r2' in model_result and model_result['ensemble_r2'] > 0.7:
            insights.append("Strong predictive relationships found in your data, enhancing estimation accuracy")
        
        return "; ".join(insights) if insights else "AI analysis completed successfully"
    
    def _design_based_estimation(self, df, target_var, confidence_level, stratification_vars):
        """Traditional design-based estimation"""
        # Check if target variable exists and has valid data
        if target_var not in df.columns:
            raise ValueError(f"Target variable '{target_var}' not found in data")
        
        weights = df.get('survey_weight', np.ones(len(df)))
        values = df[target_var].dropna()
        
        if len(values) == 0:
            raise ValueError(f"No valid values found for target variable '{target_var}'")
        
        corresponding_weights = weights[df[target_var].notna()]
        
        # Weighted estimates
        total_estimate = np.sum(values * corresponding_weights)
        mean_estimate = total_estimate / np.sum(corresponding_weights)
        
        # Standard error calculation
        n = len(values)
        variance = np.sum(corresponding_weights * (values - mean_estimate)**2) / (n - 1)
        se = np.sqrt(variance / n)
        
        # Confidence interval
        alpha = 1 - confidence_level
        t_critical = stats.t.ppf(1 - alpha/2, df=n-1)
        ci_lower = total_estimate - t_critical * se
        ci_upper = total_estimate + t_critical * se
        
        # Coefficient of variation
        cv = (se / total_estimate) * 100 if total_estimate != 0 else 0
        
        result = {
            'total': total_estimate,
            'mean': mean_estimate,
            'se': se,
            'cv': cv,
            'ci_lower': ci_lower,
            'ci_upper': ci_upper,
            'n': n
        }
        
        # Stratified estimates if requested
        if stratification_vars:
            strata_estimates = {}
            for stratum in stratification_vars:
                if stratum in df.columns:
                    for level in df[stratum].unique():
                        if pd.notna(level):  # Skip NaN levels
                            stratum_data = df[df[stratum] == level]
                            if len(stratum_data) > 0 and stratum_data[target_var].notna().any():
                                stratum_values = stratum_data[target_var].dropna()
                                stratum_weights = stratum_data.get('survey_weight', np.ones(len(stratum_data)))[stratum_data[target_var].notna()]
                                stratum_total = np.sum(stratum_values * stratum_weights)
                                strata_estimates[f"{stratum}_{level}"] = stratum_total
            result['strata_estimates'] = strata_estimates
        else:
            result['strata_estimates'] = {}
        
        return result
    
    def _model_based_estimation(self, df, target_var, confidence_level):
        """Model-based estimation using Random Forest"""
        # Prepare features (exclude target and weight columns)
        feature_cols = [col for col in df.columns 
                       if col not in [target_var, 'survey_weight'] 
                       and df[col].dtype in ['int64', 'float64']]
        
        if len(feature_cols) == 0:
            # Fallback to design-based if no features
            return self._design_based_estimation(df, target_var, confidence_level, None)
        
        X = df[feature_cols].fillna(df[feature_cols].mean())
        y = df[target_var].fillna(df[target_var].mean())
        
        # Train Random Forest model
        self.model = RandomForestRegressor(n_estimators=100, random_state=42)
        self.model.fit(X, y)
        
        # Generate predictions
        predictions = self.model.predict(X)
        
        # Calculate estimates
        weights = df.get('survey_weight', np.ones(len(df)))
        total_estimate = np.sum(predictions * weights)
        
        # Model-based standard error using cross-validation
        cv_scores = cross_val_score(self.model, X, y, cv=min(5, len(df)), scoring='neg_mean_squared_error')
        mse = -np.mean(cv_scores)
        se = np.sqrt(mse * np.sum(weights**2))
        
        # Confidence interval
        alpha = 1 - confidence_level
        z_critical = stats.norm.ppf(1 - alpha/2)
        ci_lower = total_estimate - z_critical * se
        ci_upper = total_estimate + z_critical * se
        
        cv = (se / total_estimate) * 100 if total_estimate != 0 else 0
        
        return {
            'total': total_estimate,
            'mean': total_estimate / np.sum(weights),
            'se': se,
            'cv': cv,
            'ci_lower': ci_lower,
            'ci_upper': ci_upper,
            'n': len(df),
            'model_r2': self.model.score(X, y),
            'strata_estimates': {}
        }
    
    def _hybrid_estimation(self, df, target_var, confidence_level, stratification_vars):
        """Hybrid approach combining design-based and model-based methods"""
        design_result = self._design_based_estimation(df, target_var, confidence_level, stratification_vars)
        model_result = self._model_based_estimation(df, target_var, confidence_level)
        
        # Weighted combination of estimates
        design_weight = 0.6  # Give more weight to design-based
        model_weight = 0.4
        
        hybrid_total = (design_weight * design_result['total'] + 
                       model_weight * model_result['total'])
        
        # Combined standard error
        hybrid_se = np.sqrt(design_weight**2 * design_result['se']**2 + 
                           model_weight**2 * model_result['se']**2)
        
        # Confidence interval
        alpha = 1 - confidence_level
        t_critical = stats.t.ppf(1 - alpha/2, df=len(df)-1)
        ci_lower = hybrid_total - t_critical * hybrid_se
        ci_upper = hybrid_total + t_critical * hybrid_se
        
        cv = (hybrid_se / hybrid_total) * 100 if hybrid_total != 0 else 0
        
        return {
            'total': hybrid_total,
            'mean': hybrid_total / df.get('survey_weight', np.ones(len(df))).sum(),
            'se': hybrid_se,
            'cv': cv,
            'ci_lower': ci_lower,
            'ci_upper': ci_upper,
            'n': len(df),
            'design_estimate': design_result['total'],
            'model_estimate': model_result['total'],
            'strata_estimates': design_result.get('strata_estimates', {})
        }
