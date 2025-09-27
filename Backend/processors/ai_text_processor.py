import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
import re
import random

class AITextProcessor:
    def __init__(self):
        self.models = {}
        self.vectorizers = {}
        
    def process_text_missing_values(self, df, column, method='ai_generation', context_columns=None):
        """AI-powered text missing value processing"""
        
        if method == 'ai_generation':
            return self._ai_text_generation(df, column, context_columns)
        elif method == 'context_mode':
            return self._context_aware_mode(df, column, context_columns)
        elif method == 'hybrid_ai_statistical':
            return self._hybrid_text_processing(df, column, context_columns)
        else:
            return self._traditional_mode_imputation(df, column)
    
    def _ai_text_generation(self, df, column, context_columns):
        """Generate text values using AI patterns"""
        missing_mask = df[column].isnull()
        if not missing_mask.any():
            return df[column], {}
        
        # Analyze existing patterns
        existing_values = df[column].dropna()
        if len(existing_values) == 0:
            return df[column].fillna('Unknown'), {'method': 'fallback', 'generated': 0}
        
        # Pattern analysis
        patterns = self._analyze_text_patterns(existing_values)
        
        # Generate new values based on patterns
        generated_values = []
        generation_report = {
            'method': 'AI Text Generation',
            'patterns_found': len(patterns),
            'generated_count': 0,
            'confidence_scores': []
        }
        
        for idx in df[missing_mask].index:
            # Use context if available
            if context_columns:
                context_value = self._generate_from_context(df, idx, context_columns, patterns)
                confidence = 0.8
            else:
                context_value = self._generate_from_patterns(patterns)
                confidence = 0.6
            
            generated_values.append(context_value)
            generation_report['confidence_scores'].append(confidence)
            generation_report['generated_count'] += 1
        
        # Fill missing values
        df_result = df.copy()
        df_result.loc[missing_mask, column] = generated_values
        
        return df_result[column], generation_report
    
    def _analyze_text_patterns(self, values):
        """Analyze patterns in existing text values"""
        patterns = {
            'common_words': [],
            'length_distribution': [],
            'case_patterns': [],
            'categories': {}
        }
        
        # Extract common patterns
        all_text = ' '.join(values.astype(str)).lower()
        words = re.findall(r'\b\w+\b', all_text)
        
        # Most common words
        word_freq = {}
        for word in words:
            word_freq[word] = word_freq.get(word, 0) + 1
        
        patterns['common_words'] = sorted(word_freq.items(), key=lambda x: x[1], reverse=True)[:10]
        
        # Length distribution
        patterns['length_distribution'] = [len(str(v)) for v in values]
        
        # Case patterns
        patterns['case_patterns'] = {
            'upper': sum(1 for v in values if str(v).isupper()),
            'lower': sum(1 for v in values if str(v).islower()),
            'title': sum(1 for v in values if str(v).istitle()),
            'mixed': sum(1 for v in values if not any([str(v).isupper(), str(v).islower(), str(v).istitle()]))
        }
        
        # Category analysis
        unique_values = values.value_counts()
        patterns['categories'] = dict(unique_values.head(10))
        
        return patterns
    
    def _generate_from_context(self, df, idx, context_columns, patterns):
        """Generate text value based on context from other columns"""
        # Simple context-based generation
        context_values = []
        for col in context_columns:
            if col in df.columns and not pd.isna(df.loc[idx, col]):
                context_values.append(str(df.loc[idx, col]))
        
        # Use context to influence generation
        if context_values and patterns['categories']:
            # Smart generation based on context
            categories = list(patterns['categories'].keys())
            
            # Simple heuristic: match context patterns
            for category in categories:
                if any(word in category.lower() for word in ' '.join(context_values).lower().split()):
                    return category
            
            # Fall back to weighted random selection
            return self._weighted_random_selection(patterns['categories'])
        
        return self._generate_from_patterns(patterns)
    
    def _generate_from_patterns(self, patterns):
        """Generate text value based on learned patterns"""
        if patterns['categories']:
            return self._weighted_random_selection(patterns['categories'])
        
        # Generate based on common words if no categories
        if patterns['common_words']:
            common_word = random.choice(patterns['common_words'][:3])[0]
            return common_word.title()
        
        return 'Generated_Value'
    
    def _weighted_random_selection(self, categories):
        """Select category based on frequency weights"""
        total = sum(categories.values())
        if total == 0:
            return list(categories.keys())[0] if categories else 'Unknown'
        
        weights = [freq/total for freq in categories.values()]
        return np.random.choice(list(categories.keys()), p=weights)
    
    def _context_aware_mode(self, df, column, context_columns):
        """Context-aware mode imputation"""
        if not context_columns:
            return self._traditional_mode_imputation(df, column)
        
        missing_mask = df[column].isnull()
        df_result = df.copy()
        
        for idx in df[missing_mask].index:
            # Find similar records based on context
            similar_records = self._find_similar_records(df, idx, context_columns)
            
            if len(similar_records) > 0:
                # Use mode of similar records
                mode_value = similar_records[column].mode()
                if len(mode_value) > 0:
                    df_result.loc[idx, column] = mode_value.iloc[0]
                else:
                    df_result.loc[idx, column] = 'Context_Unknown'
            else:
                # Fall back to global mode
                global_mode = df[column].mode()
                df_result.loc[idx, column] = global_mode.iloc[0] if len(global_mode) > 0 else 'Unknown'
        
        report = {
            'method': 'Context-Aware Mode',
            'filled_count': missing_mask.sum(),
            'context_columns_used': context_columns
        }
        
        return df_result[column], report
    
    def _find_similar_records(self, df, target_idx, context_columns):
        """Find records similar to target based on context columns"""
        target_row = df.loc[target_idx]
        similar_records = df.copy()
        
        for col in context_columns:
            if col in df.columns and not pd.isna(target_row[col]):
                similar_records = similar_records[similar_records[col] == target_row[col]]
        
        return similar_records
    
    def _hybrid_text_processing(self, df, column, context_columns):
        """Hybrid approach combining AI generation and statistical methods"""
        # First try AI generation
        ai_result, ai_report = self._ai_text_generation(df, column, context_columns)
        
        # Then apply context-aware refinement
        context_result, context_report = self._context_aware_mode(df, column, context_columns)
        
        # Combine results based on confidence
        final_result = df[column].copy()
        missing_mask = df[column].isnull()
        
        hybrid_report = {
            'method': 'Hybrid AI+Statistical',
            'ai_generated': ai_report['generated_count'] if 'generated_count' in ai_report else 0,
            'context_refined': context_report['filled_count'] if 'filled_count' in context_report else 0,
            'confidence_threshold': 0.7
        }
        
        # Use AI generation for high-confidence cases, context-aware for others
        for idx in df[missing_mask].index:
            if 'confidence_scores' in ai_report and len(ai_report['confidence_scores']) > 0:
                ai_confidence = ai_report['confidence_scores'][0]  # Simplified
                if ai_confidence >= 0.7:
                    final_result.loc[idx] = ai_result.loc[idx]
                else:
                    final_result.loc[idx] = context_result.loc[idx]
            else:
                final_result.loc[idx] = context_result.loc[idx]
        
        return final_result, hybrid_report
    
    def _traditional_mode_imputation(self, df, column):
        """Traditional mode imputation as fallback"""
        mode_values = df[column].mode()
        if len(mode_values) > 0:
            filled = df[column].fillna(mode_values.iloc[0])
        else:
            filled = df[column].fillna('Unknown')
        
        report = {
            'method': 'Traditional Mode',
            'fill_value': mode_values.iloc[0] if len(mode_values) > 0 else 'Unknown'
        }
        
        return filled, report
    
    def generate_text_insights(self, df, column, processing_report):
        """Generate insights about text processing"""
        insights = []
        
        if 'patterns_found' in processing_report:
            insights.append(f"Discovered {processing_report['patterns_found']} text patterns")
        
        if 'generated_count' in processing_report:
            insights.append(f"Generated {processing_report['generated_count']} intelligent text values")
        
        if 'confidence_scores' in processing_report:
            avg_confidence = np.mean(processing_report['confidence_scores'])
            insights.append(f"Average AI confidence: {avg_confidence:.1%}")
        
        return insights
