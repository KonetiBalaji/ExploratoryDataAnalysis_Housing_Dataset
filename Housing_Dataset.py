# Housing_Dataset.py

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from scipy import stats
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import logging

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class HousingAnalysis:
    def __init__(self, data_path):
        self.data = pd.read_csv(data_path)
        self.numerical_cols = self.data.select_dtypes(include='number').columns
        self.categorical_cols = self.data.select_dtypes(include='object').columns
        logging.info(f"Dataset loaded with shape: {self.data.shape}")

    def preprocess_data(self):
        """Preprocess the data including handling outliers and scaling"""
        # Handle outliers in price and area using IQR method
        for col in ['price', 'area']:
            Q1 = self.data[col].quantile(0.25)
            Q3 = self.data[col].quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR
            self.data = self.data[(self.data[col] >= lower_bound) & (self.data[col] <= upper_bound)]
        
        # Create new features
        self.data['price_per_sqft'] = self.data['price'] / self.data['area']
        self.data['total_rooms'] = self.data['bedrooms'] + self.data['bathrooms']
        self.data['has_amenities'] = (
            (self.data['guestroom'] == 'yes').astype(int) +
            (self.data['basement'] == 'yes').astype(int) +
            (self.data['hotwaterheating'] == 'yes').astype(int) +
            (self.data['airconditioning'] == 'yes').astype(int)
        )
        
        logging.info("Data preprocessing completed")

    def analyze_correlations(self):
        """Analyze correlations between features"""
        corr_matrix = self.data.select_dtypes(include='number').corr()
        
        # Create correlation heatmap
        plt.figure(figsize=(12, 8))
        sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', center=0, fmt='.2f')
        plt.title('Enhanced Correlation Heatmap')
        plt.tight_layout()
        plt.savefig('images/enhanced_correlation_heatmap.png')
        plt.close()
        
        logging.info("Correlation analysis completed")

    def perform_statistical_tests(self):
        """Perform statistical tests on categorical variables"""
        results = {}
        for col in self.categorical_cols:
            groups = [group['price'] for name, group in self.data.groupby(col)]
            f_stat, p_value = stats.f_oneway(*groups)
            results[col] = {'f_statistic': f_stat, 'p_value': p_value}
        
        logging.info("Statistical tests completed")
        return results

    def create_advanced_visualizations(self):
        """Create advanced visualizations using Plotly"""
        # Price distribution by furnishing status
        fig = px.box(self.data, x='furnishingstatus', y='price',
                    title='Price Distribution by Furnishing Status')
        fig.write_html('images/price_furnishing_interactive.html')
        
        # 3D scatter plot
        fig = px.scatter_3d(self.data, x='area', y='bedrooms', z='price',
                           color='furnishingstatus', title='3D Price Analysis')
        fig.write_html('images/3d_price_analysis.html')
        
        # Price per square foot by location features
        fig = make_subplots(rows=2, cols=2,
                           subplot_titles=('Main Road', 'Guest Room', 'Basement', 'Air Conditioning'))
        
        for i, col in enumerate(['mainroad', 'guestroom', 'basement', 'airconditioning']):
            row = i // 2 + 1
            col_num = i % 2 + 1
            fig.add_trace(
                go.Box(x=self.data[col], y=self.data['price_per_sqft']),
                row=row, col=col_num
            )
        
        fig.update_layout(height=800, title_text="Price per Square Foot Analysis")
        fig.write_html('images/price_per_sqft_analysis.html')
        
        # Feature importance plot
        if hasattr(self, 'feature_importance'):
            fig = px.bar(
                x=self.feature_importance.index,
                y=self.feature_importance.values,
                title='Feature Importance',
                labels={'x': 'Features', 'y': 'Importance Score'}
            )
            fig.write_html('images/feature_importance.html')
        
        logging.info("Advanced visualizations created")

    def build_prediction_model(self):
        """Build and evaluate prediction models"""
        # Prepare features
        X = self.data.drop(['price', 'price_per_sqft'], axis=1)
        y = self.data['price']
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        # Create preprocessing pipeline
        numeric_features = X.select_dtypes(include=['int64', 'float64']).columns
        categorical_features = X.select_dtypes(include=['object']).columns
        
        preprocessor = ColumnTransformer(
            transformers=[
                ('num', StandardScaler(), numeric_features),
                ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_features)
            ])
        
        # Linear Regression
        lr_pipeline = Pipeline([
            ('preprocessor', preprocessor),
            ('regressor', LinearRegression())
        ])
        
        # Random Forest
        rf_pipeline = Pipeline([
            ('preprocessor', preprocessor),
            ('regressor', RandomForestRegressor(n_estimators=100, random_state=42))
        ])
        
        # Train and evaluate models
        models = {
            'Linear Regression': lr_pipeline,
            'Random Forest': rf_pipeline
        }
        
        results = {}
        for name, model in models.items():
            # Train model
            model.fit(X_train, y_train)
            
            # Make predictions
            y_pred = model.predict(X_test)
            
            # Calculate metrics
            results[name] = {
                'R2': r2_score(y_test, y_pred),
                'MSE': mean_squared_error(y_test, y_pred),
                'MAE': mean_absolute_error(y_test, y_pred),
                'Cross Val Score': cross_val_score(model, X, y, cv=5).mean()
            }
            
            # Get feature importance for Random Forest
            if name == 'Random Forest':
                feature_names = (
                    list(numeric_features) +
                    [f"{col}_{val}" for col, vals in 
                     zip(categorical_features, model.named_steps['preprocessor']
                         .named_transformers_['cat'].categories_)
                     for val in vals]
                )
                self.feature_importance = pd.Series(
                    model.named_steps['regressor'].feature_importances_,
                    index=feature_names
                ).sort_values(ascending=False)
        
        logging.info("Model evaluation completed")
        return results

    def run_analysis(self):
        """Run the complete analysis pipeline"""
        self.preprocess_data()
        self.analyze_correlations()
        stats_results = self.perform_statistical_tests()
        self.create_advanced_visualizations()
        model_results = self.build_prediction_model()
        
        # Print summary
        print("\nAnalysis Summary:")
        print(f"Dataset shape after preprocessing: {self.data.shape}")
        print("\nStatistical Test Results:")
        for col, results in stats_results.items():
            print(f"{col}: p-value = {results['p_value']:.4f}")
        print("\nModel Performance:")
        for model_name, metrics in model_results.items():
            print(f"\n{model_name}:")
            for metric_name, value in metrics.items():
                print(f"{metric_name}: {value:.4f}")

if __name__ == "__main__":
    analysis = HousingAnalysis('./dataset/Housing.csv')
    analysis.run_analysis()
