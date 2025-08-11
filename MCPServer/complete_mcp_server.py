
#!/usr/bin/env python3
"""
Complete MCP Server for GDP Growth Prediction with CSV Training
Features:
- CSV file upload and model training
- 5-year future predictions with charts
- Natural language stress testing
- Comprehensive data validation
"""

import json
import asyncio
import logging
import pandas as pd
import seaborn as sns
import numpy as np
from typing import Any, Dict, List, Optional, Union, Tuple
import httpx
from pydantic import BaseModel, Field
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
import io
import os
from datetime import datetime
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')  # Use non-GUI backend
import base64
import re

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class CSVProcessor:
    """Handles CSV file processing and validation for GDP data"""

    @staticmethod
    def validate_csv_format(df: pd.DataFrame) -> Tuple[bool, str]:
        """Validate CSV has required columns and format for GDP prediction"""
        required_columns = [
            'Year',
            'Average annual rate of growth of exports (scored 1-5)', 
            'Inward FDI flows (% of fixed investment)', 
            'CO2 emissions: Oil (% change y/y)', 
            'Average annual GDP growth (scored 1-5)'
        ]

        # Check if all required columns exist
        missing_columns = [col for col in required_columns if col not in df.columns]
        if missing_columns:
            return False, f"Missing required columns: {missing_columns}"

        try:
            if not df['Year'].between(2000, 2025).all():
                return False, "Year must be between 2000 and 2025"
            if not df['Average annual rate of growth of exports (scored 1-5)'].between(1, 5).all():
                return False, "Average annual rate of growth of exports (scored 1-5) must be between 1 and 5"
            if not df['Inward FDI flows (% of fixed investment)'].between(0, 50).all():
                return False, "Inward FDI flows (% of fixed investment) must be between 0 and 50"
            if not df['CO2 emissions: Oil (% change y/y)'].between(-30, 30).all():
                return False, "CO2 emissions: Oil (% change y/y) must be between -30 and 30"
            if not df['Average annual GDP growth (scored 1-5)'].between(1, 5).all():
                return False, "Average annual GDP growth (scored 1-5) must be between 1 and 5"
            if len(df) < 15:
                return False, "CSV must contain at least 15 records for reliable training"
            if df['Year'].duplicated().any():
                return False, "Duplicate years found in data"
        except Exception as e:
            return False, f"Data validation error: {str(e)}"

        return True, "CSV format is valid"


    @staticmethod
    def process_csv_file(file_content: bytes) -> Tuple[bool, Union[pd.DataFrame, str]]:
        """Process uploaded CSV file"""
        try:
            # Read CSV from bytes
            csv_string = file_content.decode('utf-8')
            df = pd.read_csv(io.StringIO(csv_string))

            # Validate format
            is_valid, message = CSVProcessor.validate_csv_format(df)
            if not is_valid:
                return False, message
            
            # Compute the correlation matrix
            corr_matrix = df.corr()

            # Print the correlation matrix
            print(corr_matrix)

            # Optionally: visualize as a heatmap
            plt.figure(figsize=(8, 6))
            sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt=".2f")
            plt.title('Correlation Matrix')
            plt.show()

            # Clean data (remove any NaN values)
            df_clean = df.dropna()

            # Sort by year
            df_clean = df_clean.sort_values('Year').reset_index(drop=True)

            # Log processing info
            logger.info(f"Successfully processed CSV: {len(df_clean)} records from {df_clean['Year'].min()} to {df_clean['Year'].max()}")

            return True, df_clean

        except Exception as e:
            logger.error(f"CSV processing error: {e}")
            return False, f"Failed to process CSV: {str(e)}"

class GDPForecastModel:
    """Advanced GDP prediction model with forecasting capabilities"""

    def __init__(self):
        self.model = LinearRegression()
        self.feature_names = ['Average annual rate of growth of exports (scored 1-5)', 'Inward FDI flows (% of fixed investment)', 'CO2 emissions: Oil (% change y/y)']
        self.target_name = 'Average annual GDP growth (scored 1-5)'
        self.is_trained = False
        self.training_data = None
        self.coefficients = {}
        self.model_performance = {}
        self.training_timestamp = None

    def train_from_csv(self, df: pd.DataFrame) -> Dict:
        """Train model from CSV data and calculate performance metrics"""
        try:
            # Prepare features and target
            X = df[self.feature_names]
            y = df[self.target_name]

            # Store training data
            self.training_data = df.copy()

            # Split data for validation
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.2, random_state=42
            )

            # Train model
            self.model.fit(X_train, y_train)
            self.is_trained = True
            self.training_timestamp = datetime.now()

            # Evaluate model
            y_pred_train = self.model.predict(X_train)
            y_pred_test = self.model.predict(X_test)

            train_r2 = r2_score(y_train, y_pred_train)
            test_r2 = r2_score(y_test, y_pred_test)
            train_rmse = np.sqrt(mean_squared_error(y_train, y_pred_train))
            test_rmse = np.sqrt(mean_squared_error(y_test, y_pred_test))

            # Store coefficients
            self.coefficients = {
                'export_growth': float(self.model.coef_[0]),
                'fdi_flows': float(self.model.coef_[1]),
                'co2_change': float(self.model.coef_[2]),
                'intercept': float(self.model.intercept_)
            }

            # Store performance metrics
            self.model_performance = {
                'train_r2': float(train_r2),
                'test_r2': float(test_r2),
                'train_rmse': float(train_rmse),
                'test_rmse': float(test_rmse),
                'data_points': len(df),
                'year_range': f"{df['Year'].min()}-{df['Year'].max()}"
            }

            logger.info(f"Model trained successfully: R²={test_r2:.3f}, RMSE={test_rmse:.3f}")

            return {
                'success': True,
                'performance': self.model_performance,
                'coefficients': self.coefficients,
                'message': f'Model trained on {len(df)} data points from {df["Year"].min()}-{df["Year"].max()}'
            }

        except Exception as e:
            logger.error(f"Training error: {e}")
            return {
                'success': False,
                'error': f'Training failed: {str(e)}'
            }

    def predict_future_years(self, years: List[int], scenarios: Dict = None) -> Dict:
        """Predict GDP growth for future years with scenario modeling"""
        if not self.is_trained:
            return {'success': False, 'error': 'Model not trained yet'}

        try:
            predictions = []

            # Get recent trends for baseline scenario
            recent_data = self.training_data.tail(5)
            baseline_export = recent_data['Average annual rate of growth of exports (scored 1-5)'].mean()
            baseline_fdi = recent_data['Inward FDI flows (% of fixed investment)'].mean()
            baseline_co2 = recent_data['CO2 emissions: Oil (% change y/y)'].mean()

            for year in years:
                # Use scenarios if provided, otherwise use baseline with slight trend
                if scenarios and str(year) in scenarios:
                    scenario = scenarios[str(year)]
                    export_score = scenario.get('export_growth', baseline_export)
                    fdi_flows = scenario.get('fdi_flows', baseline_fdi)
                    co2_change = scenario.get('co2_change', baseline_co2)
                else:
                    # Apply modest trend based on historical data
                    years_ahead = year - self.training_data['Year'].max()
                    export_score = baseline_export + (years_ahead * 0.05)  # Slight improvement
                    fdi_flows = baseline_fdi + (years_ahead * 0.5)  # Slight FDI increase
                    co2_change = baseline_co2 - (years_ahead * 0.2)  # Slight CO2 improvement

                    # Cap values within reasonable ranges
                    export_score = max(1, min(5, export_score))
                    fdi_flows = max(0, min(50, fdi_flows))
                    co2_change = max(-30, min(30, co2_change))

                # Make prediction
                features = np.array([[export_score, fdi_flows, co2_change]])
                gdp_prediction = self.model.predict(features)[0]
                gdp_prediction = max(1, min(5, gdp_prediction))  # Cap GDP score

                predictions.append({
                    'year': year,
                    'Average annual rate of growth of exports (scored 1-5)': round(export_score, 2),
                    'Inward FDI flows (% of fixed investment)': round(fdi_flows, 2),
                    'CO2 emissions: Oil (% change y/y)': round(co2_change, 2),
                    'predicted_gdp_growth': round(gdp_prediction, 2)
                })

            return {
                'success': True,
                'predictions': predictions,
                'baseline_info': {
                    'recent_export_avg': round(baseline_export, 2),
                    'recent_fdi_avg': round(baseline_fdi, 2),
                    'recent_co2_avg': round(baseline_co2, 2)
                }
            }

        except Exception as e:
            logger.error(f"Future prediction error: {e}")
            return {'success': False, 'error': f'Prediction failed: {str(e)}'}

    def generate_prediction_chart(self, predictions: List[Dict]) -> str:
        """Generate base64 encoded chart with prediction, correlation, and 3D surface WITH TREND LINE"""
        try:
            # Create figure with mixed 2D and 3D subplots
            fig = plt.figure(figsize=(20, 12))
            
            # ============== TOP LEFT: PREDICTION CHART WITH TREND LINE ==============
            ax1 = fig.add_subplot(2, 2, 1)
            
            # Historical data
            if self.training_data is not None:
                hist_years = self.training_data['Year'].values
                hist_gdp = self.training_data['Average annual GDP growth (scored 1-5)'].values
                ax1.plot(hist_years, hist_gdp, 'b-o', label='Historical GDP Growth', linewidth=2, markersize=6)

            # Future predictions
            pred_years = [p['year'] for p in predictions]
            pred_gdp = [p['predicted_gdp_growth'] for p in predictions]
            ax1.plot(pred_years, pred_gdp, 'r--s', label='Predicted GDP Growth', linewidth=2, markersize=8)

            # *** ADD TREND LINE FOR GDP PREDICTIONS ***
            if self.training_data is not None and len(predictions) > 0:
                # Combine historical and predicted data for overall trend
                all_years = list(hist_years) + pred_years
                all_gdp = list(hist_gdp) + pred_gdp
                
                # Calculate linear trend (polynomial degree 1)
                z = np.polyfit(all_years, all_gdp, 1)
                trend_function = np.poly1d(z)
                
                # Generate trend line points
                trend_line_years = np.linspace(min(all_years), max(all_years), 100)
                trend_line_gdp = trend_function(trend_line_years)
                
                # Plot the trend line
                ax1.plot(trend_line_years, trend_line_gdp, 'g-', 
                        label=f'GDP Trend Line (slope: {z[0]:.3f})', 
                        linewidth=3, alpha=0.8)
                
                # Add trend equation as text
                trend_equation = f'Trend: y = {z[0]:.3f}x + {z[1]:.1f}'
                ax1.text(0.05, 0.95, trend_equation, 
                        transform=ax1.transAxes, fontsize=10,
                        bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.7),
                        verticalalignment='top')
                
                # Calculate trend direction
                if z[0] > 0.01:
                    trend_direction = "📈 Increasing"
                elif z[0] < -0.01:
                    trend_direction = "📉 Decreasing"
                else:
                    trend_direction = "➡️ Stable"
                    
                ax1.text(0.05, 0.85, f'Trend: {trend_direction}', 
                        transform=ax1.transAxes, fontsize=10,
                        bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.7),
                        verticalalignment='top')

            # Styling for prediction chart
            ax1.set_title('GDP Growth: Historical vs Predicted (with Trend)', fontsize=14, fontweight='bold')
            ax1.set_xlabel('Year', fontsize=12)
            ax1.set_ylabel('GDP Growth Score (1-5)', fontsize=12)
            ax1.legend(fontsize=10)
            ax1.grid(True, alpha=0.3)
            ax1.set_ylim(1, 5)

            # ============== TOP RIGHT: CORRELATION HEATMAP ==============
            ax2 = fig.add_subplot(2, 2, 2)
            
            if self.training_data is not None:
                # Create extended dataset including predictions for correlation
                extended_data = self.training_data.copy()
                
                # Add prediction data to the dataset
                for pred in predictions:
                    new_row = {
                        'Year': pred['year'],
                        'Average annual rate of growth of exports (scored 1-5)': pred['Average annual rate of growth of exports (scored 1-5)'],
                        'Inward FDI flows (% of fixed investment)': pred['Inward FDI flows (% of fixed investment)'],
                        'CO2 emissions: Oil (% change y/y)': pred['CO2 emissions: Oil (% change y/y)'],
                        'Average annual GDP growth (scored 1-5)': pred['predicted_gdp_growth']
                    }
                    extended_data = pd.concat([extended_data, pd.DataFrame([new_row])], ignore_index=True)
                
                # Select columns for correlation
                corr_columns = [
                    'Average annual rate of growth of exports (scored 1-5)',
                    'Inward FDI flows (% of fixed investment)',
                    'CO2 emissions: Oil (% change y/y)',
                    'Average annual GDP growth (scored 1-5)'
                ]
                
                corr_matrix = extended_data[corr_columns].corr()
                
                # Create heatmap
                im = ax2.imshow(corr_matrix, cmap='coolwarm', aspect='auto', vmin=-1, vmax=1)
                
                # Add correlation values as text
                for i in range(len(corr_matrix.columns)):
                    for j in range(len(corr_matrix.columns)):
                        text = ax2.text(j, i, f'{corr_matrix.iloc[i, j]:.2f}',
                                    ha="center", va="center", color="black", fontweight='bold', fontsize=10)
                
                # Shortened labels
                short_labels = ['Export Growth', 'FDI Flows', 'CO2 Change', 'GDP Growth']
                ax2.set_xticks(range(len(corr_matrix.columns)))
                ax2.set_yticks(range(len(corr_matrix.columns)))
                ax2.set_xticklabels(short_labels, rotation=45, ha='right')
                ax2.set_yticklabels(short_labels)
                ax2.set_title('Variables Correlation Matrix\n(Including Predictions)', fontsize=12, fontweight='bold')
                
                # Add colorbar
                cbar = plt.colorbar(im, ax=ax2, shrink=0.8)
                cbar.set_label('Correlation Coefficient', fontsize=10)

            # ============== BOTTOM: 3D REGRESSION SURFACE ==============
            ax3 = fig.add_subplot(2, 1, 2, projection='3d')
            
            if self.training_data is not None:
                # Create meshgrid for Export Growth and FDI Flows
                export_range = np.linspace(1, 5, 25)
                fdi_range = np.linspace(
                    self.training_data['Inward FDI flows (% of fixed investment)'].min(),
                    self.training_data['Inward FDI flows (% of fixed investment)'].max(),
                    25
                )
                X_mesh, Y_mesh = np.meshgrid(export_range, fdi_range)
                
                # Fix CO2 at average value for surface plot
                avg_co2 = self.training_data['CO2 emissions: Oil (% change y/y)'].mean()
                
                # Predict GDP for all combinations
                Z_mesh = np.zeros_like(X_mesh)
                feature_names = [
                    'Average annual rate of growth of exports (scored 1-5)',
                    'Inward FDI flows (% of fixed investment)',
                    'CO2 emissions: Oil (% change y/y)'
                ]
                
                for i in range(X_mesh.shape[0]):
                    for j in range(X_mesh.shape[1]):
                        features_df = pd.DataFrame([[X_mesh[i,j], Y_mesh[i,j], avg_co2]], 
                                                columns=feature_names)
                        Z_mesh[i,j] = self.model.predict(features_df)[0]
                
                # Plot 3D surface
                surf = ax3.plot_surface(X_mesh, Y_mesh, Z_mesh, 
                                    cmap='viridis', alpha=0.8, edgecolor='none')
                
                # Add actual historical data points
                export_data = self.training_data['Average annual rate of growth of exports (scored 1-5)']
                fdi_data = self.training_data['Inward FDI flows (% of fixed investment)']
                gdp_data = self.training_data['Average annual GDP growth (scored 1-5)']
                
                ax3.scatter(export_data, fdi_data, gdp_data, 
                        c='red', s=60, alpha=0.9, label='Historical Data', edgecolors='black')
                
                # Add predicted future points
                if len(predictions) > 0:
                    pred_export = [p['Average annual rate of growth of exports (scored 1-5)'] for p in predictions]
                    pred_fdi = [p['Inward FDI flows (% of fixed investment)'] for p in predictions]
                    pred_gdp_vals = [p['predicted_gdp_growth'] for p in predictions]
                    
                    ax3.scatter(pred_export, pred_fdi, pred_gdp_vals,
                            c='yellow', s=80, alpha=1.0, label='Future Predictions', 
                            marker='^', edgecolors='black')
                
                # Styling for 3D plot
                ax3.set_xlabel('Export Growth Score (1-5)', fontsize=12)
                ax3.set_ylabel('FDI Flows (% of Investment)', fontsize=12)
                ax3.set_zlabel('GDP Growth Score (1-5)', fontsize=12)
                ax3.set_title(f'3D GDP Prediction Surface\n(CO2 emissions fixed at {avg_co2:.1f}%)', 
                            fontsize=14, fontweight='bold')
                
                # Add colorbar
                cbar3d = plt.colorbar(surf, ax=ax3, shrink=0.6, aspect=20)
                cbar3d.set_label('Predicted GDP Growth', fontsize=10)
                
                ax3.legend()
                ax3.view_init(elev=20, azim=45)

            # Overall styling
            fig.suptitle('Complete GDP Analysis: Predictions, Correlations & 3D Model Surface', 
                        fontsize=18, fontweight='bold', y=0.95)
            plt.tight_layout()
            plt.subplots_adjust(top=0.90, hspace=0.3, wspace=0.3)

            # Convert to base64
            buffer = io.BytesIO()
            plt.savefig(buffer, format='png', dpi=150, bbox_inches='tight')
            buffer.seek(0)
            chart_base64 = base64.b64encode(buffer.getvalue()).decode()
            plt.close(fig)

            logger.info("Successfully generated comprehensive prediction charts with trend line")
            return chart_base64

        except Exception as e:
            logger.error(f"Comprehensive chart generation error: {e}")
            return ""

    def parse_stress_test_query(self, query: str) -> Dict:
            """Parse natural language query for stress testing"""
            query_lower = query.lower()

            # Extract variables and changes
            changes = {}

            # Export growth patterns
            export_patterns = [
                r'export.{0,20}(?:increase|rise|grow|up).{0,20}(\d+(?:\.\d+)?)',
                r'export.{0,20}(?:decrease|fall|drop|down).{0,20}(\d+(?:\.\d+)?)',
                r'(\d+(?:\.\d+)?).{0,20}(?:increase|rise|grow|up).{0,20}export',
                r'(\d+(?:\.\d+)?).{0,20}(?:decrease|fall|drop|down).{0,20}export'
            ]

            # FDI patterns
            fdi_patterns = [
                r'fdi.{0,20}(?:increase|rise|grow|up).{0,20}(\d+(?:\.\d+)?)',
                r'fdi.{0,20}(?:decrease|fall|drop|down).{0,20}(\d+(?:\.\d+)?)',
                r'investment.{0,20}(?:increase|rise|grow|up).{0,20}(\d+(?:\.\d+)?)',
                r'investment.{0,20}(?:decrease|fall|drop|down).{0,20}(\d+(?:\.\d+)?)'
            ]

            # CO2 patterns
            co2_patterns = [
                r'co2.{0,20}(?:increase|rise|grow|up).{0,20}(\d+(?:\.\d+)?)',
                r'co2.{0,20}(?:decrease|fall|drop|down).{0,20}(\d+(?:\.\d+)?)',
                r'emission.{0,20}(?:increase|rise|grow|up).{0,20}(\d+(?:\.\d+)?)',
                r'emission.{0,20}(?:decrease|fall|drop|down).{0,20}(\d+(?:\.\d+)?)'
            ]

            # Check for export changes
            for pattern in export_patterns:
                match = re.search(pattern, query_lower)
                if match:
                    value = float(match.group(1))
                    if 'decrease' in pattern or 'fall' in pattern or 'drop' in pattern or 'down' in pattern:
                        value = -value
                    changes['export_growth'] = value
                    break

            # Check for FDI changes
            for pattern in fdi_patterns:
                match = re.search(pattern, query_lower)
                if match:
                    value = float(match.group(1))
                    if 'decrease' in pattern or 'fall' in pattern or 'drop' in pattern or 'down' in pattern:
                        value = -value
                    changes['fdi_flows'] = value
                    break

            # Check for CO2 changes
            for pattern in co2_patterns:
                match = re.search(pattern, query_lower)
                if match:
                    value = float(match.group(1))
                    if 'decrease' in pattern or 'fall' in pattern or 'drop' in pattern or 'down' in pattern:
                        value = -value
                    changes['co2_change'] = value
                    break

            return changes

    def stress_test_from_query(self, query: str, target_year: int = None, predictions: List[Dict] = None) -> Dict:
        """Perform stress testing based on natural language query for a specific year"""
        if not self.is_trained:
            return {'success': False, 'error': 'Model not trained yet'}

        try:
            # Parse the query
            changes = self.parse_stress_test_query(query)

            if not changes:
                return {
                    'success': False,
                    'error': 'Could not understand the stress test query. Try asking like: "What if export growth increases by 1 point?" or "What if FDI decreases by 5%?"'
                }

            # Use specific year's predicted values as base scenario if provided
            if target_year and predictions:
                # Find the prediction for the target year
                target_prediction = None
                for pred in predictions:
                    if pred['year'] == target_year:
                        target_prediction = pred
                        break
                
                if target_prediction:
                    # Use the predicted values for the target year as base scenario
                    base_export = target_prediction['Average annual rate of growth of exports (scored 1-5)']
                    base_fdi = target_prediction['Inward FDI flows (% of fixed investment)']
                    base_co2 = target_prediction['CO2 emissions: Oil (% change y/y)']
                    base_gdp = target_prediction['predicted_gdp_growth']
                    scenario_source = f"{target_year} Predicted Values"
                else:
                    return {
                        'success': False,
                        'error': f'No prediction found for year {target_year}. Please generate predictions first.'
                    }
            else:
                # Fall back to recent historical averages (original behavior)
                recent_data = self.training_data.tail(5)
                base_export = recent_data['Average annual rate of growth of exports (scored 1-5)'].mean()
                base_fdi = recent_data['Inward FDI flows (% of fixed investment)'].mean()
                base_co2 = recent_data['CO2 emissions: Oil (% change y/y)'].mean()
                
                # Calculate base GDP prediction
                features_df = pd.DataFrame([[base_export, base_fdi, base_co2]], columns=self.feature_names)
                base_gdp = self.model.predict(features_df)[0]
                scenario_source = "Recent Historical Average"

            # Apply stress changes
            stressed_export = base_export + changes.get('export_growth', 0)
            stressed_fdi = base_fdi + changes.get('fdi_flows', 0)
            stressed_co2 = base_co2 + changes.get('co2_change', 0)

            # Cap values within reasonable ranges
            stressed_export = max(1, min(5, stressed_export))
            stressed_fdi = max(0, min(50, stressed_fdi))
            stressed_co2 = max(-30, min(30, stressed_co2))

            # Make stressed prediction
            stressed_features = pd.DataFrame([[stressed_export, stressed_fdi, stressed_co2]], 
                                           columns=self.feature_names)
            stressed_prediction = self.model.predict(stressed_features)[0]

            # Calculate impact
            impact = stressed_prediction - base_gdp

            return {
                'success': True,
                'query': query,
                'target_year': target_year if target_year else "Current/Recent",
                'scenario_source': scenario_source,
                'changes_detected': changes,
                'base_scenario': {
                    'export_growth': round(base_export, 2),
                    'fdi_flows': round(base_fdi, 2),
                    'co2_change': round(base_co2, 2),
                    'gdp_prediction': round(base_gdp, 2)
                },
                'stressed_scenario': {
                    'export_growth': round(stressed_export, 2),
                    'fdi_flows': round(stressed_fdi, 2),
                    'co2_change': round(stressed_co2, 2),
                    'gdp_prediction': round(stressed_prediction, 2)
                },
                'impact': {
                    'gdp_change': round(impact, 3),
                    'percentage_change': round((impact / base_gdp) * 100, 2) if base_gdp != 0 else 0,
                    'impact_description': self._describe_impact(impact)
                }
            }

        except Exception as e:
            logger.error(f"Stress test error: {e}")
            return {'success': False, 'error': f'Stress test failed: {str(e)}'}


    def _describe_impact(self, impact: float) -> str:
        """Describe the impact of stress test"""
        if abs(impact) < 0.1:
            return "Minimal impact on GDP growth"
        elif impact > 0.5:
            return "Significant positive impact on GDP growth"
        elif impact > 0.2:
            return "Moderate positive impact on GDP growth"
        elif impact > 0:
            return "Small positive impact on GDP growth"
        elif impact < -0.5:
            return "Significant negative impact on GDP growth"
        elif impact < -0.2:
            return "Moderate negative impact on GDP growth"
        else:
            return "Small negative impact on GDP growth"

class PerplexityClient:
    """Enhanced Perplexity client for economic insights"""

    def __init__(self, api_key: str):
        self.api_key = api_key
        self.base_url = "https://api.perplexity.ai/chat/completions"
        self.headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json"
        }

    async def generate_insights(self, context: Dict, question: str) -> str:
        """Generate AI insights with comprehensive context"""

        system_prompt = f"""
        You are an expert economic analyst specializing in Southeast Asian GDP growth forecasting.

        Context Information:
        - Model Performance: {context.get('model_performance', {})}
        - Recent Predictions: {context.get('predictions', {})}
        - Stress Test Results: {context.get('stress_test', {})}

        Guidelines:
        1. Focus on GDP growth, economic forecasting, and Southeast Asian economies
        2. Provide specific, data-driven insights using the context
        3. Explain economic relationships and policy implications
        4. Keep responses informative but concise (max 400 words)
        5. If asked about non-economic topics, redirect to economic analysis

        Answer the user's question using this economic context and your expertise.
        """

        payload = {
            "model": "llama-3.1-sonar-small-128k-online",
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": question}
            ],
            "temperature": 0.3,
            "max_tokens": 600
        }

        async with httpx.AsyncClient(timeout=30.0) as client:
            try:
                response = await client.post(self.base_url, json=payload, headers=self.headers)
                response.raise_for_status()
                result = response.json()
                return result['choices'][0]['message']['content']
            except Exception as e:
                logger.error(f"Perplexity API error: {e}")
                return f"Unable to generate AI insights at this time. Please try again later."

class CompleteMCPServer:
    """Complete MCP Server with all GDP prediction features"""

    def __init__(self, perplexity_api_key: str):
        self.gdp_model = GDPForecastModel()
        self.perplexity = PerplexityClient(perplexity_api_key)
        self.csv_processor = CSVProcessor()

    async def handle_csv_upload(self, file_content: bytes, filename: str) -> Dict:
        """Handle CSV upload and model training"""
        try:
            logger.info(f"Processing uploaded CSV file: {filename}")

            # Process CSV file
            success, result = self.csv_processor.process_csv_file(file_content)
            logger.info(f"Processing uploaded CSV file:result {result}")
            if not success:
                return {
                    'success': False,
                    'error': result,
                    'message': 'CSV file processing failed'
                }

            # Train model with new data
            df = result
            training_result = self.gdp_model.train_from_csv(df)

            if training_result['success']:
                return {
                    'success': True,
                    'message': f'Successfully trained model with {len(df)} records',
                    'training_result': training_result,
                    'filename': filename,
                    'data_preview': df.head(25).to_dict('records'),
                    'data_summary': {
                        'total_records': len(df),
                        'year_range': f"{df['Year'].min()}-{df['Year'].max()}",
                        'avg_gdp_growth': round(df['Average annual GDP growth (scored 1-5)'].mean(), 2),
                        'avg_export_growth': round(df['Average annual rate of growth of exports (scored 1-5)'].mean(), 2),
                        'avg_fdi_flows': round(df['Inward FDI flows (% of fixed investment)'].mean(), 2),
                        'avg_co2_change': round(df['CO2 emissions: Oil (% change y/y)'].mean(), 2)
                    }
                }
            else:
                return training_result

        except Exception as e:
            logger.error(f"CSV upload error: {e}")
            return {
                'success': False,
                'error': f'Upload processing failed: {str(e)}'
            }

    async def handle_future_predictions(self, years_ahead: int = 5, scenarios: Dict = None) -> Dict:
        """Generate predictions for upcoming years with charts"""
        try:
            if not self.gdp_model.is_trained:
                return {
                    'success': False,
                    'error': 'Please upload and train the model with CSV data first'
                }

            # Generate future years
            last_year = self.gdp_model.training_data['Year'].max()
            future_years = list(range(last_year + 1, last_year + years_ahead + 1))

            # Get predictions
            pred_result = self.gdp_model.predict_future_years(future_years, scenarios)

            if pred_result['success']:
                # Generate chart
                chart_base64 = self.gdp_model.generate_prediction_chart(pred_result['predictions'])

                return {
                    'success': True,
                    'predictions': pred_result['predictions'],
                    'baseline_info': pred_result['baseline_info'],
                    'chart': chart_base64,
                    'summary': {
                        'prediction_years': f"{future_years[0]}-{future_years[-1]}",
                        'avg_predicted_gdp': round(np.mean([p['predicted_gdp_growth'] for p in pred_result['predictions']]), 2),
                        'gdp_trend': self._analyze_trend(pred_result['predictions'])
                    }
                }
            else:
                return pred_result

        except Exception as e:
            logger.error(f"Future predictions error: {e}")
            return {
                'success': False,
                'error': f'Future predictions failed: {str(e)}'
            }

    async def handle_stress_test_query(self, query: str, target_year: int = None) -> Dict:
        """Handle natural language stress testing queries for a specific year"""
        try:
            if not self.gdp_model.is_trained:
                return {
                    'success': False,
                    'error': 'Please upload and train the model with CSV data first'
                }

            # Get predictions if target year is specified
            predictions = None
            if target_year:
                # Generate predictions to get the target year's base scenario
                last_year = self.gdp_model.training_data['Year'].max()
                future_years = list(range(last_year + 1, target_year + 1))
                pred_result = self.gdp_model.predict_future_years(future_years)
                
                if pred_result['success']:
                    predictions = pred_result['predictions']
                else:
                    return {
                        'success': False,
                        'error': f'Could not generate predictions for {target_year}: {pred_result.get("error", "Unknown error")}'
                    }

            # Call with keyword arguments
            result = self.gdp_model.stress_test_from_query(
                query, 
                target_year=target_year, 
                predictions=predictions
            )
            return result

        except Exception as e:
            logger.error(f"Stress test query error: {e}")
            return {
                'success': False,
                'error': f'Stress test failed: {str(e)}'
            }

    async def handle_ai_insights(self, question: str, context: Dict = None) -> Dict:
        """Generate AI insights with full context"""
        try:
            # Build comprehensive context
            full_context = {}

            if self.gdp_model.is_trained:
                full_context['model_performance'] = self.gdp_model.model_performance
                full_context['coefficients'] = self.gdp_model.coefficients
                full_context['training_period'] = f"{self.gdp_model.training_data['Year'].min()}-{self.gdp_model.training_data['Year'].max()}"

            if context:
                full_context.update(context)

            # Check if question is economics-related
            if not self._is_economics_question(question):
                return {
                    'success': False,
                    'error': 'Please ask questions related to GDP growth, economic forecasting, or Southeast Asian economic analysis.',
                    'suggestion': 'Try asking about: model performance, economic trends, policy impacts, or forecasting accuracy.'
                }

            # Generate insights
            ai_response = await self.perplexity.generate_insights(full_context, question)

            return {
                'success': True,
                'ai_response': ai_response,
                'question': question,
                'context_used': bool(full_context),
                'model_trained': self.gdp_model.is_trained
            }

        except Exception as e:
            logger.error(f"AI insights error: {e}")
            return {
                'success': False,
                'error': f'AI insights generation failed: {str(e)}'
            }

    def get_model_status(self) -> Dict:
        """Get comprehensive model status"""
        if not self.gdp_model.is_trained:
            return {
                'is_trained': False,
                'message': 'Model not trained yet. Please upload CSV data.'
            }

        return {
            'is_trained': True,
            'training_timestamp': self.gdp_model.training_timestamp.isoformat(),
            'performance': self.gdp_model.model_performance,
            'coefficients': self.gdp_model.coefficients,
            'data_summary': {
                'records_count': len(self.gdp_model.training_data),
                'year_range': f"{self.gdp_model.training_data['Year'].min()}-{self.gdp_model.training_data['Year'].max()}",
                'features': self.gdp_model.feature_names
            }
        }

    def _analyze_trend(self, predictions: List[Dict]) -> str:
        """Analyze trend in predictions"""
        gdp_values = [p['predicted_gdp_growth'] for p in predictions]

        if len(gdp_values) < 2:
            return "Insufficient data for trend analysis"

        slope = np.polyfit(range(len(gdp_values)), gdp_values, 1)[0]

        if slope > 0.1:
            return "Increasing GDP growth trend"
        elif slope < -0.1:
            return "Decreasing GDP growth trend"
        else:
            return "Stable GDP growth trend"

    def _is_economics_question(self, question: str) -> bool:
        """Check if question is economics-related"""
        economic_keywords = [
            'gdp', 'economy', 'economic', 'growth', 'forecast', 'prediction',
            'export', 'fdi', 'investment', 'co2', 'emissions', 'trade',
            'policy', 'inflation', 'recession', 'market', 'development',
            'southeast asia', 'asean', 'stress', 'scenario', 'model',
            'training', 'data', 'performance', 'coefficient'
        ]

        question_lower = question.lower()
        return any(keyword in question_lower for keyword in economic_keywords)

# Example usage
async def main():
    """Main server demonstration"""
    PERPLEXITY_API_KEY = "your-perplexity-api-key-here"  # Replace with actual key

    server = CompleteMCPServer(PERPLEXITY_API_KEY)

    print("🌏 Complete GDP Growth Prediction MCP Server")
    print("=" * 50)
    print("Features:")
    print("✅ CSV upload and model training")
    print("✅ 5-year future predictions with charts")
    print("✅ Natural language stress testing")
    print("✅ AI-powered economic insights")
    print("✅ Comprehensive model analytics")
    print("")
    print("Available endpoints:")
    print("- /upload-csv: Upload CSV data and train model")
    print("- /predict-future: Generate 5-year predictions with charts")
    print("- /stress-test: Natural language stress testing")
    print("- /ai-insights: AI-powered economic analysis")
    print("- /model-status: Get model information and performance")

if __name__ == "__main__":
    asyncio.run(main())
