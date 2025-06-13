from flask import Flask, request, jsonify
from flask_cors import CORS
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import PolynomialFeatures
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.pipeline import Pipeline
import json
import google.generativeai as genai
import os
from datetime import datetime

app = Flask(__name__)
CORS(app)

GEMINI_API_KEY = os.environ.get('GEMINI_API_KEY')
genai.configure(api_key=GEMINI_API_KEY)

model = genai.GenerativeModel('gemini-2.0-flash')

def generate_enhanced_data(n_samples=150):
    """Generate more realistic house data with multiple features"""
    np.random.seed(50)

    # Generate correlated features
    size = np.random.normal(1400, 300, n_samples)
    bedrooms = np.clip(np.round(size/400 + np.random.normal(0, 0.5, n_samples)), 1, 6)
    bathrooms = np.clip(bedrooms/2 + np.random.normal(0, 0.3, n_samples), 1, 4)
    age = np.random.uniform(0, 50, n_samples)
    location_score = np.random.uniform(1, 10, n_samples)

    # More complex price calculation
    base_price = size * 80
    bedroom_bonus = bedrooms * 15000
    bathroom_bonus = bathrooms * 12000
    age_penalty = age * 500
    location_bonus = location_score * 8000
    noise = np.random.normal(0, 15000, n_samples)

    price = base_price + bedroom_bonus + bathroom_bonus - age_penalty + location_bonus + noise
    price = np.clip(price, 50000, 800000)  # Realistic price range

    return pd.DataFrame({
        'size': size,
        'bedrooms': bedrooms,
        'bathrooms': bathrooms,
        'age': age,
        'location_score': location_score,
        'price': price
    })

# Define the specific feature names for the full model here, consistent with the DataFrame creation
FULL_FEATURE_NAMES = ['size', 'bedrooms', 'bathrooms', 'age', 'location_score']
SIMPLE_FEATURE_NAME = ['size'] # Define feature name for simple models

def train_models():
    """Train multiple models and return the best one with metrics"""
    df = generate_enhanced_data(n_samples=150)

    # Features for different models
    X_simple = df[SIMPLE_FEATURE_NAME] # Ensure this is a DataFrame with column name
    X_full = df[FULL_FEATURE_NAMES] # Ensure this is a DataFrame with column names
    y = df['price']

    # Split data
    X_simple_train, X_simple_test, y_train, y_test = train_test_split(
        X_simple, y, test_size=0.2, random_state=42
    )
    X_full_train, X_full_test, _, _ = train_test_split(
        X_full, y, test_size=0.2, random_state=42
    )

    models = {}

    # Simple Linear Regression
    simple_model = LinearRegression()
    simple_model.fit(X_simple_train, y_train)
    simple_pred = simple_model.predict(X_simple_test)

    models['simple'] = {
        'model': simple_model,
        'mse': mean_squared_error(y_test, simple_pred),
        'r2': r2_score(y_test, simple_pred),
        'mae': mean_absolute_error(y_test, simple_pred),
        'features': SIMPLE_FEATURE_NAME
    }

    # Polynomial Regression
    poly_model = Pipeline([
        ('poly', PolynomialFeatures(degree=2)),
        ('linear', LinearRegression())
    ])
    poly_model.fit(X_simple_train, y_train)
    poly_pred = poly_model.predict(X_simple_test)

    models['polynomial'] = {
        'model': poly_model,
        'mse': mean_squared_error(y_test, poly_pred),
        'r2': r2_score(y_test, poly_pred),
        'mae': mean_absolute_error(y_test, poly_pred),
        'features': SIMPLE_FEATURE_NAME
    }

    # Multi-feature Linear Regression
    multi_model = LinearRegression()
    multi_model.fit(X_full_train, y_train)
    multi_pred = multi_model.predict(X_full_test)

    models['multi_linear'] = {
        'model': multi_model,
        'mse': mean_squared_error(y_test, multi_pred),
        'r2': r2_score(y_test, multi_pred),
        'mae': mean_absolute_error(y_test, multi_pred),
        'features': FULL_FEATURE_NAMES
    }

    # Random Forest
    rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
    rf_model.fit(X_full_train, y_train)
    rf_pred = rf_model.predict(X_full_test)

    models['random_forest'] = {
        'model': rf_model,
        'mse': mean_squared_error(y_test, rf_pred),
        'r2': r2_score(y_test, rf_pred),
        'mae': mean_absolute_error(y_test, rf_pred),
        'features': FULL_FEATURE_NAMES
    }

    return models, df

def get_ai_insights(prediction_data, model_performance):
    """Generate AI insights using Gemini API"""
    try:
        prompt = f"""
        As a real estate AI expert, analyze this house price prediction and provide insights:

        House Details:
        - Size: {prediction_data.get('size', 'N/A')} sq ft
        - Bedrooms: {prediction_data.get('bedrooms', 'N/A')}
        - Bathrooms: {prediction_data.get('bathrooms', 'N/A')}
        - Age: {prediction_data.get('age', 'N/A')} years
        - Location Score: {prediction_data.get('location_score', 'N/A')}/10
        - Predicted Price: ${prediction_data.get('prediction', 0):,.0f}
        - Model Used: {prediction_data.get('model_type', 'N/A')}
        - Model R² Score: {model_performance.get('r2_score', 0):.3f}

        Please provide:
        1. Market Analysis (2-3 sentences)
        2. Investment Recommendation (Buy/Hold/Sell with reasoning)
        3. Key Factors affecting the price
        4. Potential risks or opportunities
        5. Comparison with market trends

        Keep the response concise, professional, and actionable. Format as JSON with keys: market_analysis, recommendation, key_factors, risks_opportunities, market_comparison.
        """

        response = model.generate_content(prompt)

        # Try to parse as JSON, fallback to structured text
        try:
            return json.loads(response.text)
        except:
            # Fallback structured response
            return {
                "market_analysis": "AI analysis generated based on current market conditions and property features.",
                "recommendation": "Consult with local real estate experts for personalized advice.",
                "key_factors": ["Property size", "Location score", "Age of property"],
                "risks_opportunities": "Market conditions may vary based on local factors.",
                "market_comparison": "Property appears to be within reasonable market range."
            }
    except Exception as e:
        print(f"AI Insights Error: {e}")
        return {
            "market_analysis": "Unable to generate AI insights at this time.",
            "recommendation": "Please consult with a real estate professional.",
            "key_factors": ["Property size", "Location", "Market conditions"],
            "risks_opportunities": "Consider local market trends and economic factors.",
            "market_comparison": "Market analysis temporarily unavailable."
        }

def get_market_trends_analysis():
    """Generate market trends analysis using Gemini AI"""
    try:
        prompt = """
        As a real estate market analyst, provide current insights about the housing market:

        Please analyze:
        1. Current market trends (3-4 key trends)
        2. Price predictions for the next 6 months
        3. Best investment strategies
        4. Factors driving market changes

        Keep it concise and actionable. Format as JSON with keys: trends, predictions, strategies, driving_factors.
        """

        response = model.generate_content(prompt)

        try:
            return json.loads(response.text)
        except:
            return {
                "trends": [
                    "Interest rates affecting buyer demand",
                    "Supply chain impacts on construction",
                    "Remote work changing location preferences",
                    "Sustainable housing gaining popularity"
                ],
                "predictions": "Market expected to stabilize with moderate growth",
                "strategies": "Focus on location, amenities, and energy efficiency",
                "driving_factors": "Economic conditions, demographics, and policy changes"
            }
    except Exception as e:
        print(f"Market Trends Error: {e}")
        return {
            "trends": ["Market analysis temporarily unavailable"],
            "predictions": "Consult current market reports",
            "strategies": "Diversify investment portfolio",
            "driving_factors": "Multiple economic and social factors"
        }

# Initialize models and data globally when the app starts
models, data = train_models()


@app.route('/api/predict', methods=['POST'])
def predict():
    try:
        request_data = request.get_json()
        model_type = request_data.get('model', 'simple')

        if model_type not in models:
            return jsonify({'error': 'Invalid model type'}), 400

        model_info = models[model_type]
        model = model_info['model']

        if model_type == 'simple' or model_type == 'polynomial':
            size = request_data['size']
            # For simple and polynomial models, create a DataFrame with the 'size' column name
            prediction_input = pd.DataFrame([[size]], columns=SIMPLE_FEATURE_NAME)
            input_features = {'size': size}
        else:
            # Multi-feature prediction: Construct a DataFrame with explicit feature names
            size = request_data['size']
            bedrooms = request_data.get('bedrooms', 3)
            bathrooms = request_data.get('bathrooms', 2)
            age = request_data.get('age', 10)
            location_score = request_data.get('location_score', 7)

            # Create a DataFrame for prediction, ensuring correct column names
            prediction_input = pd.DataFrame(
                [[size, bedrooms, bathrooms, age, location_score]],
                columns=FULL_FEATURE_NAMES
            )
            input_features = {
                'size': size,
                'bedrooms': bedrooms,
                'bathrooms': bathrooms,
                'age': age,
                'location_score': location_score
            }

        prediction = model.predict(prediction_input)[0]

        # Prepare prediction data for AI insights
        prediction_data = {
            **input_features,
            'prediction': float(prediction),
            'model_type': model_type
        }

        # Get AI insights
        ai_insights = get_ai_insights(prediction_data, model_info)

        return jsonify({
            'prediction': float(prediction),
            'model_type': model_type,
            'input_features': input_features,
            'model_metrics': {
                'r2_score': model_info['r2'],
                'mse': model_info['mse'],
                'mae': model_info['mae']
            },
            'ai_insights': ai_insights,
            'timestamp': datetime.now().isoformat()
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 400

@app.route('/api/ai_market_analysis', methods=['GET'])
def ai_market_analysis():
    try:
        market_analysis = get_market_trends_analysis()
        return jsonify({
            'market_analysis': market_analysis,
            'timestamp': datetime.now().isoformat()
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 400

@app.route('/api/ai_insights', methods=['POST'])
def ai_insights():
    try:
        request_data = request.get_json()
        question = request_data.get('question', '')

        if not question:
            return jsonify({'error': 'No question provided'}), 400

        prompt = f"""
        As a real estate AI assistant, answer this question about house pricing and real estate:

        Question: {question}

        Provide a helpful, accurate, and concise response. If the question is about specific market data,
        mention that current market conditions may vary and recommend consulting local experts.
        """

        response = model.generate_content(prompt)

        return jsonify({
            'question': question,
            'answer': response.text,
            'timestamp': datetime.now().isoformat()
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 400

@app.route('/api/regression_line', methods=['POST'])
def get_regression_line():
    try:
        request_data = request.get_json()
        model_type = request_data.get('model', 'simple')

        if model_type not in models:
            return jsonify({'error': 'Invalid model type'}), 400

        model_info = models[model_type]
        model = model_info['model']

        # Generate points for regression line
        size_range = np.linspace(800, 2200, 100)

        if model_type == 'simple' or model_type == 'polynomial':
            # Create a DataFrame for generating the regression line for simple/polynomial
            predictions_input = pd.DataFrame(size_range.reshape(-1, 1), columns=SIMPLE_FEATURE_NAME)
            predictions = model.predict(predictions_input)
        else:
            # For multi-feature models, use average values for other features
            avg_bedrooms = data['bedrooms'].mean()
            avg_bathrooms = data['bathrooms'].mean()
            avg_age = data['age'].mean()
            avg_location = data['location_score'].mean()

            # Create a DataFrame for generating the regression line, maintaining feature names
            features_for_line = pd.DataFrame(
                np.column_stack([
                    size_range,
                    np.full(len(size_range), avg_bedrooms),
                    np.full(len(size_range), avg_bathrooms),
                    np.full(len(size_range), avg_age),
                    np.full(len(size_range), avg_location)
                ]),
                columns=FULL_FEATURE_NAMES
            )
            predictions = model.predict(features_for_line)

        return jsonify({
            'size_range': size_range.tolist(),
            'predictions': predictions.tolist(),
            'model_type': model_type
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 400

@app.route('/api/data', methods=['GET'])
def get_data():
    try:
        return jsonify({
            'data': data.to_dict('records')
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 400

@app.route('/api/models', methods=['GET'])
def get_models():
    try:
        model_comparison = {}
        for name, info in models.items():
            model_comparison[name] = {
                'r2_score': info['r2'],
                'mse': info['mse'],
                'mae': info['mae'],
                'features': info['features']
            }

        return jsonify({
            'models': model_comparison
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 400

@app.route('/api/feature_importance', methods=['GET'])
def get_feature_importance():
    try:
        rf_model = models['random_forest']['model']
        features = models['random_forest']['features']

        importance = rf_model.feature_importances_

        return jsonify({
            'features': features,
            'importance': importance.tolist()
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 400

@app.route('/api/price_analysis', methods=['POST'])
def price_analysis():
    try:
        request_data = request.get_json()
        size = request_data['size']

        # Get predictions from all models for comparison
        predictions = {}

        for model_name, model_info in models.items():
            model = model_info['model']

            if model_name in ['simple', 'polynomial']:
                # Use DataFrame for simple and polynomial models too
                pred_df = pd.DataFrame([[size]], columns=SIMPLE_FEATURE_NAME)
                pred = model.predict(pred_df)
            else:
                # Use default values for other features, creating a DataFrame for prediction
                features_data = [[size, 3, 2, 10, 7]]
                pred_df = pd.DataFrame(features_data, columns=FULL_FEATURE_NAMES)
                pred = model.predict(pred_df)

            predictions[model_name] = float(pred[0])

        # Calculate price per sq ft
        price_per_sqft = {name: pred/size for name, pred in predictions.items()}

        # Market analysis based on size
        if size < 1000:
            market_segment = "Compact/Starter Home"
        elif size < 1500:
            market_segment = "Mid-Range Family Home"
        elif size < 2000:
            market_segment = "Large Family Home"
        else:
            market_segment = "Luxury/Executive Home"

        return jsonify({
            'predictions': predictions,
            'price_per_sqft': price_per_sqft,
            'market_segment': market_segment,
            'size': size
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 400

@app.route('/api/health', methods=['GET'])
def health():
    return jsonify({'status': 'healthy', 'ai_enabled': GEMINI_API_KEY != 'your-gemini-api-key-here'})

if __name__ == '__main__':
    print("🚀 Starting AI-Powered House Price Predictor...")
    print(f"🤖 AI Features: {'Enabled' if GEMINI_API_KEY != 'your-gemini-api-key-here' else 'Disabled (Set GEMINI_API_KEY)'}")
    app.run(debug=True, port=5000)