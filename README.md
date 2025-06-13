# 🏠 AI-Powered House Price Predictor

A full-stack machine learning web application to predict house prices based on various input features, now enhanced with *AI-driven insights and market analysis. Built with **React* on the frontend and *Flask* on the backend, it leverages a trained ML model for real-time predictions and the *Google Gemini API* for expert commentary.

---

## ✨ Features

* *Real-time Price Prediction*: Predict house prices using multiple regression models (Linear, Polynomial, Random Forest).
* *Interactive Data Visualization*: Explore historical data and model regression lines.
* *Model Comparison*: View and compare performance metrics (R², MSE, MAE) of different ML models.
* *Feature Importance*: Understand which factors most influence house prices (for Random Forest model).
* *AI-Powered Insights (Gemini API)*:
    * *Property-Specific Analysis*: Get market analysis, investment recommendations, key factors, risks, and opportunities for a predicted house.
    * *General Market Trends*: Obtain current housing market trends, future predictions, investment strategies, and driving factors.
    * *Interactive AI Assistant*: Ask general questions about real estate and house pricing.
* *Modern UI: Built with **Tailwind CSS v3* for a clean, responsive, and customizable user interface.

---

## 📁 Project Structure
<pre> ```text house-price-predictor/ ├── backend/ # Flask server, ML model, and AI integration │ ├── app.py # Main Flask application with ML model training and AI logic │ ├── .env.example # Example for backend environment variables (e.g., API keys) │ └── requirements.txt # Python dependencies ├── frontend/ # React frontend │ ├── public/ # Public assets │ ├── src/ # React components, styles, and logic │ │ └── App.css # Main CSS file, includes Tailwind directives │ ├── .env.example # Example for frontend environment variables (e.g., API base URL) │ ├── package.json # Frontend dependencies and scripts │ ├── tailwind.config.js # Tailwind CSS configuration │ └── postcss.config.js # PostCSS configuration for Tailwind └── README.md # Project documentation ``` </pre>

house-price-predictor/
├── backend/                  # Flask server, ML model, and AI integration
│   ├── app.py                # Main Flask application with ML model training and AI logic
│   ├── .env.example          # Example for backend environment variables (e.g., API keys)
│   └── requirements.txt      # Python dependencies
├── frontend/                 # React frontend
│   ├── public/               # Public assets
│   ├── src/                  # React components, styles, and logic
│   │   └── App.css           # Main CSS file, includes Tailwind directives
│   ├── .env.example          # Example for frontend environment variables (e.g., API base URL)
│   ├── package.json          # Frontend dependencies and scripts
│   ├── tailwind.config.js    # Tailwind CSS configuration
│   └── postcss.config.js     # PostCSS configuration for Tailwind
└── README.md                 # Project documentation


---

## 🛠 Tech Stack

* *Frontend: React, JavaScript, HTML, **Tailwind CSS v3, **Recharts*
* *Backend*: Flask (Python), Scikit-learn (Regression Models), Pandas, NumPy
* *Artificial Intelligence: **Google Gemini API*
* *Communication*: RESTful API (Flask)

---

## 🚀 Getting Started

### ✅ Prerequisites

Make sure you have the following installed on your system:

* [Node.js](https://nodejs.org/) (LTS recommended)
* [Python 3.8+](https://www.python.org/downloads/)
* pip (Python package installer)
* (Optional but Recommended) git for cloning the repository

---

### 📦 Backend Setup (Flask + ML + AI)

1.  *Navigate to backend folder*:

    ```
    cd house-price-predictor/backend
    ```
    

2.  *Create and activate a Python virtual environment (highly recommended)*:

    ```
    python -m venv venv
    ```

    # On Windows:
    venv\Scripts\activate
    # On macOS/Linux:
    source venv/bin/activate
    

3.  *Install backend dependencies*:
    ```
    pip install -r requirements.txt
    ```
    
    (This will install Flask, Flask-CORS, pandas, numpy, scikit-learn, and google-generativeai).

    **requirements.txt content:**
    
    flask==2.3.3
    flask-cors==4.0.0
    pandas==2.1.1
    numpy==1.26.4
    scikit-learn==1.3.2
    google-generativeai==0.3.0 # Or the latest stable version
    python-dotenv==1.0.0      # For loading .env files (ensure it's present)
    

4.  *Set up environment variables*:
    Create a file named .env in the backend directory (next to app.py).
    Obtain your API key from the [Google AI Studio](https://aistudio.google.com/app/apikey) and add it to the .env file:
    dotenv
    # backend/.env
    GEMINI_API_KEY='YOUR_YOUR_GEMINI_API_KEY_HERE'
    DEBUG=True # Set to False in production
    
    *Note*: If you don't set GEMINI_API_KEY, AI features will be disabled by default.

5.  *Run the Flask server*:
    ```
    python app.py
    ```

    By default, the server runs at:
    👉 http://127.0.0.1:5000

---

### 💻 Frontend Setup (React + Tailwind CSS v3)

1.  *Navigate to frontend folder*:
    ```
    cd house-price-predictor/frontend
    ```
    

2.  *Install frontend dependencies*:
    ```
    npm install
    ```

    # Or if you use yarn: yarn install
    
    This will install react, react-dom, recharts, and other standard React dependencies.

3.  *Install Tailwind CSS v3 and its peer dependencies*:
    ```
    npm install -D tailwindcss@3 postcss autoprefixer
    ```

    # Or if you use yarn: yarn add -D tailwindcss@3 postcss autoprefixer
    

4.  *Initialize Tailwind CSS configuration files*:
    ```
    npx tailwindcss init -p
    ```

    This command will create tailwind.config.js and postcss.config.js in your frontend directory.

5.  **Configure tailwind.config.js**:
    Open tailwind.config.js and ensure the content array includes paths to all your React component files so Tailwind can scan them for classes.
    javascript
    // tailwind.config.js
    /** @type {import('tailwindcss').Config} */
    module.exports = {
      content: [
        "./src/**/*.{js,jsx,ts,tsx}",
        "./public/index.html",
      ],
      theme: {
        extend: {},
      },
      plugins: [],
    }
    

6.  *Add Tailwind directives to your CSS*:
    Open your main CSS file (e.g., src/App.css or src/index.css) and add the following at the very top. Any custom CSS should go after these directives.
    css
    /* src/App.css or src/index.css */
    @tailwind base;
    @tailwind components;
    @tailwind utilities;

    /* Your custom styles (e.g., scrollbar, specific animations) */
    /* ... */
    

7.  *Set up frontend environment variables (optional)*:
    Create a file named .env in the frontend directory (next to package.json).
    dotenv
    # frontend/.env
    REACT_APP_API_URL=http://localhost:5000/api
    
    Access it in your React code using process.env.REACT_APP_API_URL.

8.  *Start the React app*:
    ```
    npm start
    ```

    # Or if you use yarn: yarn start
    
    The app will open in your browser at:
    👉 http://localhost:3000

---

## 🔗 Connecting Frontend to Backend

Ensure your React app's API calls are directed to the correct backend address (e.g., http://localhost:5000/api). The REACT_APP_API_URL in your frontend .env is designed for this.

---

## 🖥 API Endpoints

The Flask backend exposes the following endpoints:

* **POST /api/predict**: Predicts house price based on input features.
    * *Request Body*: { "size": 1500, "bedrooms": 3, "bathrooms": 2, "age": 10, "location_score": 7, "model": "random_forest" }
    * *Response*: Predicted price, model metrics, AI insights.
* **GET /api/ai_market_analysis**: Provides general real estate market trends and predictions using AI.
* **POST /api/ai_insights**: Answers general questions about real estate using AI.
    * *Request Body*: { "question": "What is the best time to buy a house?" }
* **POST /api/regression_line**: Returns data points for plotting regression lines for different models.
    * *Request Body*: { "model": "simple" }
* **GET /api/data**: Fetches the raw dataset used for training models.
* **GET /api/models**: Provides comparison metrics for all trained models.
* **GET /api/feature_importance**: Returns feature importances for the Random Forest model.
* **GET /api/price_analysis**: Provides predictions and price-per-sqft analysis across models for a given size.
    * *Request Body*: { "size": 1500 }
* **GET /api/health**: Checks the health status of the backend and AI integration.

---

## 🧪 Testing

You can use tools like [Postman](https://www.postman.com/), [Insomnia](https://insomnia.rest/), or your browser's DevTools to test the API endpoints during development.

---

## 🤝 Contributions

Contributions, issues, and feature requests are welcome! Feel free to fork the repository, create a new branch, and submit pull requests.

---

## 📜 License

This project is licensed under the MIT License. See the LICENSE file (if present) for details.

---

## 📧 Contact

For any questions or support, please feel free to open an issue in this repository.

---

> *Note:* Always ensure CORS (Cross-Origin Resource Sharing) is configured properly in your Flask backend when connecting with a separate React frontend. This project has flask-cors enabled by default.