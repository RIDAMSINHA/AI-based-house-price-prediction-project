import React, { useState, useEffect } from 'react';
import { LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer, BarChart, Bar, PieChart, Pie, Cell } from 'recharts';
import { Brain, Home, TrendingUp, MessageSquare, Zap, Target, BarChart3, Lightbulb } from 'lucide-react';

function App() {
  const [size, setSize] = useState(1500);
  const [bedrooms, setBedrooms] = useState(3);
  const [bathrooms, setBathrooms] = useState(2);
  const [age, setAge] = useState(10);
  const [locationScore, setLocationScore] = useState(7);
  const [selectedModel, setSelectedModel] = useState('random_forest');
  const [prediction, setPrediction] = useState(null);
  const [data, setData] = useState([]);
  const [regressionLine, setRegressionLine] = useState(null);
  const [loading, setLoading] = useState(false);
  const [modelComparison, setModelComparison] = useState({});
  const [featureImportance, setFeatureImportance] = useState(null);
  const [priceAnalysis, setPriceAnalysis] = useState(null);
  const [activeTab, setActiveTab] = useState('prediction');
  const [aiInsights, setAiInsights] = useState(null);
  const [marketAnalysis, setMarketAnalysis] = useState(null);
  const [aiQuestion, setAiQuestion] = useState('');
  const [aiAnswer, setAiAnswer] = useState('');
  const [aiLoading, setAiLoading] = useState(false);

  useEffect(() => {
    fetchData();
    fetchModelComparison();
    fetchFeatureImportance();
    fetchMarketAnalysis();
  }, []);

  useEffect(() => {
    if (selectedModel) {
      fetchRegressionLine();
    }
  }, [selectedModel]);

  const API_BASE = 'http://localhost:5000/api';

  const fetchData = async () => {
    try {
      const response = await fetch(`${API_BASE}/data`);
      const result = await response.json();
      setData(result.data);
    } catch (error) {
      console.error('Error fetching data:', error);
    }
  };

  const fetchRegressionLine = async () => {
    try {
      const response = await fetch(`${API_BASE}/regression_line`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ model: selectedModel })
      });
      const result = await response.json();
      setRegressionLine(result);
    } catch (error) {
      console.error('Error fetching regression line:', error);
    }
  };

  const fetchModelComparison = async () => {
    try {
      const response = await fetch(`${API_BASE}/models`);
      const result = await response.json();
      setModelComparison(result.models);
    } catch (error) {
      console.error('Error fetching models:', error);
    }
  };

  const fetchFeatureImportance = async () => {
    try {
      const response = await fetch(`${API_BASE}/feature_importance`);
      const result = await response.json();
      setFeatureImportance(result);
    } catch (error) {
      console.error('Error fetching feature importance:', error);
    }
  };

  const fetchMarketAnalysis = async () => {
    try {
      const response = await fetch(`${API_BASE}/ai_market_analysis`);
      const result = await response.json();
      setMarketAnalysis(result.market_analysis);
    } catch (error) {
      console.error('Error fetching market analysis:', error);
    }
  };

  const handlePredict = async () => {
    setLoading(true);
    try {
      const requestData = {
        size: size,
        model: selectedModel,
        bedrooms: bedrooms,
        bathrooms: bathrooms,
        age: age,
        location_score: locationScore
      };

      const response = await fetch(`${API_BASE}/predict`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(requestData)
      });
      const result = await response.json();
      setPrediction(result);
      setAiInsights(result.ai_insights);
      
      // Get price analysis
      const analysisResponse = await fetch(`${API_BASE}/price_analysis`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ size: size })
      });
      const analysisResult = await analysisResponse.json();
      setPriceAnalysis(analysisResult);
      
    } catch (error) {
      console.error('Error making prediction:', error);
    }
    setLoading(false);
  };

  const handleAiQuestion = async () => {
    if (!aiQuestion.trim()) return;
    
    setAiLoading(true);
    try {
      const response = await fetch(`${API_BASE}/ai_insights`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ question: aiQuestion })
      });
      const result = await response.json();
      setAiAnswer(result.answer);
    } catch (error) {
      console.error('Error getting AI answer:', error);
      setAiAnswer('Sorry, I couldn\'t process your question at the moment. Please try again.');
    }
    setAiLoading(false);
  };

  const formatPrice = (price) => {
    return new Intl.NumberFormat('en-US', {
      style: 'currency',
      currency: 'USD',
      minimumFractionDigits: 0,
      maximumFractionDigits: 0,
    }).format(price);
  };

  const getModelDisplayName = (modelName) => {
    const names = {
      'simple': 'Linear Regression',
      'polynomial': 'Polynomial Regression',
      'multi_linear': 'Multi-Feature Linear',
      'random_forest': 'Random Forest'
    };
    return names[modelName] || modelName;
  };

  const chartData = data.map(d => ({
    size: Math.round(d.size),
    price: Math.round(d.price)
  }));

  const featureImportanceData = featureImportance ? 
    featureImportance.features.map((feature, index) => ({
      name: feature.replace('_', ' ').toUpperCase(),
      value: featureImportance.importance[index] * 100
    })) : [];

  const COLORS = ['#0088FE', '#00C49F', '#FFBB28', '#FF8042', '#8884d8'];

  return (
    <div className="min-h-screen bg-gradient-to-br from-indigo-900 via-purple-900 to-pink-900">
      {/* Header */}
      <div className="bg-black bg-opacity-20 backdrop-blur-md">
        <div className="max-w-7xl mx-auto px-4 py-6">
          <div className="flex items-center justify-between">
            <div className="flex items-center space-x-3">
              <div className="bg-gradient-to-r from-blue-500 to-purple-600 p-3 rounded-xl">
                <Brain className="w-8 h-8 text-white" />
              </div>
              <div>
                <h1 className="text-3xl font-bold text-white">AI Real Estate Predictor</h1>
                <p className="text-indigo-200">Advanced ML & AI-Powered Property Valuation</p>
              </div>
            </div>
            <div className="flex items-center space-x-2 bg-white bg-opacity-10 px-4 py-2 rounded-full">
              <Zap className="w-5 h-5 text-yellow-400" />
              <span className="text-white font-medium">AI Powered</span>
            </div>
          </div>
        </div>
      </div>

      <div className="max-w-7xl mx-auto px-4 py-8">
        {/* Navigation Tabs */}
        <div className="flex flex-wrap justify-center mb-8 space-x-2">
          {[
            { id: 'prediction', label: 'Price Prediction', icon: Target },
            { id: 'analysis', label: 'Model Analysis', icon: BarChart3 },
            { id: 'ai-chat', label: 'AI Assistant', icon: MessageSquare },
            { id: 'market', label: 'Market Insights', icon: TrendingUp }
          ].map(({ id, label, icon: Icon }) => (
            <button
              key={id}
              onClick={() => setActiveTab(id)}
              className={`flex items-center space-x-2 px-6 py-3 rounded-full font-medium transition-all duration-300 ${
                activeTab === id
                  ? 'bg-white text-indigo-900 shadow-lg transform scale-105'
                  : 'bg-white bg-opacity-20 text-white hover:bg-opacity-30'
              }`}
            >
              <Icon className="w-5 h-5" />
              <span>{label}</span>
            </button>
          ))}
        </div>

        {/* Prediction Tab */}
        {activeTab === 'prediction' && (
          <div className="space-y-8">
            {/* Input Controls */}
            <div className="bg-white bg-opacity-10 backdrop-blur-md rounded-3xl p-8 border border-white border-opacity-20">
              <h2 className="text-2xl font-bold text-white mb-6 flex items-center">
                <Home className="w-6 h-6 mr-2" />
                Property Details
              </h2>
              
              <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6">
                <div>
                  <label className="block text-white font-medium mb-2">ML Model</label>
                  <select
                    value={selectedModel}
                    onChange={(e) => setSelectedModel(e.target.value)}
                    className="w-full p-3 rounded-xl bg-white bg-opacity-20 text-white border border-white border-opacity-30 focus:ring-2 focus:ring-blue-500 focus:border-transparent"
                  >
                    <option value="simple">Linear Regression</option>
                    <option value="polynomial">Polynomial Regression</option>
                    <option value="multi_linear">Multi-Feature Linear</option>
                    <option value="random_forest">Random Forest (AI)</option>
                  </select>
                </div>

                <div>
                  <label className="block text-white font-medium mb-2">Size (sq ft)</label>
                  <input
                    type="number"
                    value={size}
                    onChange={(e) => setSize(Number(e.target.value))}
                    className="w-full p-3 rounded-xl bg-white bg-opacity-20 text-white border border-white border-opacity-30 focus:ring-2 focus:ring-blue-500 focus:border-transparent"
                    min="500"
                    max="5000"
                  />
                </div>

                <div>
                  <label className="block text-white font-medium mb-2">Bedrooms</label>
                  <input
                    type="number"
                    value={bedrooms}
                    onChange={(e) => setBedrooms(Number(e.target.value))}
                    className="w-full p-3 rounded-xl bg-white bg-opacity-20 text-white border border-white border-opacity-30 focus:ring-2 focus:ring-blue-500 focus:border-transparent"
                    min="1"
                    max="8"
                  />
                </div>

                <div>
                  <label className="block text-white font-medium mb-2">Bathrooms</label>
                  <input
                    type="number"
                    value={bathrooms}
                    onChange={(e) => setBathrooms(Number(e.target.value))}
                    className="w-full p-3 rounded-xl bg-white bg-opacity-20 text-white border border-white border-opacity-30 focus:ring-2 focus:ring-blue-500 focus:border-transparent"
                    min="1"
                    max="6"
                    step="0.5"
                  />
                </div>

                <div>
                  <label className="block text-white font-medium mb-2">Age (years)</label>
                  <input
                    type="number"
                    value={age}
                    onChange={(e) => setAge(Number(e.target.value))}
                    className="w-full p-3 rounded-xl bg-white bg-opacity-20 text-white border border-white border-opacity-30 focus:ring-2 focus:ring-blue-500 focus:border-transparent"
                    min="0"
                    max="100"
                  />
                </div>

                <div>
                  <label className="block text-white font-medium mb-2">Location Score (1-10)</label>
                  <input
                    type="number"
                    value={locationScore}
                    onChange={(e) => setLocationScore(Number(e.target.value))}
                    className="w-full p-3 rounded-xl bg-white bg-opacity-20 text-white border border-white border-opacity-30 focus:ring-2 focus:ring-blue-500 focus:border-transparent"
                    min="1"
                    max="10"
                  />
                </div>
              </div>

              <div className="mt-8 text-center">
                <button
                  onClick={handlePredict}
                  className="bg-gradient-to-r from-blue-500 to-purple-600 hover:from-blue-600 hover:to-purple-700 text-white font-bold py-4 px-10 rounded-full text-xl shadow-lg transform transition-transform duration-300 hover:scale-105"
                  disabled={loading}
                >
                  {loading ? 'Predicting...' : 'Predict Price'}
                </button>
              </div>
            </div>

            {/* Prediction Result & AI Insights */}
            {prediction && (
              <div className="grid grid-cols-1 lg:grid-cols-2 gap-8">
                <div className="bg-white bg-opacity-10 backdrop-blur-md rounded-3xl p-8 border border-white border-opacity-20 flex flex-col justify-between">
                  <div>
                    <h2 className="text-2xl font-bold text-white mb-6 flex items-center">
                      <Target className="w-6 h-6 mr-2" />
                      Predicted Price
                    </h2>
                    <p className="text-5xl font-extrabold text-blue-300 mb-4 animate-pulse">
                      {formatPrice(prediction.prediction)}
                    </p>
                    <p className="text-white text-lg">
                      Predicted using the <span className="font-bold text-purple-300">{getModelDisplayName(prediction.model_type)}</span> model.
                    </p>
                    <div className="mt-4 text-sm text-indigo-200">
                      <p>R² Score: {prediction.model_metrics.r2_score.toFixed(3)}</p>
                      <p>Mean Absolute Error (MAE): {formatPrice(prediction.model_metrics.mae)}</p>
                    </div>
                  </div>

                  {priceAnalysis && (
                    <div className="mt-6 bg-white bg-opacity-5 rounded-2xl p-6 border border-white border-opacity-10">
                      <h3 className="text-xl font-bold text-white mb-4">Price Analysis</h3>
                      <div className="grid grid-cols-1 sm:grid-cols-2 gap-4 text-white text-sm">
                        <div>
                          <p className="font-semibold text-indigo-300">Market Segment:</p>
                          <p>{priceAnalysis.market_segment}</p>
                        </div>
                        <div>
                          <p className="font-semibold text-indigo-300">Price per sq ft (Random Forest):</p>
                          <p>{formatPrice(priceAnalysis.price_per_sqft.random_forest)}</p>
                        </div>
                        {Object.entries(priceAnalysis.predictions).length > 1 && (
                          <div className="col-span-full">
                            <p className="font-semibold text-indigo-300 mb-2">Predictions by Model:</p>
                            <ul className="list-disc list-inside space-y-1">
                              {Object.entries(priceAnalysis.predictions).map(([modelName, price]) => (
                                <li key={modelName} className="flex items-center">
                                  <span className="w-2 h-2 bg-blue-400 rounded-full mr-2"></span>
                                  {getModelDisplayName(modelName)}: {formatPrice(price)}
                                </li>
                              ))}
                            </ul>
                          </div>
                        )}
                      </div>
                    </div>
                  )}
                </div>

                {aiInsights && (
                  <div className="bg-white bg-opacity-10 backdrop-blur-md rounded-3xl p-8 border border-white border-opacity-20">
                    <h2 className="text-2xl font-bold text-white mb-6 flex items-center">
                      <Lightbulb className="w-6 h-6 mr-2" />
                      AI Insights
                    </h2>
                    <div className="space-y-4 text-white text-lg">
                      <p><span className="font-bold text-blue-300">Market Analysis:</span> {aiInsights.market_analysis}</p>
                      <p><span className="font-bold text-blue-300">Recommendation:</span> {aiInsights.recommendation}</p>
                      <p><span className="font-bold text-blue-300">Key Factors:</span> {aiInsights.key_factors.join(', ')}</p>
                      <p><span className="font-bold text-blue-300">Risks/Opportunities:</span> {aiInsights.risks_opportunities}</p>
                      <p><span className="font-bold text-blue-300">Market Comparison:</span> {aiInsights.market_comparison}</p>
                    </div>
                  </div>
                )}
              </div>
            )}

            {/* Price vs. Size Chart */}
            <div className="bg-white bg-opacity-10 backdrop-blur-md rounded-3xl p-8 border border-white border-opacity-20">
              <h2 className="text-2xl font-bold text-white mb-6 flex items-center">
                <TrendingUp className="w-6 h-6 mr-2" />
                Price vs. Size Scatter Plot with Regression Line
              </h2>
              <ResponsiveContainer width="100%" height={400}>
                <LineChart
                  data={chartData}
                  margin={{ top: 5, right: 30, left: 20, bottom: 5 }}
                >
                  <CartesianGrid strokeDasharray="3 3" strokeOpacity={0.3} />
                  <XAxis dataKey="size" stroke="#9CA3AF" label={{ value: "Size (sq ft)", position: "insideBottom", offset: -5, fill: "#E0E7FF" }} />
                  <YAxis stroke="#9CA3AF" label={{ value: "Price (USD)", angle: -90, position: "insideLeft", fill: "#E0E7FF" }} />
                  <Tooltip
                    formatter={(value) => formatPrice(value)}
                    labelFormatter={(label) => `Size: ${label} sq ft`}
                    contentStyle={{ backgroundColor: 'rgba(0,0,0,0.7)', border: 'none', borderRadius: '8px' }}
                    itemStyle={{ color: 'white' }}
                    labelStyle={{ color: '#A78BFA' }}
                  />
                  <Line
                    type="monotone"
                    dataKey="price"
                    stroke="#8884d8"
                    strokeWidth={2}
                    dot={{ stroke: '#8884d8', strokeWidth: 1, r: 3 }}
                    activeDot={{ r: 5 }}
                    name="Actual Price"
                  />
                  {regressionLine && (
                    <Line
                      type="monotone"
                      dataKey="prediction"
                      stroke="#82ca9d"
                      strokeWidth={3}
                      dot={false}
                      activeDot={false}
                      name={`${getModelDisplayName(selectedModel)} Regression`}
                      data={regressionLine.size_range.map((s, i) => ({
                        size: Math.round(s),
                        prediction: Math.round(regressionLine.predictions[i])
                      }))}
                    />
                  )}
                </LineChart>
              </ResponsiveContainer>
            </div>
          </div>
        )}

        {/* Model Analysis Tab */}
        {activeTab === 'analysis' && (
          <div className="space-y-8">
            <div className="bg-white bg-opacity-10 backdrop-blur-md rounded-3xl p-8 border border-white border-opacity-20">
              <h2 className="text-2xl font-bold text-white mb-6 flex items-center">
                <BarChart3 className="w-6 h-6 mr-2" />
                Model Performance Comparison
              </h2>
              {Object.keys(modelComparison).length > 0 ? (
                <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-6 text-white text-center">
                  {Object.entries(modelComparison).map(([modelName, metrics]) => (
                    <div key={modelName} className="bg-white bg-opacity-5 rounded-2xl p-6 border border-white border-opacity-10 shadow-lg">
                      <h3 className="text-xl font-semibold text-purple-300 mb-3">{getModelDisplayName(modelName)}</h3>
                      <p className="text-lg mb-1"><span className="font-medium text-blue-200">R² Score:</span> {metrics.r2_score.toFixed(3)}</p>
                      <p className="text-lg mb-1"><span className="font-medium text-blue-200">MSE:</span> {metrics.mse.toFixed(0)}</p>
                      <p className="text-lg"><span className="font-medium text-blue-200">MAE:</span> {metrics.mae.toFixed(0)}</p>
                    </div>
                  ))}
                </div>
              ) : (
                <p className="text-white text-lg">Loading model comparison data...</p>
              )}
            </div>

            {featureImportance && (
              <div className="bg-white bg-opacity-10 backdrop-blur-md rounded-3xl p-8 border border-white border-opacity-20">
                <h2 className="text-2xl font-bold text-white mb-6 flex items-center">
                  <BarChart3 className="w-6 h-6 mr-2" />
                  Feature Importance (Random Forest)
                </h2>
                <ResponsiveContainer width="100%" height={300}>
                  <BarChart
                    data={featureImportanceData}
                    layout="vertical"
                    margin={{ top: 20, right: 30, left: 60, bottom: 5 }}
                  >
                    <CartesianGrid strokeDasharray="3 3" strokeOpacity={0.3} horizontal={false} />
                    <XAxis type="number" stroke="#9CA3AF" label={{ value: "Importance (%)", position: "insideBottom", offset: -5, fill: "#E0E7FF" }} />
                    <YAxis dataKey="name" type="category" stroke="#9CA3AF" interval={0} tick={{ fill: "#E0E7FF" }} />
                    <Tooltip
                      formatter={(value) => `${value.toFixed(1)}%`}
                      contentStyle={{ backgroundColor: 'rgba(0,0,0,0.7)', border: 'none', borderRadius: '8px' }}
                      itemStyle={{ color: 'white' }}
                      labelStyle={{ color: '#A78BFA' }}
                    />
                    <Bar dataKey="value" fill="#82ca9d" barSize={30} radius={[0, 10, 10, 0]} />
                  </BarChart>
                </ResponsiveContainer>
              </div>
            )}
          </div>
        )}

        {/* AI Assistant Tab */}
        {activeTab === 'ai-chat' && (
          <div className="bg-white bg-opacity-10 backdrop-blur-md rounded-3xl p-8 border border-white border-opacity-20">
            <h2 className="text-2xl font-bold text-white mb-6 flex items-center">
              <MessageSquare className="w-6 h-6 mr-2" />
              Ask the AI Assistant
            </h2>
            <div className="mb-6">
              <textarea
                className="w-full p-4 rounded-xl bg-white bg-opacity-20 text-white border border-white border-opacity-30 focus:ring-2 focus:ring-blue-500 focus:border-transparent"
                rows="4"
                placeholder="e.g., What factors influence property value the most? What's the outlook for real estate in the next year?"
                value={aiQuestion}
                onChange={(e) => setAiQuestion(e.target.value)}
              ></textarea>
              <button
                onClick={handleAiQuestion}
                className="mt-4 bg-gradient-to-r from-teal-500 to-green-600 hover:from-teal-600 hover:to-green-700 text-white font-bold py-3 px-8 rounded-full shadow-lg transition-transform duration-300 hover:scale-105"
                disabled={aiLoading || !aiQuestion.trim()}
              >
                {aiLoading ? 'Thinking...' : 'Get AI Answer'}
              </button>
            </div>
            {aiAnswer && (
              <div className="bg-white bg-opacity-5 rounded-2xl p-6 border border-white border-opacity-10">
                <h3 className="text-xl font-bold text-white mb-3">AI's Response:</h3>
                <p className="text-white text-lg whitespace-pre-wrap">{aiAnswer}</p>
              </div>
            )}
          </div>
        )}

        {/* Market Insights Tab */}
        {activeTab === 'market' && (
          <div className="bg-white bg-opacity-10 backdrop-blur-md rounded-3xl p-8 border border-white border-opacity-20">
            <h2 className="text-2xl font-bold text-white mb-6 flex items-center">
              <TrendingUp className="w-6 h-6 mr-2" />
              General Market Insights (AI Generated)
            </h2>
            {marketAnalysis ? (
              <div className="space-y-6 text-white text-lg">
                <div>
                  <p className="font-bold text-blue-300">Current Market Trends:</p>
                  <ul className="list-disc list-inside ml-4 space-y-1">
                    {marketAnalysis.trends.map((trend, index) => (
                      <li key={index}>{trend}</li>
                    ))}
                  </ul>
                </div>
                <div>
                  <p className="font-bold text-blue-300">Price Predictions (Next 6 Months):</p>
                  <p>{marketAnalysis.predictions}</p>
                </div>
                <div>
                  <p className="font-bold text-blue-300">Best Investment Strategies:</p>
                  <p>{marketAnalysis.strategies}</p>
                </div>
                <div>
                  <p className="font-bold text-blue-300">Factors Driving Market Changes:</p>
                  <p>{marketAnalysis.driving_factors}</p>
                </div>
              </div>
            ) : (
              <p className="text-white text-lg">Loading market insights...</p>
            )}
          </div>
        )}
      </div>
    </div>
  );
}

export default App;