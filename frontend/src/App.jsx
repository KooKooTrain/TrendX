import { useState } from "react";
import Chart from "chart.js/auto";
import { Line } from "react-chartjs-2";
import axios from "axios";

function App() {
  const LOOKBACK_DAYS = 30;
  const [prices, setPrices] = useState(generateData(LOOKBACK_DAYS));
  const [prediction, setPrediction] = useState(null);

  async function runPrediction() {
    try {
      const res = await axios.post("http://127.0.0.1:8000/predict", { prices });
      setPrediction(res.data);
      setPrices([...prices, res.data.predicted_price]);
    } catch (err) {
      console.error(err);
    }
  }

  return (
    <div className="p-6 min-h-screen bg-gradient-to-br from-gray-200 to-gray-300 dark:from-gray-800 dark:to-gray-900 text-gray-900 dark:text-gray-100 font-sans">
      <header className="mb-6 flex justify-between items-center">
        <h1 className="text-3xl font-bold text-indigo-500">TrendX</h1>
        <button className="px-4 py-2 bg-indigo-500 text-white rounded" onClick={runPrediction}>
          Run Trend Classifier
        </button>
      </header>

      <div className="mb-6 p-4 bg-white dark:bg-gray-800 rounded shadow">
        <Line
          data={{
            labels: prices.map((_, i) => `Day ${i+1}`),
            datasets: [{
              label: "Closing Price ($)",
              data: prices,
              borderColor: "#6366f1",
              backgroundColor: "rgba(99,102,241,0.2)",
              tension: 0.4,
            }]
          }}
        />
      </div>

      {prediction && (
        <div className="p-4 bg-green-100 dark:bg-green-900/40 rounded shadow">
          <p><strong>Predicted Movement:</strong> {prediction.movement}</p>
          <p><strong>Confidence:</strong> {prediction.confidence}%</p>
          <p><strong>Predicted Price:</strong> ${prediction.predicted_price}</p>
        </div>
      )}
    </div>
  );
}

// helper to generate dummy historical prices
function generateData(days) {
  let arr = [50];
  for (let i = 1; i < days; i++) {
    let change = (Math.random() - 0.45) * 2;
    arr.push(Math.max(10, arr[i - 1] + change));
  }
  return arr;
}

export default App;
