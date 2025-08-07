
import React, { useState } from 'react'
import CsvUploader from './Component/Csvuploader'
import PredictionChart from './Component/Prediction'
import StressTestInput from './Component/Stressinput'
import AIInsights from './Component/AIInsights'
import './App.css'

function App() {
  const [modelReady, setModelReady] = useState(false)
  return (
    <>
      <div style={{ maxWidth: 900, margin: 'auto', padding: 20 }}>
      <h1>Southeast Asia GDP Growth Predictor</h1>
      <CsvUploader onUploadSuccess={(isReady: boolean) => setModelReady(isReady)} />
      {modelReady && (
        <>
          <h2>Predictions for Upcoming Years</h2>
          <PredictionChart />
          <h2>Stress Testing</h2>
          <StressTestInput />
          <h2>AI Economic Insights</h2>
          <AIInsights />
        </>
      )}
    </div>
    </>
  )
}

export default App
