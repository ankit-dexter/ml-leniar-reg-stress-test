import React, { useState } from 'react'
// @ts-expect-error
import { stressTestQuery } from '../api'

interface Scenario {
  export_growth: number;
  fdi_flows: number;
  co2_change: number;
  gdp_prediction: number;
}

interface ImpactSummary {
  gdp_change: number;
  percentage_change: number;
  impact_description: string;
}

interface StressTestResponse {
  success: boolean;
  query: string;
  changes_detected: Record<string, number>;
  base_scenario: Scenario;
  stressed_scenario: Scenario;
  impact: ImpactSummary;
}

export default function StressTestInput() {
  const [query, setQuery] = useState<string>('')
  const [response, setResponse] = useState<StressTestResponse | null>(null)
  const [error, setError] = useState(null)
  const [loading, setLoading] = useState(false)

  const handleSubmit = async () => {
    if (!query.trim()) return
    setLoading(true)
    setError(null)
    setResponse(null)
    try {
      const result = await stressTestQuery(query)
      if (result.success) {
        setResponse(result)
      } else {
        setError(result.detail || 'Failed to get stress test result')
      }
    } catch (e) {
      console.log(e)
      setError(new Error((e as Error).message))
    }
    setLoading(false)
  }

  return (
    <div style={{ padding: '20px', maxWidth: '900px', margin: '0 auto' }}>
      <h2>🔍 Stress Test Scenario</h2>

      <textarea
        rows={3}
        value={query}
        onChange={(e) => setQuery(e.target.value)}
        placeholder="e.g., What if export growth increases by 1 point?"
        style={{ width: '100%', marginBottom: '10px', fontSize: '14px' }}
      />

      <button onClick={handleSubmit} disabled={loading} style={{ marginBottom: '20px' }}>
        {loading ? 'Analyzing...' : 'Run Stress Test'}
      </button>

      {error && <p style={{ color: 'red' }}>❌ {error}</p>}

      {response && !error && (
        <div style={{ border: '1px solid #ddd', padding: '20px', borderRadius: '6px' }}>
          <div style={{ marginBottom: '10px' }}>
            <strong>Query:</strong> {response.query}
          </div>

          <h4>🔄 Changes Detected</h4>
          <div>
            {Object.entries(response.changes_detected).map(([key, value]) => (
              <div key={key}>
                <strong>{key}:</strong> +{value}
              </div>
            ))}
          </div>

          <div style={{ display: 'flex', gap: '40px', marginTop: '20px' }}>
            <div style={{ flex: 1 }}>
              <h4>📘 Base Scenario</h4>
              <div><strong>Export Growth:</strong> {response.base_scenario.export_growth}</div>
              <div><strong>FDI Flows:</strong> {response.base_scenario.fdi_flows}</div>
              <div><strong>CO₂ Change:</strong> {response.base_scenario.co2_change}</div>
              <div><strong>GDP Prediction:</strong> {response.base_scenario.gdp_prediction}</div>
            </div>

            <div style={{ flex: 1 }}>
              <h4>📙 Stressed Scenario</h4>
              <div><strong>Export Growth:</strong> {response.stressed_scenario.export_growth}</div>
              <div><strong>FDI Flows:</strong> {response.stressed_scenario.fdi_flows}</div>
              <div><strong>CO₂ Change:</strong> {response.stressed_scenario.co2_change}</div>
              <div><strong>GDP Prediction:</strong> {response.stressed_scenario.gdp_prediction}</div>
            </div>
          </div>

          <div style={{ marginTop: '20px' }}>
            <h4>📈 Impact Summary</h4>
            <div><strong>GDP Change:</strong> {response.impact.gdp_change}</div>
            <div><strong>Percentage Change:</strong> {response.impact.percentage_change}%</div>
            <div><strong>Description:</strong> {response.impact.impact_description}</div>
          </div>
        </div>
      )}
    </div>
  )
}
