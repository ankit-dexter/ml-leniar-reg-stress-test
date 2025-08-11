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
  target_year?: number | string;
  scenario_source?: string;
  changes_detected: Record<string, number>;
  base_scenario: Scenario;
  stressed_scenario: Scenario;
  impact: ImpactSummary;
}

export default function StressTestInput() {
  const [query, setQuery] = useState<string>('')
  const [targetYear, setTargetYear] = useState<string>('')
  const [useSpecificYear, setUseSpecificYear] = useState<boolean>(false)
  const [response, setResponse] = useState<StressTestResponse | null>(null)
  const [error, setError] = useState(null)
  const [loading, setLoading] = useState(false)

  // Generate year options
  const currentYear = new Date().getFullYear()
  const yearOptions = Array.from({length: 5}, (_, i) => currentYear + 1 + i)

  const handleSubmit = async () => {
    if (!query.trim()) return
    
    if (useSpecificYear && !targetYear) {
      setError(new Error('Please select a target year'))
      return
    }

    setLoading(true)
    setError(null)
    setResponse(null)
    
    try {
      const requestData = useSpecificYear ? 
        { query, target_year: parseInt(targetYear) } : 
        { query }
      
      const result = await stressTestQuery(requestData)
      if (result.success) {
        // *** FIX: Extract the stress_test_result from the response ***
        setResponse(result.stress_test_result)
      } else {
        setError(result.error || 'Failed to get stress test result')
      }
    } catch (e) {
      console.log(e)
      setError(new Error((e as Error).message))
    }
    setLoading(false)
  }

  return (
    <div style={{ padding: '20px', margin: '0 auto' }}>
      <h2 style={{ marginBottom: '25px' }}>🧪 GDP Stress Test Analysis</h2>

      {/* Query Input */}
      <div style={{ marginBottom: '20px' }}>
        <label style={{ display: 'block', marginBottom: '8px', fontWeight: 'bold' }}>
          Stress Test Query
        </label>
        <textarea
          rows={3}
          value={query}
          onChange={(e) => setQuery(e.target.value)}
          placeholder="e.g., What if export growth increases by 1 point? or What if CO2 emissions change by 2%?"
          style={{ 
            width: '100%', 
            padding: '12px', 
            fontSize: '14px',
            border: '2px solid #e9ecef',
            borderRadius: '8px',
            fontFamily: 'inherit',
            resize: 'vertical'
          }}
        />
      </div>

      {/* Year Selection */}
      <div style={{ 
        marginBottom: '20px', 
        padding: '18px', 
        border: '2px solid #e9ecef', 
        borderRadius: '8px',
        backgroundColor: '#f8f9fa'
      }}>
        <div style={{ marginBottom: '12px' }}>
          <label style={{ display: 'flex', alignItems: 'center', cursor: 'pointer', color: '#495057' }}>
            <input
              type="checkbox"
              checked={useSpecificYear}
              onChange={(e) => setUseSpecificYear(e.target.checked)}
              style={{ marginRight: '10px', transform: 'scale(1.2)' }}
            />
            <strong>🎯 Test against specific year's predictions</strong>
          </label>
        </div>

        {useSpecificYear && (
          <div style={{ marginLeft: '25px', marginTop: '15px' }}>
            <label style={{ marginRight: '12px', fontWeight: 'bold', color: 'black' }}>
              Target Year:
            </label>
            <select
              value={targetYear}
              onChange={(e) => setTargetYear(e.target.value)}
              style={{ 
                padding: '8px 12px', 
                fontSize: '14px',
                border: '1px solid #ced4da',
                borderRadius: '6px',
                backgroundColor: 'white',
                color: 'black',
              }}
            >
              <option value="">Select Year</option>
              {yearOptions.map(year => (
                <option key={year} value={year}>{year}</option>
              ))}
            </select>
            <div style={{ 
              fontSize: '13px', 
              color: '#6c757d', 
              marginTop: '8px',
              fontStyle: 'italic'
            }}>
              💡 Uses {targetYear || 'selected year'}'s predicted values as baseline
            </div>
          </div>
        )}

        {!useSpecificYear && (
          <div style={{ 
            fontSize: '13px', 
            color: '#6c757d', 
            fontStyle: 'italic',
            marginLeft: '25px'
          }}>
            📊 Uses recent historical averages as baseline
          </div>
        )}
      </div>

      {/* Submit Button */}
      <button 
        onClick={handleSubmit} 
        disabled={loading || !query.trim() || (useSpecificYear && !targetYear)}
        style={{ 
          padding: '12px 24px',
          fontSize: '16px',
          fontWeight: 'bold',
          border: 'none',
          borderRadius: '8px',
          backgroundColor: (loading || !query.trim() || (useSpecificYear && !targetYear)) ? '#6c757d' : '#007bff',
          color: 'white',
          cursor: (loading || !query.trim() || (useSpecificYear && !targetYear)) ? 'not-allowed' : 'pointer',
          marginBottom: '25px',
          transition: 'background-color 0.3s'
        }}
      >
        {loading ? '⏳ Analyzing...' : '🚀 Run Stress Test'}
      </button>

      {/* Error Display */}
      {error && (
        <div style={{ 
          padding: '15px', 
          backgroundColor: '#f8d7da', 
          border: '1px solid #f5c6cb',
          borderRadius: '8px',
          color: '#721c24',
          marginBottom: '20px'
        }}>
          ❌ {error}
        </div>
      )}

      {/* Results Display */}
      {response && !error && (
        <div style={{ 
          border: '2px solid #dee2e6', 
          borderRadius: '12px', 
          overflow: 'hidden',
          backgroundColor: 'white',
          boxShadow: '0 4px 6px rgba(0, 0, 0, 0.1)'
        }}>
          
          {/* Header Section */}
          <div style={{ 
            padding: '20px', 
            backgroundColor: '#f8f9fa',
            borderBottom: '1px solid #dee2e6'
          }}>
            <div style={{ fontSize: '16px', marginBottom: '8px' }}>
              <strong>🔍 Query:</strong> <span style={{ color: '#495057' }}>{response.query}</span>
              {response.target_year && response.target_year !== "Current/Recent" && (
                <span style={{ 
                  marginLeft: '15px', 
                  backgroundColor: '#007bff',
                  color: 'white',
                  padding: '4px 12px',
                  borderRadius: '20px',
                  fontSize: '14px',
                  fontWeight: 'bold'
                }}>
                  📅 Testing for {response.target_year}
                </span>
              )}
            </div>

            {response.scenario_source && (
              <div style={{ 
                fontSize: '14px', 
                color: '#6c757d',
                fontStyle: 'italic'
              }}>
                📊 Base Scenario: {response.scenario_source}
              </div>
            )}
          </div>

          {/* Changes Detected */}
          <div style={{ padding: '20px', borderBottom: '1px solid #dee2e6' }}>
            <h4 style={{ margin: '0 0 15px 0', color: '#495057' }}>🔄 Changes Detected</h4>
            <div style={{ display: 'flex', gap: '15px', flexWrap: 'wrap' }}>
              {Object.entries(response.changes_detected).map(([key, value]) => (
                <div key={key} style={{
                  padding: '10px 15px',
                  backgroundColor: value > 0 ? '#d4edda' : '#f8d7da',
                  border: `1px solid ${value > 0 ? '#c3e6cb' : '#f5c6cb'}`,
                  borderRadius: '25px',
                  display: 'flex',
                  alignItems: 'center',
                  gap: '8px'
                }}>
                  <strong style={{ textTransform: 'capitalize', color: 'gray' }}>
                    {key.replace('_', ' ')}:
                  </strong>
                  <span style={{ 
                    color: value > 0 ? '#155724' : '#721c24',
                    fontWeight: 'bold',
                    fontSize: '16px'
                  }}>
                    {value > 0 ? '+' : ''}{value}
                  </span>
                </div>
              ))}
            </div>
          </div>

          {/* Base vs Stressed Scenarios */}
          <div style={{ display: 'flex', minHeight: '250px' }}>
            {/* Base Scenario */}
            <div style={{ 
              flex: 1, 
              padding: '25px', 
              backgroundColor: '#e3f2fd',
              borderRight: '1px solid #dee2e6'
            }}>
              <h4 style={{ margin: '0 0 20px 0', display: 'flex', alignItems: 'center', gap: '8px' , color: 'gray'}}>
                📘 Base Scenario
              </h4>
              <div style={{ gap: '15px', color: '#1976d2' }}>
                <div style={{ marginBottom: '12px', display: 'flex', justifyContent: 'space-between' }}>
                  <strong>Average annual rate of growth of exports (scored 1-5)</strong>
                  <span style={{ color: '#1976d2', fontWeight: 'bold' }}>{response.base_scenario.export_growth}</span>
                </div>
                <div style={{ marginBottom: '12px', display: 'flex', justifyContent: 'space-between' }}>
                  <strong>Inward FDI flows (% of fixed investment)</strong>
                  <span style={{ color: '#1976d2', fontWeight: 'bold' }}>{response.base_scenario.fdi_flows}%</span>
                </div>
                <div style={{ marginBottom: '12px', display: 'flex', justifyContent: 'space-between' }}>
                  <strong>CO2 emissions: Oil (% change y/y)</strong>
                  <span style={{ color: '#1976d2', fontWeight: 'bold' }}>{response.base_scenario.co2_change}%</span>
                </div>
                <div style={{ 
                  borderTop: '2px solid #90caf9', 
                  paddingTop: '15px', 
                  marginTop: '15px',
                  textAlign: 'center'
                }}>
                  <div style={{ fontSize: '14px', color: '#666', marginBottom: '5px' }}>Average annual GDP growth (scored 1-5) Prediction</div>
                  <div style={{ fontSize: '28px', fontWeight: 'bold', color: '#1976d2' }}>
                    {response.base_scenario.gdp_prediction}
                  </div>
                </div>
              </div>
            </div>

            {/* Stressed Scenario */}
            <div style={{ 
              flex: 1, 
              padding: '25px', 
              backgroundColor: '#fff3e0'
            }}>
              <h4 style={{ margin: '0 0 20px 0', color: '#ef6c00', display: 'flex', alignItems: 'center', gap: '8px' }}>
                📙 Stressed Scenario
              </h4>
              <div style={{ gap: '15px', color: '#f57c00' }}>
                <div style={{ marginBottom: '12px', display: 'flex', justifyContent: 'space-between' }}>
                  <strong>Average annual rate of growth of exports (scored 1-5)</strong>
                  <span style={{ color: '#f57c00', fontWeight: 'bold' }}>{response.stressed_scenario.export_growth}</span>
                </div>
                <div style={{ marginBottom: '12px', display: 'flex', justifyContent: 'space-between' }}>
                  <strong>Inward FDI flows (% of fixed investment)</strong>
                  <span style={{ color: '#f57c00', fontWeight: 'bold' }}>{response.stressed_scenario.fdi_flows}%</span>
                </div>
                <div style={{ marginBottom: '12px', display: 'flex', justifyContent: 'space-between' }}>
                  <strong>CO2 emissions: Oil (% change y/y)</strong>
                  <span style={{ color: '#f57c00', fontWeight: 'bold' }}>{response.stressed_scenario.co2_change}%</span>
                </div>
                <div style={{ 
                  borderTop: '2px solid #ffb74d', 
                  paddingTop: '15px', 
                  marginTop: '15px',
                  textAlign: 'center'
                }}>
                  <div style={{ fontSize: '14px', color: '#666', marginBottom: '5px' }}>Average annual GDP growth (scored 1-5) Prediction</div>
                  <div style={{ fontSize: '28px', fontWeight: 'bold', color: '#f57c00' }}>
                    {response.stressed_scenario.gdp_prediction}
                  </div>
                </div>
              </div>
            </div>
          </div>

          {/* Impact Summary */}
          <div style={{ 
            padding: '25px', 
            backgroundColor: '#f8f9fa',
            borderTop: '1px solid #dee2e6'
          }}>
            <h4 style={{ margin: '0 0 20px 0', color: '#495057', textAlign: 'center' }}>
              📈 Impact Analysis
            </h4>
            
            <div style={{ display: 'flex', gap: '30px', justifyContent: 'center', alignItems: 'center', flexWrap: 'wrap' }}>
              {/* GDP Change */}
              <div style={{ textAlign: 'center', minWidth: '150px' }}>
                <div style={{ 
                  fontSize: '36px', 
                  fontWeight: 'bold',
                  color: response.impact.gdp_change > 0 ? '#28a745' : '#dc3545',
                  marginBottom: '8px'
                }}>
                  {response.impact.gdp_change > 0 ? '+' : ''}{response.impact.gdp_change}
                </div>
                <div style={{ fontSize: '14px', color: '#6c757d', fontWeight: 'bold' }}>
                  GDP Points Change
                </div>
              </div>
              
              {/* Percentage Change */}
              <div style={{ textAlign: 'center', minWidth: '150px' }}>
                <div style={{ 
                  fontSize: '36px', 
                  fontWeight: 'bold',
                  color: response.impact.percentage_change > 0 ? '#28a745' : '#dc3545',
                  marginBottom: '8px'
                }}>
                  {response.impact.percentage_change > 0 ? '+' : ''}{response.impact.percentage_change}%
                </div>
                <div style={{ fontSize: '14px', color: '#6c757d', fontWeight: 'bold' }}>
                  Percentage Change
                </div>
              </div>

              {/* Impact Description */}
              <div style={{ 
                textAlign: 'center', 
                flex: 1, 
                minWidth: '200px',
                padding: '15px',
                backgroundColor: 'white',
                borderRadius: '8px',
                border: '1px solid #dee2e6'
              }}>
                <div style={{ 
                  fontSize: '18px', 
                  fontWeight: 'bold', 
                  color: '#495057',
                  lineHeight: '1.4'
                }}>
                  {response.impact.impact_description}
                </div>
              </div>
            </div>
          </div>
        </div>
      )}

      {/* Help Section */}
      <div style={{ 
        marginTop: '30px', 
        padding: '20px', 
        backgroundColor: '#f8f9fa', 
        borderRadius: '8px',
        border: '1px solid #e9ecef'
      }}>
        <div style={{ marginBottom: '15px' }}>
          <strong style={{ color: '#495057', fontSize: '16px' }}>💡 How to Use Stress Testing</strong>
        </div>
        <div style={{ fontSize: '14px', color: '#6c757d', lineHeight: '1.6' }}>
          <div style={{ marginBottom: '8px' }}>
            <strong>📊 Historical Base:</strong> Compare against recent average economic conditions (last 5 years)
          </div>
          <div style={{ marginBottom: '8px' }}>
            <strong>🎯 Year-Specific Base:</strong> Compare against a specific future year's predicted values (e.g., 2026)
          </div>
          <div style={{ marginBottom: '8px' }}>
            <strong>📝 Example Queries:</strong>
          </div>
          <div style={{ marginLeft: '20px', fontSize: '13px' }}>
            • "What if CO2 emissions increase by 5%?"<br/>
            • "What if export growth drops by 2 points?"<br/>
            • "What if FDI decreases by 10%?"
          </div>
        </div>
      </div>
    </div>
  )
}
