/* eslint-disable @typescript-eslint/no-explicit-any */
import React, { useState } from 'react'
// eslint-disable-next-line @typescript-eslint/ban-ts-comment
// @ts-expect-error
import { stressTestQuery } from '../api'

export default function StressTestInput() {
  const [query, setQuery] = useState('')
  interface ApiResponse {
    error?: string;
    data?: any; // Replace 'any' with the actual type of data if known
  }
  
  const [response, setResponse] = useState<ApiResponse | null>(null)
  const [loading, setLoading] = useState(false)

  const handleSubmit = async () => {
    if (!query.trim()) return
    setLoading(true)
    setResponse(null)
    try {
      const result = await stressTestQuery(query)
      if (result.success) {
        setResponse(result.data)
      } else {
        setResponse({ error: result.error || 'Failed to get stress test result' })
      }
    } catch (e) {
      setResponse({ error: (e as Error).message })
    }
    setLoading(false)
  }

  return (
    <div>
      <textarea
        rows={3}
        placeholder="Enter stress test scenario, e.g., What if CO2 emissions increase by 10%?"
        value={query}
        onChange={(e) => setQuery(e.target.value)}
      />
          {response?.error ? <b>Error:</b> : <><b>Stress Test Result:</b><br />{JSON.stringify(response, null, 2)}</>}
        {loading ? 'Analyzing...' : 'Run Stress Test'}
      <button onClick={handleSubmit}></button>
      {response && (
        <div style={{ whiteSpace: 'pre-line', marginTop: 10 }}>
          {response.error ? <b>Error:</b> : <><b>Stress Test Result:</b><br />{JSON.stringify(response, null, 2)}</>}
        </div>
      )}
    </div>
  )
}
