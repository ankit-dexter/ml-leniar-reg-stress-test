import React, { useEffect, useState } from 'react'
// @ts-expect-error
import { getPredictions } from '../api'

interface PredictionRow {
  year: number
  export_growth: number
  fdi: number
  co2_change: number
  predicted_gdp_growth: number
}

interface PredictionData {
  chart: string
  predictions: PredictionRow[]
}

export default function PredictionChart() {
  const [data, setData] = useState<PredictionData | null>(null)
  const [loading, setLoading] = useState(true)

  useEffect(() => {
    async function fetchData() {
      const result = await getPredictions()
      console.log(result, "result")

      if (result?.success && result.predictions?.length) {
        // eslint-disable-next-line @typescript-eslint/no-explicit-any
        const predictions = result.predictions.map((p: any) => ({
          year: p.year,
          export_growth: p["Average annual rate of growth of exports (scored 1-5)"],
          fdi: p["Inward FDI flows (% of fixed investment)"],
          co2_change: p["CO2 emissions: Oil (% change y/y)"],
          predicted_gdp_growth: p.predicted_gdp_growth,
        }))

        const formattedData: PredictionData = {
          chart: result.chart,
          predictions,
        }

        setData(formattedData)
      }

      setLoading(false)
    }

    if (!data) fetchData()
  }, [])

  if (loading) return <div>Loading prediction data...</div>
  if (!data) return <div>No prediction data available</div>

  return (
    <>
      <div className="white-border" style={{ overflowX: 'auto', marginTop: '2rem' }}>
        <h3>Prediction Data Table (2026–2030)</h3>
        <table style={{ borderCollapse: 'collapse', width: '100%' }}>
          <thead>
            <tr>
              <th style={th}>Year</th>
              <th style={th}>Average annual rate of growth of exports (scored 1-5)</th>
              <th style={th}>Inward FDI flows (% of fixed investment)</th>
              <th style={th}>CO2 emissions: Oil (% change y/y)</th>
              <th style={th}>Predicted Average annual GDP growth (scored 1-5)</th>
            </tr>
          </thead>
          <tbody>
            {data.predictions.map((row, i) => (
              <tr key={i}>
                <td style={td}>{row.year}</td>
                <td style={td}>{row.export_growth.toFixed(2)}</td>
                <td style={td}>{row.fdi.toFixed(2)}</td>
                <td style={td}>{row.co2_change.toFixed(2)}</td>
                <td style={td}>{row.predicted_gdp_growth.toFixed(2)}</td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>

      <div className='white-border' style={{ marginTop: '2rem' }}>
        <h3>GDP Growth Prediction Chart</h3>
        <img
          src={`data:image/png;base64,${data.chart}`}
          alt="Prediction Chart"
          style={{ maxWidth: '100%' }}
        />
      </div>
    </>
  )
}

const th: React.CSSProperties = {
  border: '1px solid black',
  padding: '8px',
  backgroundColor: '#f2f2f2',
  textAlign: 'center',
  color:'black',
}

const td: React.CSSProperties = {
  border: '1px solid #ddd',
  padding: '8px',
  textAlign: 'center',
}
