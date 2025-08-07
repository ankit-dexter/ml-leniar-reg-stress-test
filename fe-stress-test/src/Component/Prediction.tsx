import React, { useEffect, useState } from 'react'
// eslint-disable-next-line @typescript-eslint/ban-ts-comment
// @ts-expect-error
import { getPredictions } from '../api'
import { Line } from 'react-chartjs-2'
import 'chart.js/auto'

export default function PredictionChart() {
  interface PredictionData {
    chart: any;
    years: number[];
    actual: number[];
    predicted: number[];
  }

  const [data, setData] = useState<PredictionData | null>(null)
  const [loading, setLoading] = useState(true)

  useEffect(() => {
    async function fetchData() {
      const result = await getPredictions()
      console.log(result, "result")
      if (result.chart) {
        setData(result) // expected format: { years: [], actual: [], predicted: [] }
      }
      setLoading(false)
    }
    if (!data)
      fetchData()
  }, [])

  if (loading) return <div>Loading prediction chart...</div>
  if (!data) return <div>No prediction data available</div>

  const chartData = {
    labels: data.years,
    datasets: [
      {
        label: 'Actual GDP Growth',
        data: data.actual,
        borderColor: 'blue',
        fill: false,
      },
      {
        label: 'Predicted GDP Growth',
        data: data.predicted,
        borderColor: 'green',
        borderDash: [5, 5],
        fill: false,
      },
    ],
  }

  return <>
    {data && <div className='white-border'> <img
      src={`data:image/png;base64,${data.chart}`}
      alt="Plot"
      style={{ maxWidth: '100%' }}
    /></div>}</>
}
