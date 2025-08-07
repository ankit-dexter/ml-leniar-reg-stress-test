import React, { useState } from 'react'
// eslint-disable-next-line @typescript-eslint/ban-ts-comment
// @ts-expect-error
import { uploadCsv } from '../api'

interface CsvUploaderProps {
  onUploadSuccess: (isReady: boolean) => void;
}

export default function CsvUploader({ onUploadSuccess }: CsvUploaderProps) {
  interface TrainingResult {
    training_result: unknown;
    performance: {
      train_r2: number;
      test_r2: number;
      train_rmse: number;
      test_rmse: number;
      year_range: string;
    };
    coefficients: {
      export_growth: number;
      fdi_flows: number;
      co2_change: number;
      intercept: number;
    };
    data_summary: {
      total_records: number;
      year_range: string;
      avg_gdp_growth: number;
      avg_export_growth: number;
      avg_fdi_flows: number;
      avg_co2_change: number;
    };
  }

  const [response, setResponse] = useState<TrainingResult | null>(null)
  const [file, setFile] = useState(null)
  const [loading, setLoading] = useState(false)
  const [message, setMessage] = useState('')

  const handleFileChange = (e) => {
    setFile(e.target.files[0])
  }

  const handleUpload = async () => {
    if (!file) {
      setMessage('Please select a CSV file first')
      return
    }
    setLoading(true)
    setMessage('')
    try {

      onUploadSuccess(false)
      const result = await uploadCsv(file)
      if (result.success) {
        setResponse(result)
        setMessage(result.training_result.message)
        onUploadSuccess(true)
      } else {
        setMessage(`Upload failed: ${result.error || 'Unknown error'}`)
      }
    } catch (e) {
      setMessage('Upload error: ' + (e as Error).message)
    }
    setLoading(false)
  }

  const perf = response ? (response.training_result as { performance: { train_r2: number; test_r2: number; train_rmse: number; test_rmse: number; year_range: string; } }).performance : null
  const coef = response ? (response.training_result as { coefficients: { export_growth: number; fdi_flows: number; co2_change: number; intercept: number; } }).coefficients : null
  const summary = response ? response.data_summary : null

  return (
    <div className='white-border'>
      <input type="file" accept=".csv" onChange={handleFileChange} />
      <button onClick={handleUpload} disabled={loading}>
        {loading ? 'Uploading...' : 'Upload CSV & Train Model'}
      </button>
      {message && <p>{message}</p>}
      <div>
        {response && <>
          <div style={{ padding: '20px', maxWidth: '900px', margin: '0 auto' }}>
            <h2 style={{ fontSize: '24px', fontWeight: 'bold' }}>📈 GDP Growth Forecast Overview</h2>
            <div className='trainingData'>
              <section style={{ marginBottom: '24px' }}>
                <h3 style={{ fontSize: '18px', fontWeight: 'bold' }}>📊 Model Performance</h3>
                <div><strong>Train R²:</strong> {perf ? perf.train_r2.toFixed(3) : 'N/A'}</div>
                <div><strong>Test R²:</strong> {perf ? perf.test_r2.toFixed(3) : 'N/A'}</div>
                <div><strong>Train RMSE:</strong> {perf ? perf.train_rmse.toFixed(3) : 'N/A'}</div>
                <div><strong>Test RMSE:</strong> {perf ? perf.test_rmse.toFixed(3) : 'N/A'}</div>
                <div><strong>Year Range Used:</strong> {perf ? perf.year_range : 'N/A'}</div>
              </section>

              <section style={{ marginBottom: '24px' }}>
                <h3 style={{ fontSize: '18px', fontWeight: 'bold' }}>📉 Regression Coefficients</h3>
                <div><strong>Export Growth:</strong> {coef ? coef.export_growth.toFixed(4) : 'N/A'}</div>
                <div><strong>FDI Flows:</strong> {coef ? coef.fdi_flows.toFixed(4) : 'N/A'}</div>
                <div><strong>CO₂ Change:</strong> {coef ? coef.co2_change.toFixed(4) : 'N/A'}</div>
                <div><strong>Intercept:</strong> {coef ? coef.intercept.toFixed(4) : 'N/A'}</div>
              </section>

              <section>
                <h3 style={{ fontSize: '18px', fontWeight: 'bold' }}>📌 Dataset Summary</h3>

                <div><strong>Total Records:</strong> {summary ? summary.total_records : 'N/A'}</div>
                <div><strong>Year Range:</strong> {summary ? summary.year_range : 'N/A'}</div>
                <div><strong>Avg GDP Growth:</strong> {summary ? summary.avg_gdp_growth : 'N/A'}</div>
                <div><strong>Avg Export Growth:</strong> {summary ? summary.avg_export_growth : 'N/A'}</div>
                <div><strong>Avg FDI Flows:</strong> {summary ? summary.avg_fdi_flows : 'N/A'}</div>
                <div><strong>Avg CO₂ Change:</strong> {summary ? summary.avg_co2_change : 'N/A'}</div>

              </section>
            </div>
          </div>
        </>}
      </div>
    </div>
  )
}
