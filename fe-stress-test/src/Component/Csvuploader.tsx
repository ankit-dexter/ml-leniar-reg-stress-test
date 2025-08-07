import React, { useState } from 'react'
// eslint-disable-next-line @typescript-eslint/ban-ts-comment
// @ts-expect-error
import { uploadCsv } from '../api'

interface CsvUploaderProps {
  onUploadSuccess: () => void;
}

export default function CsvUploader({ onUploadSuccess }: CsvUploaderProps) {
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
      const result = await uploadCsv(file)
      if (result.success) {
        setMessage('File uploaded and model trained successfully!')
        onUploadSuccess()
      } else {
        setMessage(`Upload failed: ${result.error || 'Unknown error'}`)
      }
    } catch (e) {
      setMessage('Upload error: ' + (e as Error).message)
    }
    setLoading(false)
  }

  return (
    <div>
      <input type="file" accept=".csv" onChange={handleFileChange} />
      <button onClick={handleUpload} disabled={loading}>
        {loading ? 'Uploading...' : 'Upload CSV & Train Model'}
      </button>
      {message && <p>{message}</p>}
    </div>
  )
}
