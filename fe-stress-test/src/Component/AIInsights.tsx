import React, { useState } from 'react'
// eslint-disable-next-line @typescript-eslint/ban-ts-comment
// @ts-expect-error
import { getAIInsights } from '../api'

export default function AIInsights() {
  const [question, setQuestion] = useState('')
  const [answer, setAnswer] = useState('')
  const [loading, setLoading] = useState(false)

  const askQuestion = async () => {
    if (!question.trim()) return
    setLoading(true)
    setAnswer('')
    try {
      const result = await getAIInsights(question)
      if (result.success) {
        setAnswer(result.ai_response || 'No response received')
      } else {
        setAnswer('Failed to get AI insights.')
      }
    } catch (e) {
      setAnswer('Error: ' + (e as Error).message)
    } finally {
      setLoading(false)
    }
  }

  return (
    <div>
      <input
        type="text"
        placeholder="Ask economic insights..."
        value={question}
        onChange={(e) => setQuestion(e.target.value)}
        style={{ width: '70%', marginRight: '10px' }}
      />
      <button onClick={askQuestion} disabled={loading || !question.trim()}>
        {loading ? 'Getting answer...' : 'Ask AI'}
      </button>
      {answer && <div className="ai-answer" style={{ 
        marginTop: '30px', 
        padding: '20px', 
        backgroundColor: '#f8f9fa', 
        borderRadius: '8px',
        border: '1px solid #e9ecef',
        color: '#343a40',
      }}>{answer}</div>}
    </div>
  )
}
