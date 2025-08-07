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
        setAnswer(result.answer || 'No response received')
      } else {
        setAnswer('Failed to get AI insights.')
      }
    } catch (e) {
      setAnswer('Error: ' + (e as Error).message)
    }
    setLoading(false)
  }

  return (
    <div>
      <input
        type="text"
        placeholder="Ask economic insights..."
        value={question}
        onChange={(e) => setQuestion(e.target.value)}
      />
      <button onClick={askQuestion} disabled={loading}>
        {loading ? 'Getting answer...' : 'Ask AI'}
      </button>
      {answer && <div className="ai-answer">{answer}</div>}
    </div>
  )
}
