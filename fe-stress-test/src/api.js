const API_BASE = "http://localhost:8000/api";

export async function uploadCsv(file) {
  const formData = new FormData();
  formData.append("file", file);
  const response = await fetch(`${API_BASE}/upload-csv`, {
    method: "POST",
    body: formData,
  });
  return await response.json();
}

export async function getPredictions() {
  const response = await fetch(`${API_BASE}/predict-future`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ years: 5 }),
  });
  return await response.json();
}

export const stressTestQuery = async (requestData) => {
  // If requestData is a string (old way), convert to object
  const payload =
    typeof requestData === "string" ? { query: requestData } : requestData;

  const response = await fetch(`${API_BASE}/stress-test`, {
    method: "POST",
    headers: {
      "Content-Type": "application/json",
    },
    body: JSON.stringify(payload),
  });

  return response.json();
};

export async function getAIInsights(question) {
  const response = await fetch(`${API_BASE}/ai-insights`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ question }),
  })
  return await response.json()
}
