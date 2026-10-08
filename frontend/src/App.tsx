import React, { useState, useEffect } from 'react'
import axios from 'axios'

function App() {
  const [imageUrl, setImageUrl] = useState<string | null>(null)
  const [mcSamples, setMcSamples] = useState(30)
  const [isLoading, setIsLoading] = useState(false)
  const [error, setError] = useState<string | null>(null)
  const [results, setResults] = useState<{
    segmentation: string
    uncertainty: string
    overlay: string
    metrics: {
      mean_confidence: number
      max_uncertainty: number
      mean_uncertainty: number
      road_coverage: number
    }
} | null>(null)

  const handleFileChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0]
    if (!file) return

    const reader = new FileReader()
    reader.onload = (e) => {
      setImageUrl(e.target?.result as string)
      setError(null)
      setResults(null)
    }
    reader.readAsDataURL(file)
  }

  const handlePredict = async () => {
    if (!imageUrl) return

    setIsLoading(true)
    setError(null)

    try {
      const formData = new FormData()
      formData.append('file', await (async () => {
        const res = await fetch(imageUrl)
        const blob = await res.blob()
        return blob
      })(), 'image.jpg')
      formData.append('mc_samples', mcSamples.toString())

      const response = await axios.post(
        'http://localhost:8000/predict',
        formData,
        {
          headers: {
            'Content-Type': 'multipart/form-data',
          },
          onUploadProgress: (progress) => {
            console.log(`Progress: ${progress.loaded} / ${progress.total}`)
          },
        }
      )

      setResults(response.data)
    } catch (err: any) {
      setError(`Inference failed: ${err.message}`)
    } finally {
      setIsLoading(false)
    }
  }

  return (
    <div className="min-h-screen bg-gradient-to-b from-gray-900 via-gray-700 to-gray-500 text-white p-6">
      <header className="text-center mb-12">
        <h1 className="text-4xl font-bold mb-2">
          Uncertainty-Aware Road Segmentation
        </h1>
        <p className="text-gray-300">
          Upload a road image to segment drivable regions with confidence estimation
        </p>
      </header>

      <div className="max-w-6xl mx-auto">
        {/* Upload Section */}
        <section className="mb-12">
          <h2 className="text-xl font-semibold mb-4">Input</h2>
          <div className="border-2 border-dashed border-gray-600 rounded-lg p-8 text-center cursor-pointer hover:border-primary transition-colors">
            <input
              type="file"
              id="fileInput"
              className="hidden"
              accept="image/*"
              onChange={handleFileChange}
            />
            <div
              onClick={() => document.getElementById('fileInput')?.click()}
              className="hover:hover:bg-gray-800"
            >
              <div className="text-4xl mb-4">&#128247;</div>
              <p>Drag & drop an image here or click to browse</p>
            </div>
          </div>

          {imageUrl && (
            <img
              src={imageUrl}
              alt="Uploaded image"
              className="w-full rounded mt-4 max-h-64 object-contain"
            />
          )}

          <div className="mt-6 controls">
            <label className="block text-sm text-gray-300 mb-2">
              MC Dropout Samples{' '}
              <span className="font-mono text-blue-400">{mcSamples}</span>
            </label>
            <input
              type="range"
              min="1"
              max="100"
              value={mcSamples}
              onChange={(e) => setMcSamples(Number(e.target.value))}
              className="w-full accent-color-primary mt-1"
            />
            <button
              onClick={handlePredict}
              disabled={isLoading}
              className={`btn btn-primary w-full py-3 rounded mt-4 transition-colors ${
                isLoading ? 'opacity-50 cursor-not-allowed' : ''
              }`}
            >
              {isLoading ? 'Running Inference...' : 'Run Inference'}
            </button>
          </div>
        </section>

        {/* Results Section */}
        {results && (
          <section>
            <h2 className="text-xl font-semibold mb-6">Results</h2>
            <div className="grid grid-cols-1 md:grid-cols-2 gap-8">
              <div>
                <h3 className="text-lg font-medium mb-4">Segmentation Mask</h3>
                <img
                  src={results.segmentation}
                  alt="Segmentation mask"
                  className="rounded w-full"
                />
              </div>
              <div>
                <h3 className="text-lg font-medium mb-4">Uncertainty Heatmap</h3>
                <img
                  src={results.uncertainty}
                  alt="Uncertainty heatmap"
                  className="rounded w-full"
                />
              </div>
            </div>

            <div className="mt-8">
              <h3 className="text-lg font-medium mb-4">Metrics</h3>
              <div className="grid grid-cols-2 gap-4">
                <div className="bg-gray-800 rounded-lg p-4">
                  <p className="text-sm text-gray-400 mb-2">Mean Confidence</p>
                  <p className="text-2xl font-bold text-cyan-400">{results.metrics.mean_confidence.toFixed(4)}</p>
                </div>
                <div className="bg-gray-800 rounded-lg p-4">
                  <p className="text-sm text-gray-400 mb-2">Max Uncertainty</p>
                  <p className="text-2xl font-bold text-red-400">{results.metrics.max_uncertainty.toFixed(4)}</p>
                </div>
                <div className="bg-gray-800 rounded-lg p-4">
                  <p className="text-sm text-gray-400 mb-2">Mean Uncertainty</p>
                  <p className="text-2xl font-bold text-orange-400">{results.metrics.mean_uncertainty.toFixed(4)}</p>
                </div>
                <div className="bg-gray-800 rounded-lg p-4">
                  <p className="text-sm text-gray-400 mb-2">Road Coverage</p>
                  <p className="text-2xl font-bold text-green-400">
                    {results.metrics.road_coverage.toFixed(1)}%
                  </p>
                </div>
              </div>
            </div>
          </section>
        )}

        {error && (
          <div className="mt-8 p-4 rounded-lg bg-red-600 text-white text-center">
            <p>{error}</p>
          </div>
        )}
      </div>
    </div>
  )
}

export default App