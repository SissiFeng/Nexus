import { useState, useEffect, useCallback, useRef } from 'react'
import { 
  Play, 
  Pause, 
  RotateCcw, 
  TrendingUp, 
  Target, 
  Zap,
  ChevronRight,
  BarChart3,
  Info
} from 'lucide-react'

interface DataPoint {
  x: number
  y: number
  value: number
  iteration: number
}

interface SampleDataset {
  id: string
  name: string
  description: string
  params: string[]
  objective: string
  data: DataPoint[]
  bestValue: number
}

const SAMPLE_DATASETS: SampleDataset[] = [
  {
    id: 'catalyst',
    name: 'Catalyst Optimization',
    description: 'Optimize catalytic reaction yield based on temperature and pressure',
    params: ['Temperature (°C)', 'Pressure (bar)'],
    objective: 'Yield (%)',
    data: [
      { x: 80, y: 1.5, value: 45, iteration: 1 },
      { x: 120, y: 2.0, value: 62, iteration: 2 },
      { x: 100, y: 2.5, value: 58, iteration: 3 },
      { x: 140, y: 1.8, value: 71, iteration: 4 },
      { x: 160, y: 2.2, value: 78, iteration: 5 },
      { x: 150, y: 2.8, value: 74, iteration: 6 },
      { x: 170, y: 2.0, value: 82, iteration: 7 },
      { x: 165, y: 2.3, value: 85, iteration: 8 },
      { x: 175, y: 2.1, value: 87, iteration: 9 },
      { x: 172, y: 2.2, value: 89, iteration: 10 },
    ],
    bestValue: 89
  },
  {
    id: 'formulation',
    name: 'Drug Formulation',
    description: 'Optimize drug formulation for maximum dissolution rate',
    params: ['API Concentration (%)', 'Excipient Ratio'],
    objective: 'Dissolution Rate (mg/min)',
    data: [
      { x: 20, y: 0.5, value: 12, iteration: 1 },
      { x: 35, y: 0.8, value: 18, iteration: 2 },
      { x: 45, y: 0.6, value: 22, iteration: 3 },
      { x: 40, y: 1.0, value: 28, iteration: 4 },
      { x: 50, y: 0.9, value: 32, iteration: 5 },
      { x: 55, y: 1.1, value: 35, iteration: 6 },
      { x: 48, y: 1.2, value: 38, iteration: 7 },
      { x: 52, y: 1.15, value: 41, iteration: 8 },
      { x: 50, y: 1.18, value: 43, iteration: 9 },
      { x: 51, y: 1.17, value: 44, iteration: 10 },
    ],
    bestValue: 44
  },
  {
    id: 'materials',
    name: 'Material Synthesis',
    description: 'Optimize material properties through process parameters',
    params: ['Sintering Temp (°C)', 'Time (hours)'],
    objective: 'Conductivity (S/m)',
    data: [
      { x: 400, y: 2, value: 150, iteration: 1 },
      { x: 600, y: 3, value: 280, iteration: 2 },
      { x: 550, y: 4, value: 320, iteration: 3 },
      { x: 700, y: 2.5, value: 410, iteration: 4 },
      { x: 750, y: 3.5, value: 480, iteration: 5 },
      { x: 800, y: 3, value: 520, iteration: 6 },
      { x: 780, y: 3.2, value: 545, iteration: 7 },
      { x: 820, y: 3.1, value: 568, iteration: 8 },
      { x: 810, y: 3.15, value: 582, iteration: 9 },
      { x: 815, y: 3.12, value: 591, iteration: 10 },
    ],
    bestValue: 591
  }
]

function InteractivePlayground() {
  const [selectedDataset, setSelectedDataset] = useState<SampleDataset>(SAMPLE_DATASETS[0])
  const [isRunning, setIsRunning] = useState(false)
  const [currentIteration, setCurrentIteration] = useState(0)
  const [visiblePoints, setVisiblePoints] = useState<DataPoint[]>([])
  const [bestValue, setBestValue] = useState(0)
  const [showExplanation, setShowExplanation] = useState(true)
  const intervalRef = useRef<ReturnType<typeof setInterval> | null>(null)

  const reset = useCallback(() => {
    setIsRunning(false)
    setCurrentIteration(0)
    setVisiblePoints([])
    setBestValue(0)
    if (intervalRef.current) {
      clearInterval(intervalRef.current)
    }
  }, [])

  const runIteration = useCallback(() => {
    if (currentIteration < selectedDataset.data.length) {
      const newPoint = selectedDataset.data[currentIteration]
      setVisiblePoints(prev => [...prev, newPoint])
      setBestValue(prev => Math.max(prev, newPoint.value))
      setCurrentIteration(prev => prev + 1)
    } else {
      setIsRunning(false)
    }
  }, [currentIteration, selectedDataset])

  useEffect(() => {
    reset()
  }, [selectedDataset, reset])

  useEffect(() => {
    if (isRunning) {
      intervalRef.current = setInterval(() => {
        runIteration()
      }, 800)
    } else if (intervalRef.current) {
      clearInterval(intervalRef.current)
    }

    return () => {
      if (intervalRef.current) {
        clearInterval(intervalRef.current)
      }
    }
  }, [isRunning, runIteration])

  const getConvergenceData = () => {
    return visiblePoints.map((_, i) => ({
      iteration: i + 1,
      best: Math.max(...visiblePoints.slice(0, i + 1).map(p => p.value))
    }))
  }

  const maxValue = Math.max(...selectedDataset.data.map(d => d.value))
  const minValue = Math.min(...selectedDataset.data.map(d => d.value))

  return (
    <section id="playground" className="section playground-section">
      <div className="container">
        <div className="section-header">
          <span className="section-tag">Try It Now</span>
          <h2 className="section-title">
            Interactive Optimization Demo
          </h2>
          <p className="section-subtitle">
            Experience Bayesian optimization in action. No installation required — 
            run simulations directly in your browser with real sample data.
          </p>
        </div>

        {showExplanation && (
          <div className="playground-explanation">
            <div className="explanation-header">
              <Info size={20} />
              <span>How it works</span>
              <button 
                className="explanation-close"
                onClick={() => setShowExplanation(false)}
              >
                ×
              </button>
            </div>
            <div className="explanation-steps">
              <div className="explanation-step">
                <div className="step-number">1</div>
                <div className="step-content">
                  <strong>Select a dataset</strong>
                  <p>Choose from real-world scenarios: catalyst optimization, drug formulation, or material synthesis.</p>
                </div>
              </div>
              <div className="explanation-step">
                <div className="step-number">2</div>
                <div className="step-content">
                  <strong>Run optimization</strong>
                  <p>Watch as the algorithm intelligently explores the parameter space to find optimal conditions.</p>
                </div>
              </div>
              <div className="explanation-step">
                <div className="step-number">3</div>
                <div className="step-content">
                  <strong>Analyze results</strong>
                  <p>View convergence curves and parameter importance in real-time.</p>
                </div>
              </div>
            </div>
          </div>
        )}

        <div className="playground-container">
          {/* Dataset Selection */}
          <div className="playground-sidebar">
            <div className="playground-panel">
              <h3 className="panel-title">
                <Target size={18} />
                Select Dataset
              </h3>
              <div className="dataset-list">
                {SAMPLE_DATASETS.map(dataset => (
                  <button
                    key={dataset.id}
                    className={`dataset-card ${selectedDataset.id === dataset.id ? 'active' : ''}`}
                    onClick={() => setSelectedDataset(dataset)}
                  >
                    <div className="dataset-name">{dataset.name}</div>
                    <div className="dataset-desc">{dataset.description}</div>
                    <div className="dataset-params">
                      {dataset.params.map((p, i) => (
                        <span key={i} className="param-tag">{p}</span>
                      ))}
                    </div>
                  </button>
                ))}
              </div>
            </div>

            <div className="playground-panel">
              <h3 className="panel-title">
                <Zap size={18} />
                Controls
              </h3>
              <div className="control-buttons">
                <button
                  className="control-btn primary"
                  onClick={() => setIsRunning(!isRunning)}
                  disabled={currentIteration >= selectedDataset.data.length}
                >
                  {isRunning ? <Pause size={18} /> : <Play size={18} />}
                  {isRunning ? 'Pause' : currentIteration === 0 ? 'Start Optimization' : 'Continue'}
                </button>
                <button
                  className="control-btn secondary"
                  onClick={reset}
                >
                  <RotateCcw size={18} />
                  Reset
                </button>
              </div>
              
              <div className="progress-info">
                <div className="progress-label">
                  <span>Progress</span>
                  <span>{currentIteration} / {selectedDataset.data.length} iterations</span>
                </div>
                <div className="progress-bar">
                  <div 
                    className="progress-fill"
                    style={{ width: `${(currentIteration / selectedDataset.data.length) * 100}%` }}
                  />
                </div>
              </div>
            </div>

            <div className="playground-panel stats-panel">
              <h3 className="panel-title">
                <TrendingUp size={18} />
                Current Results
              </h3>
              <div className="stats-grid">
                <div className="stat-box">
                  <div className="stat-label">Best {selectedDataset.objective}</div>
                  <div className="stat-value highlight">{bestValue.toFixed(1)}</div>
                </div>
                <div className="stat-box">
                  <div className="stat-label">Improvement</div>
                  <div className="stat-value">
                    {visiblePoints.length > 1 
                      ? `+${((bestValue - visiblePoints[0]?.value) / visiblePoints[0]?.value * 100).toFixed(0)}%`
                      : '—'}
                  </div>
                </div>
                <div className="stat-box">
                  <div className="stat-label">Efficiency</div>
                  <div className="stat-value">
                    {currentIteration > 0 
                      ? `${(bestValue / maxValue * 100).toFixed(0)}%`
                      : '—'}
                  </div>
                </div>
              </div>
            </div>
          </div>

          {/* Visualization Area */}
          <div className="playground-visualization">
            <div className="viz-content">
              {/* Parameter Space Plot */}
              <div className="plot-container">
                <div className="plot-header">
                  <span className="plot-title">Parameter Space Exploration</span>
                  <span className="plot-subtitle">
                    {selectedDataset.params[0]} vs {selectedDataset.params[1]}
                  </span>
                </div>
                <div className="scatter-plot">
                  <div className="plot-y-axis">
                    <span>{selectedDataset.params[1].split(' ')[0]}</span>
                  </div>
                  <div className="plot-area">
                    {visiblePoints.map((point, i) => {
                      const xRange = Math.max(...selectedDataset.data.map(d => d.x)) - Math.min(...selectedDataset.data.map(d => d.x))
                      const yRange = Math.max(...selectedDataset.data.map(d => d.y)) - Math.min(...selectedDataset.data.map(d => d.y))
                      const xMin = Math.min(...selectedDataset.data.map(d => d.x))
                      const yMin = Math.min(...selectedDataset.data.map(d => d.y))
                      
                      const xPercent = ((point.x - xMin) / xRange) * 80 + 10
                      const yPercent = 100 - ((point.y - yMin) / yRange) * 80 - 10
                      const _pointValue = point.value
                      const intensity = (_pointValue - minValue) / (maxValue - minValue)
                      
                      return (
                        <div
                          key={i}
                          className="plot-point"
                          style={{
                            left: `${xPercent}%`,
                            top: `${yPercent}%`,
                            background: `hsl(${220 + intensity * 60}, 80%, ${50 + intensity * 20}%)`,
                            transform: 'translate(-50%, -50%) scale(1)',
                            animation: 'pointAppear 0.3s ease-out'
                          }}
                          title={`Iteration ${point.iteration}: ${point.value.toFixed(1)}`}
                        >
                          <span className="point-label">{i + 1}</span>
                        </div>
                      )
                    })}
                  </div>
                  <div className="plot-x-axis">
                    <span>{selectedDataset.params[0].split(' ')[0]}</span>
                  </div>
                </div>
                <div className="plot-legend">
                  <div className="legend-item">
                    <div className="legend-color" style={{ background: 'hsl(220, 80%, 50%)' }} />
                    <span>Lower values</span>
                  </div>
                  <div className="legend-item">
                    <div className="legend-color" style={{ background: 'hsl(280, 80%, 70%)' }} />
                    <span>Higher values</span>
                  </div>
                </div>
              </div>

              {/* Convergence Plot */}
              <div className="plot-container">
                <div className="plot-header">
                  <span className="plot-title">Convergence Curve</span>
                  <span className="plot-subtitle">Best value over iterations</span>
                </div>
                <div className="convergence-plot">
                  {getConvergenceData().length > 0 ? (
                    <svg viewBox="0 0 300 150" className="convergence-svg">
                      {/* Grid lines */}
                      {[0, 1, 2, 3, 4].map(i => (
                        <line
                          key={`h-${i}`}
                          x1="0"
                          y1={i * 37.5}
                          x2="300"
                          y2={i * 37.5}
                          stroke="var(--color-border)"
                          strokeWidth="1"
                          strokeDasharray="4"
                        />
                      ))}
                      
                      {/* Area fill */}
                      {getConvergenceData().length >= 1 && (
                        <polygon
                          points={`0,150 ${getConvergenceData().map((d, i) => {
                            const denominator = selectedDataset.data.length - 1 || 1
                            const x = (i / denominator) * 280 + 10
                            const y = 150 - ((d.best - minValue) / (maxValue - minValue || 1)) * 130 - 10
                            return `${x},${y}`
                          }).join(' ')} 300,150`}
                          fill="url(#areaGradient)"
                          opacity="0.3"
                        />
                      )}
                      
                      {/* Line */}
                      {getConvergenceData().length >= 1 && (
                        <polyline
                          points={getConvergenceData().map((d, i) => {
                            const denominator = selectedDataset.data.length - 1 || 1
                            const x = (i / denominator) * 280 + 10
                            const y = 150 - ((d.best - minValue) / (maxValue - minValue || 1)) * 130 - 10
                            return `${x},${y}`
                          }).join(' ')}
                          fill="none"
                          stroke="var(--color-primary)"
                          strokeWidth="2"
                          strokeLinecap="round"
                          strokeLinejoin="round"
                        />
                      )}
                      
                      {/* Data points */}
                      {getConvergenceData().map((d, i) => {
                        const denominator = selectedDataset.data.length - 1 || 1
                        const x = (i / denominator) * 280 + 10
                        const y = 150 - ((d.best - minValue) / (maxValue - minValue || 1)) * 130 - 10
                        return (
                          <circle
                            key={i}
                            cx={x}
                            cy={y}
                            r="5"
                            fill="var(--color-primary)"
                            stroke="white"
                            strokeWidth="2"
                          />
                        )
                      })}
                      
                      <defs>
                        <linearGradient id="areaGradient" x1="0" y1="0" x2="0" y2="1">
                          <stop offset="0%" stopColor="var(--color-primary)" stopOpacity="0.5" />
                          <stop offset="100%" stopColor="var(--color-primary)" stopOpacity="0" />
                        </linearGradient>
                      </defs>
                    </svg>
                  ) : (
                    <div className="plot-empty">
                      <BarChart3 size={48} />
                      <p>Click "Start Optimization" to see convergence</p>
                    </div>
                  )}
                </div>
                <div className="plot-footer">
                  <span>Iteration</span>
                  <div className="iteration-markers">
                    {Array.from({ length: 5 }, (_, i) => (
                      <span key={i}>{Math.round((i / 4) * (selectedDataset.data.length - 1)) + 1}</span>
                    ))}
                  </div>
                </div>
              </div>
            </div>

            {/* Suggestion Cards */}
            {currentIteration > 0 && currentIteration < selectedDataset.data.length && (
              <div className="suggestions-preview">
                <div className="suggestions-header">
                  <Target size={16} />
                  <span>Next Suggested Experiment</span>
                </div>
                <div className="suggestion-card next">
                  <div className="suggestion-number">#{currentIteration + 1}</div>
                  <div className="suggestion-params">
                    {selectedDataset.params.map((param, i) => {
                      const key = i === 0 ? 'x' : 'y'
                      const nextPoint = selectedDataset.data[currentIteration]
                      return (
                        <div key={param} className="suggestion-param">
                          <span className="param-name">{param.split(' ')[0]}:</span>
                          <span className="param-value">{nextPoint[key as keyof DataPoint]}</span>
                        </div>
                      )
                    })}
                  </div>
                  <div className="suggestion-predicted">
                    <span className="predicted-label">Predicted {selectedDataset.objective}:</span>
                    <span className="predicted-value">
                      {selectedDataset.data[currentIteration].value.toFixed(1)}
                    </span>
                  </div>
                  <ChevronRight size={20} className="suggestion-arrow" />
                </div>
              </div>
            )}

            {currentIteration >= selectedDataset.data.length && (
              <div className="optimization-complete">
                <div className="complete-badge">✓</div>
                <h4>Optimization Complete!</h4>
                <p>
                  Found optimal parameters with {selectedDataset.objective} = {bestValue.toFixed(1)}
                </p>
                <div className="optimal-params">
                  <strong>Optimal Conditions:</strong>
                  {selectedDataset.params.map((param, i) => {
                    const bestPoint = selectedDataset.data.reduce((max, p) => p.value > max.value ? p : max)
                    const key = i === 0 ? 'x' : 'y'
                    return (
                      <div key={param} className="optimal-param">
                        {param}: <strong>{bestPoint[key as keyof DataPoint]}</strong>
                      </div>
                    )
                  })}
                </div>
              </div>
            )}
          </div>
        </div>
      </div>
    </section>
  )
}

export default InteractivePlayground
