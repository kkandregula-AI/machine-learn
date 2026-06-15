import React, { useState, useEffect, useRef } from 'react';
import { Play, Pause, RotateCcw, Trash2, HelpCircle } from 'lucide-react';

// Preset Datasets (Normalized between -0.8 and 0.8)
const PRESETS = {
  housing: [
    { x: -0.6, y: -0.5 },
    { x: -0.4, y: -0.3 },
    { x: -0.2, y: -0.1 },
    { x: 0.0, y: 0.1 },
    { x: 0.2, y: 0.3 },
    { x: 0.4, y: 0.5 },
    { x: 0.6, y: 0.7 }
  ],
  salary: [
    { x: -0.7, y: -0.6 },
    { x: -0.5, y: -0.4 },
    { x: -0.3, y: -0.1 },
    { x: -0.1, y: 0.1 },
    { x: 0.1, y: 0.2 },
    { x: 0.3, y: 0.4 },
    { x: 0.5, y: 0.65 },
    { x: 0.7, y: 0.8 }
  ],
  icecream: [
    { x: -0.6, y: -0.45 },
    { x: -0.4, y: -0.1 },
    { x: -0.2, y: -0.3 },
    { x: 0.0, y: 0.15 },
    { x: 0.2, y: 0.05 },
    { x: 0.4, y: 0.4 },
    { x: 0.6, y: 0.35 }
  ]
};

export default function RegressionVisualizer() {
  const [modelType, setModelType] = useState('linear'); // 'linear', 'quadratic', 'cubic'
  const [learningRate, setLearningRate] = useState(0.05);
  const [regType, setRegType] = useState('none'); // 'none', 'l1', 'l2'
  const [regStrength, setRegStrength] = useState(0.01);
  const [activePreset, setActivePreset] = useState('housing');

  // Weights state: [w0 (bias), w1 (x^1), w2 (x^2), w3 (x^3)]
  const [weights, setWeights] = useState([0, 0, 0, 0]);
  const [points, setPoints] = useState([...PRESETS.housing]);
  const [isTraining, setIsTraining] = useState(false);
  const [lossHistory, setLossHistory] = useState([]);
  const [epochCount, setEpochCount] = useState(0);

  const canvasRef = useRef(null);
  const animationRef = useRef(null);

  // Initialize weights based on model type
  const resetWeights = (type = modelType) => {
    setEpochCount(0);
    setLossHistory([]);
    if (type === 'linear') {
      setWeights([0, 0.5, 0, 0]); // bias=0, w1=0.5
    } else if (type === 'quadratic') {
      setWeights([0, 0.3, 0.2, 0]); // bias=0, w1=0.3, w2=0.2
    } else {
      setWeights([0, 0.2, 0.1, 0.2]); // w0, w1, w2, w3
    }
  };

  // Run on mount or preset switch
  useEffect(() => {
    resetWeights();
  }, [modelType]);

  // Load preset
  const loadPreset = (presetName) => {
    setActivePreset(presetName);
    if (presetName === 'custom') {
      setPoints([]);
      resetWeights();
      return;
    }
    setPoints([...PRESETS[presetName]]);
    resetWeights();
  };

  // Model Predict Function: y = w0 + w1*x + w2*x^2 + w3*x^3
  const predict = (x, w) => {
    return w[0] + w[1] * x + w[2] * Math.pow(x, 2) + w[3] * Math.pow(x, 3);
  };

  // Calculate Mean Squared Error Loss
  const calculateLoss = (w) => {
    if (points.length === 0) return 0;
    let sumSqrError = 0;
    points.forEach((p) => {
      const pred = predict(p.x, w);
      sumSqrError += Math.pow(pred - p.y, 2);
    });
    
    let baseLoss = sumSqrError / points.length;

    // Add Regularization Penalty (skip w[0] which is bias)
    if (regType === 'l1') {
      const penalty = regStrength * (Math.abs(w[1]) + Math.abs(w[2]) + Math.abs(w[3]));
      baseLoss += penalty;
    } else if (regType === 'l2') {
      const penalty = 0.5 * regStrength * (Math.pow(w[1], 2) + Math.pow(w[2], 2) + Math.pow(w[3], 2));
      baseLoss += penalty;
    }

    return baseLoss;
  };

  // Perform One Gradient Descent Step
  const stepGradientDescent = () => {
    if (points.length === 0) return;

    const n = points.length;
    let d_w0 = 0; // bias gradient
    let d_w1 = 0; // x^1 gradient
    let d_w2 = 0; // x^2 gradient
    let d_w3 = 0; // x^3 gradient

    // Compute gradients from data points
    points.forEach((p) => {
      const error = predict(p.x, weights) - p.y;
      d_w0 += error;
      d_w1 += error * p.x;
      d_w2 += error * Math.pow(p.x, 2);
      d_w3 += error * Math.pow(p.x, 3);
    });

    // Average the gradients
    d_w0 = (2 / n) * d_w0;
    d_w1 = (2 / n) * d_w1;
    d_w2 = (2 / n) * d_w2;
    d_w3 = (2 / n) * d_w3;

    // Apply Regularization derivative (bias is excluded)
    if (regType === 'l1') {
      d_w1 += regStrength * Math.sign(weights[1]);
      d_w2 += regStrength * Math.sign(weights[2]);
      d_w3 += regStrength * Math.sign(weights[3]);
    } else if (regType === 'l2') {
      d_w1 += regStrength * weights[1];
      d_w2 += regStrength * weights[2];
      d_w3 += regStrength * weights[3];
    }

    // Update Weights (safeguarded against explosion)
    const newWeights = [
      weights[0] - learningRate * d_w0,
      weights[1] - learningRate * d_w1,
      modelType !== 'linear' ? weights[2] - learningRate * d_w2 : 0,
      modelType === 'cubic' ? weights[3] - learningRate * d_w3 : 0
    ];

    // Check for NaN / numerical issues
    if (newWeights.some(isNaN) || newWeights.some(v => !isFinite(v))) {
      setIsTraining(false);
      alert('Loss exploded! Try decreasing the Learning Rate.');
      resetWeights();
      return;
    }

    setWeights(newWeights);
    const loss = calculateLoss(newWeights);
    setLossHistory((prev) => [...prev.slice(-100), loss]); // Store last 100 losses
    setEpochCount((prev) => prev + 1);
  };

  // Main Loop
  useEffect(() => {
    if (isTraining) {
      const loop = () => {
        // Run 2 GD steps per frame to make it visually faster and smoother
        stepGradientDescent();
        stepGradientDescent();
        animationRef.current = requestAnimationFrame(loop);
      };
      animationRef.current = requestAnimationFrame(loop);
    } else {
      cancelAnimationFrame(animationRef.current);
    }

    return () => cancelAnimationFrame(animationRef.current);
  }, [isTraining, weights, points, learningRate, modelType, regType, regStrength]);

  // Handle Canvas Click (add/remove points)
  const handleCanvasClick = (e) => {
    const canvas = canvasRef.current;
    const rect = canvas.getBoundingClientRect();
    const px = e.clientX - rect.left;
    const py = e.clientY - rect.top;

    // Map canvas pixel coords back to [-1, 1] normalized space
    const xVal = (px / canvas.width) * 2 - 1;
    const yVal = -((py / canvas.height) * 2 - 1); // Invert Y axis

    // Check if clicked close to an existing point to delete it
    const clickThreshold = 0.05;
    const existingIndex = points.findIndex(
      (p) => Math.sqrt(Math.pow(p.x - xVal, 2) + Math.pow(p.y - yVal, 2)) < clickThreshold
    );

    if (existingIndex !== -1) {
      // Delete point
      const newPoints = points.filter((_, idx) => idx !== existingIndex);
      setPoints(newPoints);
      setActivePreset('custom');
    } else {
      // Add point
      setPoints([...points, { x: xVal, y: yVal }]);
      setActivePreset('custom');
    }
    
    // Clear loss history when data changes to prevent graph misalignment
    setLossHistory([]);
    setEpochCount(0);
  };

  // Draw Scatter plot & Fitting Line
  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    const ctx = canvas.getContext('2d');
    const width = canvas.width;
    const height = canvas.height;

    // Clear background
    ctx.fillStyle = '#0b0b0e';
    ctx.fillRect(0, 0, width, height);

    // Draw grid lines
    ctx.strokeStyle = 'rgba(255,255,255,0.03)';
    ctx.lineWidth = 1;
    for (let i = 1; i < 10; i++) {
      // Vertical grid lines
      const gx = (i / 10) * width;
      ctx.beginPath();
      ctx.moveTo(gx, 0);
      ctx.lineTo(gx, height);
      ctx.stroke();

      // Horizontal grid lines
      const gy = (i / 10) * height;
      ctx.beginPath();
      ctx.moveTo(0, gy);
      ctx.lineTo(width, gy);
      ctx.stroke();
    }

    // Draw central axes
    ctx.strokeStyle = 'rgba(255,255,255,0.1)';
    ctx.lineWidth = 1.5;
    ctx.beginPath();
    ctx.moveTo(width / 2, 0);
    ctx.lineTo(width / 2, height);
    ctx.moveTo(0, height / 2);
    ctx.lineTo(width, height / 2);
    ctx.stroke();

    // Mapping helper function from [-1, 1] to canvas coordinates
    const mapX = (x) => ((x + 1) / 2) * width;
    const mapY = (y) => ((-y + 1) / 2) * height;

    // Draw Fitting Function (Line or Curve)
    ctx.strokeStyle = 'var(--color-regression)';
    ctx.lineWidth = 4;
    ctx.shadowBlur = 15;
    ctx.shadowColor = 'var(--color-regression-glow)';
    ctx.beginPath();

    const resolution = 100;
    for (let i = 0; i <= resolution; i++) {
      const normX = (i / resolution) * 2 - 1; // from -1 to 1
      const normY = predict(normX, weights);
      const cx = mapX(normX);
      const cy = mapY(normY);

      if (i === 0) {
        ctx.moveTo(cx, cy);
      } else {
        ctx.lineTo(cx, cy);
      }
    }
    ctx.stroke();
    ctx.shadowBlur = 0; // Reset shadow

    // Draw Data Points
    points.forEach((p) => {
      const cx = mapX(p.x);
      const cy = mapY(p.y);

      // Draw glowing dot
      ctx.beginPath();
      ctx.arc(cx, cy, 7, 0, 2 * Math.PI);
      ctx.fillStyle = '#ffffff';
      ctx.fill();
      ctx.lineWidth = 2;
      ctx.strokeStyle = 'var(--color-regression)';
      ctx.stroke();
    });
  }, [points, weights]);

  // Format Weights for display
  const renderFormula = () => {
    const b = weights[0].toFixed(2);
    const w1 = weights[1].toFixed(2);
    const w2 = weights[2].toFixed(2);
    const w3 = weights[3].toFixed(2);

    if (modelType === 'linear') {
      return `y = (${w1}) * x + (${b})`;
    } else if (modelType === 'quadratic') {
      return `y = (${w2}) * x² + (${w1}) * x + (${b})`;
    } else {
      return `y = (${w3}) * x³ + (${w2}) * x² + (${w1}) * x + (${b})`;
    }
  };

  const currentLoss = calculateLoss(weights);

  return (
    <div className="animate-fade-in" style={{ display: 'grid', gridTemplateColumns: '1fr', gap: '1.5rem' }}>
      {/* KPI stats top row */}
      <div className="stats-grid">
        <div className="stat-card" style={{ borderColor: 'var(--color-regression-glow)' }}>
          <span className="stat-label">Model Type</span>
          <span className="stat-value" style={{ fontSize: '1.15rem', color: 'var(--color-regression)', textTransform: 'capitalize' }}>
            {modelType} Model
          </span>
        </div>
        <div className="stat-card">
          <span className="stat-label">Training Epochs</span>
          <span className="stat-value">{epochCount}</span>
        </div>
        <div className="stat-card">
          <span className="stat-label">Current Loss (MSE)</span>
          <span className="stat-value" style={{ color: currentLoss < 0.05 ? '#6ee7b7' : '#fcd34d' }}>
            {points.length > 0 ? currentLoss.toFixed(5) : '0.00000'}
          </span>
        </div>
        <div className="stat-card">
          <span className="stat-label">Data Points</span>
          <span className="stat-value">{points.length}</span>
        </div>
      </div>

      <div className="dashboard-layout">
        {/* Control Panel Sidebar */}
        <div className="glass-card flex flex-col gap-4 accent-regression">
          <h3 style={{ borderBottom: '1px solid var(--border-color)', paddingBottom: '0.5rem' }}>Parameters</h3>
          
          {/* Datasets presets */}
          <div>
            <span className="form-label" style={{ marginBottom: '0.4rem' }}>Case Study Data</span>
            <div style={{ display: 'flex', flexWrap: 'wrap', gap: '0.4rem' }}>
              <button 
                className={`case-study-btn ${activePreset === 'housing' ? 'active-regression' : ''}`}
                onClick={() => loadPreset('housing')}
              >
                Housing Size
              </button>
              <button 
                className={`case-study-btn ${activePreset === 'salary' ? 'active-regression' : ''}`}
                onClick={() => loadPreset('salary')}
              >
                Salary Curve
              </button>
              <button 
                className={`case-study-btn ${activePreset === 'icecream' ? 'active-regression' : ''}`}
                onClick={() => loadPreset('icecream')}
              >
                Ice Cream Temp
              </button>
              <button 
                className={`case-study-btn ${activePreset === 'custom' ? 'active-regression' : ''}`}
                onClick={() => loadPreset('custom')}
              >
                Custom Draw
              </button>
            </div>
          </div>

          {/* Model selection */}
          <div className="form-group">
            <span className="form-label">Hypothesis function</span>
            <select value={modelType} onChange={(e) => setModelType(e.target.value)}>
              <option value="linear">Linear: y = wx + b</option>
              <option value="quadratic">Quadratic: y = w₂x² + w₁x + b</option>
              <option value="cubic">Cubic: y = w₃x³ + w₂x² + w₁x + b</option>
            </select>
          </div>

          {/* Learning Rate slider */}
          <div className="form-group">
            <span className="form-label">
              Learning Rate (η) <span className="form-value">{learningRate}</span>
            </span>
            <input 
              type="range" 
              className="slider-regression"
              min="0.005" 
              max="0.3" 
              step="0.005"
              value={learningRate} 
              onChange={(e) => setLearningRate(parseFloat(e.target.value))}
            />
          </div>

          {/* Regularization Type */}
          <div className="form-group">
            <span className="form-label">Regularization Penalty</span>
            <div style={{ display: 'grid', gridTemplateColumns: 'repeat(3, 1fr)', gap: '0.25rem' }}>
              {['none', 'l1', 'l2'].map((type) => (
                <button
                  key={type}
                  className="case-study-btn"
                  style={{
                    backgroundColor: regType === type ? 'rgba(6, 182, 212, 0.15)' : 'rgba(255,255,255,0.02)',
                    borderColor: regType === type ? 'var(--color-regression)' : 'var(--border-color)',
                    color: regType === type ? '#fff' : 'var(--text-secondary)'
                  }}
                  onClick={() => setRegType(type)}
                >
                  {type.toUpperCase()}
                </button>
              ))}
            </div>
          </div>

          {/* Regularization Strength Slider (Conditional) */}
          {regType !== 'none' && (
            <div className="form-group">
              <span className="form-label">
                Strength (λ) <span className="form-value">{regStrength.toFixed(3)}</span>
              </span>
              <input 
                type="range" 
                className="slider-regression"
                min="0.001" 
                max="0.1" 
                step="0.001"
                value={regStrength} 
                onChange={(e) => setRegStrength(parseFloat(e.target.value))}
              />
            </div>
          )}

          {/* Training Action buttons */}
          <div style={{ display: 'flex', flexDirection: 'column', gap: '0.5rem', marginTop: '0.5rem' }}>
            <button 
              className="btn btn-regression" 
              onClick={() => setIsTraining(!isTraining)}
              disabled={points.length === 0}
            >
              {isTraining ? <Pause size={16} /> : <Play size={16} />}
              {isTraining ? 'Pause Training' : 'Train Model'}
            </button>
            <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '0.5rem' }}>
              <button className="btn btn-secondary" onClick={() => stepGradientDescent()} disabled={isTraining || points.length === 0}>
                Step
              </button>
              <button className="btn btn-secondary" onClick={() => resetWeights()}>
                <RotateCcw size={14} />
                Reset
              </button>
            </div>
            <button className="btn btn-secondary" style={{ color: 'var(--color-danger)', borderColor: 'rgba(239, 68, 68, 0.15)' }} onClick={() => { setPoints([]); setActivePreset('custom'); resetWeights(); }}>
              <Trash2 size={14} />
              Clear Points
            </button>
          </div>
        </div>

        {/* Center Main Work Space: Plot Canvas & Live formula */}
        <div className="flex flex-col gap-4">
          <div className="glass-card flex flex-col gap-3" style={{ padding: '1.25rem' }}>
            <div className="flex justify-between items-center" style={{ flexWrap: 'wrap', gap: '0.5rem' }}>
              <div className="font-mono" style={{ fontSize: '0.9rem', color: 'var(--color-regression)', wordBreak: 'break-all' }}>
                Equation: {renderFormula()}
              </div>
              <span style={{ fontSize: '0.75rem', color: 'var(--text-muted)', display: 'flex', alignItems: 'center', gap: '0.25rem' }}>
                <HelpCircle size={12} />
                Click graph to add/remove points
              </span>
            </div>
            
            <div style={{ position: 'relative', width: '100%', aspectRatio: '1.5', maxHeight: '420px', minHeight: '280px' }}>
              <canvas 
                ref={canvasRef} 
                width={700}
                height={460}
                style={{ 
                  width: '100%', 
                  height: '100%', 
                  borderRadius: '12px', 
                  border: '1px solid var(--border-color)',
                  cursor: 'crosshair',
                  display: 'block'
                }} 
                onClick={handleCanvasClick}
              />
            </div>
          </div>

          {/* Loss Curve Plot in SVG */}
          <div className="glass-card" style={{ padding: '1.25rem' }}>
            <h4 style={{ fontSize: '0.85rem', color: 'var(--text-secondary)', marginBottom: '0.5rem' }}>Training Loss History</h4>
            <div style={{ width: '100%', height: '80px', position: 'relative' }}>
              {lossHistory.length > 1 ? (
                <svg width="100%" height="100%" viewBox="0 0 400 80" preserveAspectRatio="none">
                  <defs>
                    <linearGradient id="lossGrad" x1="0" y1="0" x2="0" y2="1">
                      <stop offset="0%" stopColor="var(--color-regression)" stopOpacity="0.4" />
                      <stop offset="100%" stopColor="var(--color-regression)" stopOpacity="0" />
                    </linearGradient>
                  </defs>
                  {/* Draw filled area */}
                  <path
                    d={`
                      M 0 80
                      ${lossHistory.map((val, idx) => {
                        const x = (idx / (lossHistory.length - 1)) * 400;
                        // Normalize loss height (cap max at double the min loss, or absolute max)
                        const maxL = Math.max(...lossHistory);
                        const minL = Math.min(...lossHistory);
                        const range = maxL - minL || 1;
                        const y = 75 - ((val - minL) / range) * 70;
                        return `L ${x} ${y}`;
                      }).join(' ')}
                      L 400 80 Z
                    `}
                    fill="url(#lossGrad)"
                  />
                  {/* Draw stroke line */}
                  <path
                    d={lossHistory.map((val, idx) => {
                      const x = (idx / (lossHistory.length - 1)) * 400;
                      const maxL = Math.max(...lossHistory);
                      const minL = Math.min(...lossHistory);
                      const range = maxL - minL || 1;
                      const y = 75 - ((val - minL) / range) * 70;
                      return `${idx === 0 ? 'M' : 'L'} ${x} ${y}`;
                    }).join(' ')}
                    fill="none"
                    stroke="var(--color-regression)"
                    strokeWidth="2.5"
                  />
                </svg>
              ) : (
                <div style={{ height: '100%', display: 'flex', alignItems: 'center', justifyContent: 'center', color: 'var(--text-muted)', fontSize: '0.8rem' }}>
                  No loss data yet. Start training to plot loss curve.
                </div>
              )}
            </div>
            <div className="flex justify-between" style={{ fontSize: '0.7rem', color: 'var(--text-muted)', marginTop: '0.2rem' }}>
              <span>Start</span>
              <span>Epoch: {epochCount}</span>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}
