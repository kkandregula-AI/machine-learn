import React, { useState, useEffect, useRef } from 'react';
import { Play, Pause, RotateCcw, Plus, Minus, Info } from 'lucide-react';

// Dataset generators
const generateGaussianData = (n = 100) => {
  const points = [];
  for (let i = 0; i < n; i++) {
    const label = Math.random() > 0.5 ? 1 : 0;
    const r = 0.2 + Math.random() * 0.15;
    const theta = Math.random() * 2 * Math.PI;
    if (label === 1) {
      points.push({
        x: -0.4 + r * Math.cos(theta),
        y: 0.4 + r * Math.sin(theta),
        label
      });
    } else {
      points.push({
        x: 0.4 + r * Math.cos(theta),
        y: -0.4 + r * Math.sin(theta),
        label
      });
    }
  }
  return points;
};

const generateXORData = (n = 120) => {
  const points = [];
  for (let i = 0; i < n; i++) {
    const x = Math.random() * 2 - 1;
    const y = Math.random() * 2 - 1;
    // Buffer near axes for clear boundary
    if (Math.abs(x) < 0.1 || Math.abs(y) < 0.1) {
      i--;
      continue;
    }
    const label = (x > 0 && y > 0) || (x < 0 && y < 0) ? 1 : 0;
    points.push({ x, y, label });
  }
  return points;
};

const generateCircleData = (n = 120) => {
  const points = [];
  for (let i = 0; i < n; i++) {
    const label = Math.random() > 0.5 ? 1 : 0;
    const theta = Math.random() * 2 * Math.PI;
    if (label === 1) {
      // Outer ring
      const r = 0.65 + Math.random() * 0.15;
      points.push({ x: r * Math.cos(theta), y: r * Math.sin(theta), label });
    } else {
      // Inner circle
      const r = Math.random() * 0.35;
      points.push({ x: r * Math.cos(theta), y: r * Math.sin(theta), label });
    }
  }
  return points;
};

const generateSpiralData = (n = 160) => {
  const points = [];
  // Spiral 1
  for (let i = 0; i < n / 2; i++) {
    const r = (i / (n / 2)) * 0.85;
    const theta = (i / (n / 2)) * 2.5 * Math.PI;
    const noiseX = (Math.random() * 2 - 1) * 0.04;
    const noiseY = (Math.random() * 2 - 1) * 0.04;
    points.push({
      x: r * Math.cos(theta) + noiseX,
      y: r * Math.sin(theta) + noiseY,
      label: 1
    });
  }
  // Spiral 2 (180 degrees rotated)
  for (let i = 0; i < n / 2; i++) {
    const r = (i / (n / 2)) * 0.85;
    const theta = (i / (n / 2)) * 2.5 * Math.PI + Math.PI;
    const noiseX = (Math.random() * 2 - 1) * 0.04;
    const noiseY = (Math.random() * 2 - 1) * 0.04;
    points.push({
      x: r * Math.cos(theta) + noiseX,
      y: r * Math.sin(theta) + noiseY,
      label: 0
    });
  }
  return points;
};

export default function NeuralNetworkVisualizer() {
  const [hiddenLayers, setHiddenLayers] = useState([4, 2]); // Neurons per hidden layer
  const [learningRate, setLearningRate] = useState(0.08);
  const [activation, setActivation] = useState('tanh'); // 'relu', 'sigmoid', 'tanh', 'linear'
  const [activeDataset, setActiveDataset] = useState('xor');
  const [epochCount, setEpochCount] = useState(0);
  const [accuracy, setAccuracy] = useState(0);
  const [loss, setLoss] = useState(0);
  const [isTraining, setIsTraining] = useState(false);
  const [points, setPoints] = useState([]);

  // Network weights & biases state
  // We keep weights and biases as a ref to do fast training loops without causing re-render lag,
  // and sync to visual state periodically or when training pauses.
  const networkRef = useRef({ weights: [], biases: [] });
  const [networkState, setNetworkState] = useState({ weights: [], biases: [] });

  const boundaryCanvasRef = useRef(null);
  const animationRef = useRef(null);

  // Load dataset points
  useEffect(() => {
    let data = [];
    if (activeDataset === 'gaussian') data = generateGaussianData();
    else if (activeDataset === 'xor') data = generateXORData();
    else if (activeDataset === 'circle') data = generateCircleData();
    else if (activeDataset === 'spiral') data = generateSpiralData();
    setPoints(data);
    initializeNetwork(hiddenLayers, data);
  }, [activeDataset]);

  // Topology changes helper
  const updateTopology = (newHidden) => {
    setHiddenLayers(newHidden);
    initializeNetwork(newHidden, points);
  };

  // Activation function math
  const actFn = (x) => {
    if (activation === 'relu') return Math.max(0, x);
    if (activation === 'sigmoid') return 1 / (1 + Math.exp(-x));
    if (activation === 'tanh') return Math.tanh(x);
    return x; // linear
  };

  const actDeriv = (actVal) => {
    if (activation === 'relu') return actVal > 0 ? 1 : 0;
    if (activation === 'sigmoid') return actVal * (1 - actVal);
    if (activation === 'tanh') return 1 - Math.pow(actVal, 2);
    return 1; // linear
  };

  // Initialize weights & biases (Xavier-like initialization)
  const initializeNetwork = (hidden = hiddenLayers, currentPoints = points) => {
    const sizes = [2, ...hidden, 1]; // Input: 2, Hidden layers, Output: 1
    const weights = [];
    const biases = [];

    for (let i = 1; i < sizes.length; i++) {
      const rows = sizes[i];
      const cols = sizes[i - 1];
      const layerW = [];
      const layerB = [];

      for (let r = 0; r < rows; r++) {
        const row = [];
        for (let c = 0; c < cols; c++) {
          const limit = Math.sqrt(2 / cols);
          row.push((Math.random() * 2 - 1) * limit);
        }
        layerW.push(row);
        layerB.push((Math.random() * 2 - 1) * 0.1);
      }
      weights.push(layerW);
      biases.push(layerB);
    }

    networkRef.current = { weights, biases };
    setNetworkState({ weights, biases });
    setEpochCount(0);
    setLoss(0.5);
    setAccuracy(0.5);

    // Initial render of boundary
    drawBoundary(weights, biases);
  };

  // Feedforward propagation
  // Returns { activations, zs }
  const feedForward = (input, w, b) => {
    let act = [...input];
    const activations = [act];
    const zs = [];

    for (let l = 0; l < w.length; l++) {
      const nextAct = [];
      const nextZs = [];
      const layerW = w[l];
      const layerB = b[l];

      for (let r = 0; r < layerW.length; r++) {
        let z = layerB[r];
        for (let c = 0; c < layerW[r].length; c++) {
          z += layerW[r][c] * act[c];
        }
        nextZs.push(z);
        // Output layer uses sigmoid for binary output classification [0, 1]
        const activationVal = (l === w.length - 1) ? (1 / (1 + Math.exp(-z))) : actFn(z);
        nextAct.push(activationVal);
      }
      zs.push(nextZs);
      activations.push(nextAct);
      act = nextAct;
    }
    return { activations, zs };
  };

  // Perform Backpropagation and Weight updates for 1 sample
  const trainOnSample = (x1, x2, target, w, b) => {
    const { activations, zs } = feedForward([x1, x2], w, b);
    const L = w.length;
    const output = activations[L][0];

    // Compute Output Delta (Cross-Entropy loss derivative w.r.t output z)
    // dL/dz = output - target
    let deltas = [[output - target]];

    // Backpropagate deltas
    for (let l = L - 2; l >= 0; l--) {
      const layerW = w[l + 1];
      const currentZs = zs[l];
      const currentDeltas = [];

      for (let j = 0; j < w[l].length; j++) {
        let error = 0;
        for (let k = 0; k < layerW.length; k++) {
          error += layerW[k][j] * deltas[0][k];
        }
        currentDeltas.push(error * actDeriv(activations[l + 1][j]));
      }
      deltas.unshift(currentDeltas);
    }

    // Update weights and biases using SGD gradients
    for (let l = 0; l < L; l++) {
      const layerW = w[l];
      const layerB = b[l];
      const prevAct = activations[l];
      const currentDeltas = deltas[l];

      for (let j = 0; j < layerW.length; j++) {
        layerB[j] -= learningRate * currentDeltas[j];
        for (let i = 0; i < layerW[j].length; i++) {
          layerW[j][i] -= learningRate * currentDeltas[j] * prevAct[i];
        }
      }
    }
  };

  // Evaluate Network Loss and Accuracy on all data points
  const evaluatePerformance = (w, b) => {
    if (points.length === 0) return { loss: 0, accuracy: 0 };
    
    let totalLoss = 0;
    let correctCount = 0;

    points.forEach((p) => {
      const { activations } = feedForward([p.x, p.y], w, b);
      const predVal = activations[activations.length - 1][0];
      
      // Binary Cross-Entropy loss
      const logEps = 1e-15;
      const predSafe = Math.min(Math.max(predVal, logEps), 1 - logEps);
      const lossVal = -(p.label * Math.log(predSafe) + (1 - p.label) * Math.log(1 - predSafe));
      totalLoss += lossVal;

      const predClass = predVal >= 0.5 ? 1 : 0;
      if (predClass === p.label) {
        correctCount++;
      }
    });

    return {
      loss: totalLoss / points.length,
      accuracy: correctCount / points.length
    };
  };

  // Draw 2D Boundary predictions
  const drawBoundary = (w, b) => {
    const canvas = boundaryCanvasRef.current;
    if (!canvas) return;
    const ctx = canvas.getContext('2d');
    const cw = canvas.width;
    const ch = canvas.height;

    // Draw boundary pixels
    const imgData = ctx.createImageData(cw, ch);
    const data = imgData.data;

    // Grid prediction
    for (let py = 0; py < ch; py++) {
      const normY = -((py / ch) * 2.4 - 1.2); // Map canvas Y to [-1.2, 1.2]
      for (let px = 0; px < cw; px++) {
        const normX = (px / cw) * 2.4 - 1.2; // Map canvas X to [-1.2, 1.2]

        const { activations } = feedForward([normX, normY], w, b);
        const pred = activations[activations.length - 1][0];

        // Map prediction value to color gradient: Blue (pred -> 1) vs Orange (pred -> 0)
        // Blue: rgb(59, 130, 246)
        // Orange: rgb(245, 158, 11)
        const idx = (py * cw + px) * 4;
        
        // Linear interpolation between orange (0) and blue (1)
        data[idx]     = Math.floor(245 + (59 - 245) * pred); // R
        data[idx + 1] = Math.floor(158 + (130 - 158) * pred); // G
        data[idx + 2] = Math.floor(11 + (246 - 11) * pred);    // B
        data[idx + 3] = 135; // Alpha opacity (half translucent so points shine through)
      }
    }
    ctx.putImageData(imgData, 0, 0);

    // Draw data points on top of boundary
    const mapX = (x) => ((x + 1.2) / 2.4) * cw;
    const mapY = (y) => ((-y + 1.2) / 2.4) * ch;

    points.forEach((p) => {
      const cx = mapX(p.x);
      const cy = mapY(p.y);

      ctx.beginPath();
      ctx.arc(cx, cy, 5.5, 0, 2 * Math.PI);
      ctx.fillStyle = p.label === 1 ? '#3b82f6' : '#f59e0b';
      ctx.fill();
      ctx.lineWidth = 1.5;
      ctx.strokeStyle = '#ffffff';
      ctx.stroke();
    });
  };

  // Run Training loop in requestAnimationFrame
  useEffect(() => {
    if (isTraining && points.length > 0) {
      const trainLoop = () => {
        const w = networkRef.current.weights;
        const b = networkRef.current.biases;

        // Perform 5 complete epochs per animation frame for speed
        for (let ep = 0; ep < 5; ep++) {
          // Shuffle points for better SGD convergence
          const shuffled = [...points].sort(() => Math.random() - 0.5);
          shuffled.forEach((p) => {
            trainOnSample(p.x, p.y, p.label, w, b);
          });
        }

        const metrics = evaluatePerformance(w, b);
        setLoss(metrics.loss);
        setAccuracy(metrics.accuracy);
        setEpochCount((prev) => prev + 5);

        // Render boundary
        drawBoundary(w, b);

        animationRef.current = requestAnimationFrame(trainLoop);
      };
      animationRef.current = requestAnimationFrame(trainLoop);
    } else {
      cancelAnimationFrame(animationRef.current);
      // Sync weights to state for the SVG connection diagram when training pauses
      setNetworkState({ ...networkRef.current });
    }

    return () => cancelAnimationFrame(animationRef.current);
  }, [isTraining, points, learningRate, activation]);

  // Topology modification handlers
  const addHiddenLayer = () => {
    if (hiddenLayers.length >= 4) return; // Cap at 4 layers
    updateTopology([...hiddenLayers, 3]);
  };

  const removeHiddenLayer = () => {
    if (hiddenLayers.length <= 1) return; // Keep at least 1
    updateTopology(hiddenLayers.slice(0, -1));
  };

  const adjustNeurons = (layerIdx, diff) => {
    const updated = [...hiddenLayers];
    const newVal = updated[layerIdx] + diff;
    if (newVal >= 1 && newVal <= 8) {
      updated[layerIdx] = newVal;
      updateTopology(updated);
    }
  };

  // SVG Drawing Helpers for Neural Graph
  const drawSVGGraph = () => {
    const w = networkState.weights;
    const sizes = [2, ...hiddenLayers, 1];
    const totalLayers = sizes.length;

    const svgWidth = 400;
    const svgHeight = 240;
    const layerSpacing = svgWidth / (totalLayers - 1 || 1);

    // Calculate node position helper
    const getNodePos = (layerIdx, nodeIdx) => {
      const layerSize = sizes[layerIdx];
      const x = layerIdx * layerSpacing;
      // Vertically center nodes inside SVG
      const spacingY = svgHeight / (layerSize + 1);
      const y = (nodeIdx + 1) * spacingY;
      return { x, y };
    };

    const lines = [];
    const nodes = [];

    // 1. Draw connection lines (Weights)
    for (let l = 0; l < w.length; l++) {
      const layerW = w[l];
      const nextLayerSize = sizes[l + 1];
      const currentLayerSize = sizes[l];

      for (let r = 0; r < nextLayerSize; r++) {
        const targetPos = getNodePos(l + 1, r);
        for (let c = 0; c < currentLayerSize; c++) {
          const sourcePos = getNodePos(l, c);
          const weightVal = layerW[r][c];

          // Connection styling (thickness and opacity based on weight magnitude)
          const weightMag = Math.min(Math.abs(weightVal), 3);
          const thickness = 0.5 + weightMag * 1.5;
          const opacity = 0.15 + (weightMag / 3) * 0.7;
          const strokeColor = weightVal >= 0 ? 'var(--color-neural)' : 'var(--color-warning)'; // blue (+) vs orange (-)

          lines.push(
            <line
              key={`w-${l}-${r}-${c}`}
              x1={sourcePos.x}
              y1={sourcePos.y}
              x2={targetPos.x}
              y2={targetPos.y}
              stroke={strokeColor}
              strokeWidth={thickness}
              strokeOpacity={opacity}
              style={{ transition: 'stroke-width 0.2s, stroke-opacity 0.2s' }}
            />
          );
        }
      }
    }

    // 2. Draw nodes
    for (let l = 0; l < totalLayers; l++) {
      const layerSize = sizes[l];
      let fillGlow = 'var(--text-muted)';
      if (l === 0) fillGlow = 'var(--color-regression)';
      else if (l === totalLayers - 1) fillGlow = 'var(--color-cnn)';
      else fillGlow = 'var(--color-neural)';

      for (let n = 0; n < layerSize; n++) {
        const { x, y } = getNodePos(l, n);
        nodes.push(
          <g key={`n-${l}-${n}`}>
            <circle
              cx={x}
              cy={y}
              r={9}
              fill="#0b0b0e"
              stroke={fillGlow}
              strokeWidth="2.5"
              style={{ filter: `drop-shadow(0 0 3px ${fillGlow}66)` }}
            />
            {/* Inner fill for active state */}
            <circle
              cx={x}
              cy={y}
              r={4}
              fill={fillGlow}
              opacity="0.3"
            />
          </g>
        );
      }
    }

    return (
      <svg width="100%" height="100%" viewBox="-15 -10 430 260">
        {lines}
        {nodes}
      </svg>
    );
  };

  return (
    <div className="animate-fade-in" style={{ display: 'grid', gridTemplateColumns: '1fr', gap: '1.5rem' }}>
      {/* Metrics Header row */}
      <div className="stats-grid">
        <div className="stat-card" style={{ borderColor: 'var(--color-neural-glow)' }}>
          <span className="stat-label">Model Accuracy</span>
          <span className="stat-value" style={{ color: accuracy > 0.85 ? '#6ee7b7' : '#fcd34d' }}>
            {(accuracy * 100).toFixed(1)}%
          </span>
        </div>
        <div className="stat-card">
          <span className="stat-label">Training Loss</span>
          <span className="stat-value" style={{ fontSize: '1.25rem', marginTop: '0.2rem' }}>
            {loss.toFixed(5)}
          </span>
        </div>
        <div className="stat-card">
          <span className="stat-label">Training Epochs</span>
          <span className="stat-value">{epochCount}</span>
        </div>
        <div className="stat-card">
          <span className="stat-label">Network Architecture</span>
          <span className="stat-value" style={{ fontSize: '1.15rem' }}>
            2 → {hiddenLayers.join(' → ')} → 1
          </span>
        </div>
      </div>

      <div className="dashboard-layout">
        {/* Sidebar Controls */}
        <div className="glass-card flex flex-col gap-4 accent-neural">
          <h3 style={{ borderBottom: '1px solid var(--border-color)', paddingBottom: '0.5rem' }}>Configurator</h3>

          {/* Dataset presets selector */}
          <div>
            <span className="form-label" style={{ marginBottom: '0.4rem' }}>Dataset Pattern</span>
            <div style={{ display: 'flex', flexWrap: 'wrap', gap: '0.4rem' }}>
              {['gaussian', 'xor', 'circle', 'spiral'].map((ds) => (
                <button
                  key={ds}
                  className={`case-study-btn ${activeDataset === ds ? 'active-neural' : ''}`}
                  onClick={() => setActiveDataset(ds)}
                  style={{ textTransform: 'capitalize' }}
                >
                  {ds}
                </button>
              ))}
            </div>
          </div>

          {/* Architecture editor */}
          <div>
            <span className="form-label" style={{ marginBottom: '0.5rem' }}>Hidden Layers ({hiddenLayers.length})</span>
            <div style={{ display: 'flex', gap: '0.5rem', marginBottom: '0.8rem' }}>
              <button className="btn btn-secondary w-full" onClick={addHiddenLayer} disabled={hiddenLayers.length >= 4}>
                <Plus size={14} /> Add Layer
              </button>
              <button className="btn btn-secondary w-full" onClick={removeHiddenLayer} disabled={hiddenLayers.length <= 1}>
                <Minus size={14} /> Remove
              </button>
            </div>
            
            {/* Neurons adjustments per layer */}
            <div style={{ display: 'flex', flexDirection: 'column', gap: '0.4rem' }}>
              {hiddenLayers.map((count, idx) => (
                <div key={idx} className="flex justify-between items-center" style={{ background: 'rgba(255,255,255,0.02)', padding: '0.4rem 0.6rem', borderRadius: '6px', border: '1px solid var(--border-color)' }}>
                  <span style={{ fontSize: '0.75rem', color: 'var(--text-secondary)' }}>Hidden Layer {idx + 1}</span>
                  <div className="flex items-center gap-2">
                    <button style={{ background: 'rgba(255,255,255,0.05)', border: 'none', width: '22px', height: '22px', borderRadius: '4px', cursor: 'pointer', color: '#fff', display: 'flex', alignItems: 'center', justifyContent: 'center' }} onClick={() => adjustNeurons(idx, -1)}>
                      -
                    </button>
                    <span className="font-mono" style={{ fontSize: '0.8rem', width: '15px', textAlign: 'center' }}>{count}</span>
                    <button style={{ background: 'rgba(255,255,255,0.05)', border: 'none', width: '22px', height: '22px', borderRadius: '4px', cursor: 'pointer', color: '#fff', display: 'flex', alignItems: 'center', justifyContent: 'center' }} onClick={() => adjustNeurons(idx, 1)}>
                      +
                    </button>
                  </div>
                </div>
              ))}
            </div>
          </div>

          {/* Activation Function selector */}
          <div className="form-group">
            <span className="form-label">Activation Function</span>
            <select value={activation} onChange={(e) => { setActivation(e.target.value); initializeNetwork(hiddenLayers); }}>
              <option value="tanh">Tanh (Hyperbolic Tangent)</option>
              <option value="relu">ReLU (Rectified Linear Unit)</option>
              <option value="sigmoid">Sigmoid (Logistic)</option>
              <option value="linear">Linear (No Activation)</option>
            </select>
          </div>

          {/* Learning Rate Slider */}
          <div className="form-group">
            <span className="form-label">
              Learning Rate (η) <span className="form-value">{learningRate}</span>
            </span>
            <input
              type="range"
              className="slider-neural"
              min="0.01"
              max="0.25"
              step="0.01"
              value={learningRate}
              onChange={(e) => setLearningRate(parseFloat(e.target.value))}
            />
          </div>

          {/* Training Control actions */}
          <div style={{ display: 'flex', flexDirection: 'column', gap: '0.5rem', marginTop: '0.5rem' }}>
            <button
              className="btn btn-neural"
              onClick={() => setIsTraining(!isTraining)}
            >
              {isTraining ? <Pause size={16} /> : <Play size={16} />}
              {isTraining ? 'Pause Training' : 'Train Model'}
            </button>
            <div style={{ display: 'grid', gridTemplateColumns: '1fr', gap: '0.5rem' }}>
              <button className="btn btn-secondary" onClick={() => initializeNetwork()}>
                <RotateCcw size={14} /> Re-Initialize weights
              </button>
            </div>
          </div>
        </div>

        {/* Workspace Visualizer Columns */}
        <div style={{ display: 'grid', gridTemplateColumns: '1fr', gap: '1.5rem' }}>
          {/* Decision boundary canvas */}
          <div className="glass-card flex flex-col gap-3" style={{ padding: '1.25rem' }}>
            <h4 style={{ fontSize: '0.9rem', color: 'var(--text-secondary)', borderBottom: '1px solid var(--border-color)', paddingBottom: '0.4rem' }}>
              2D Decision Boundary Space
            </h4>
            <div style={{ display: 'flex', justifyContent: 'center' }}>
              <div style={{ position: 'relative', width: '100%', maxWidth: '380px', aspectRatio: '1' }}>
                <canvas
                  ref={boundaryCanvasRef}
                  width={340}
                  height={340}
                  style={{
                    width: '100%',
                    height: '100%',
                    borderRadius: '12px',
                    border: '1px solid var(--border-color)',
                    display: 'block'
                  }}
                />
              </div>
            </div>
            <div className="flex justify-between" style={{ fontSize: '0.75rem', color: 'var(--text-muted)' }}>
              <span style={{ display: 'flex', alignItems: 'center', gap: '0.25rem' }}>
                <span style={{ width: '8px', height: '8px', borderRadius: '50%', backgroundColor: 'var(--color-neural)' }} /> Output Class 1
              </span>
              <span style={{ display: 'flex', alignItems: 'center', gap: '0.25rem' }}>
                <span style={{ width: '8px', height: '8px', borderRadius: '50%', backgroundColor: 'var(--color-warning)' }} /> Output Class 0
              </span>
            </div>
          </div>

          {/* Network Graph Visual SVG */}
          <div className="glass-card flex flex-col gap-3" style={{ padding: '1.25rem' }}>
            <h4 style={{ fontSize: '0.9rem', color: 'var(--text-secondary)', borderBottom: '1px solid var(--border-color)', paddingBottom: '0.4rem' }}>
              Neural Network Graph Model
            </h4>
            <div style={{ height: '240px', width: '100%' }}>
              {drawSVGGraph()}
            </div>
            <div style={{ fontSize: '0.7rem', color: 'var(--text-muted)', display: 'flex', gap: '0.75rem', justifyContent: 'center' }}>
              <span style={{ color: 'var(--color-neural)' }}>── Positive Weight (+)</span>
              <span style={{ color: 'var(--color-warning)' }}>── Negative Weight (-)</span>
              <span>Thickness = Weight Magnitude</span>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}
