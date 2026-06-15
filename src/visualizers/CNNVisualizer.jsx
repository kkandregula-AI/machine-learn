import React, { useState, useEffect, useRef } from 'react';
import { Play, Pause, SkipForward, RotateCcw, Image, Cpu } from 'lucide-react';

// Preset Images (10x10 grid, grayscale values 0-255)
const IMAGE_PRESETS = {
  digit7: [
    [0,   0,   0,   0,   0,   0,   0,   0,   0,   0],
    [0, 240, 240, 240, 240, 240, 240, 240,   0,   0],
    [0,   0,   0,   0,   0,   0,  80, 240,   0,   0],
    [0,   0,   0,   0,   0,   0, 120, 240,   0,   0],
    [0,   0,   0,   0,   0, 160, 240,  80,   0,   0],
    [0,   0,   0,   0, 200, 240,  40,   0,   0,   0],
    [0,   0,   0, 220, 240,  60,   0,   0,   0,   0],
    [0,   0,   0, 240, 140,   0,   0,   0,   0,   0],
    [0,   0,   0, 240,  80,   0,   0,   0,   0,   0],
    [0,   0,   0,   0,   0,   0,   0,   0,   0,   0]
  ],
  box: [
    [0,   0,   0,   0,   0,   0,   0,   0,   0,   0],
    [0, 220, 220, 220, 220, 220, 220, 220, 220,   0],
    [0, 220,   0,   0,   0,   0,   0,   0, 220,   0],
    [0, 220,   0,   0,   0,   0,   0,   0, 220,   0],
    [0, 220,   0,   0,   0,   0,   0,   0, 220,   0],
    [0, 220,   0,   0,   0,   0,   0,   0, 220,   0],
    [0, 220,   0,   0,   0,   0,   0,   0, 220,   0],
    [0, 220,   0,   0,   0,   0,   0,   0, 220,   0],
    [0, 220, 220, 220, 220, 220, 220, 220, 220,   0],
    [0,   0,   0,   0,   0,   0,   0,   0,   0,   0]
  ],
  cross: [
    [240,   0,   0,   0,   0,   0,   0,   0,   0, 240],
    [  0, 240,   0,   0,   0,   0,   0,   0, 240,   0],
    [  0,   0, 240,   0,   0,   0,   0, 240,   0,   0],
    [  0,   0,   0, 240,   0,   0, 240,   0,   0,   0],
    [  0,   0,   0,   0, 240, 240,   0,   0,   0,   0],
    [  0,   0,   0,   0, 240, 240,   0,   0,   0,   0],
    [  0,   0,   0, 240,   0,   0, 240,   0,   0,   0],
    [  0,   0, 240,   0,   0,   0,   0, 240,   0,   0],
    [  0, 240,   0,   0,   0,   0,   0,   0, 240,   0],
    [240,   0,   0,   0,   0,   0,   0,   0,   0, 240]
  ]
};

// Preset Kernels (3x3 grid)
const KERNELS = {
  sobelV: {
    name: 'Sobel Vertical (Edges)',
    matrix: [
      [-1, 0, 1],
      [-2, 0, 2],
      [-1, 0, 1]
    ]
  },
  sobelH: {
    name: 'Sobel Horizontal (Edges)',
    matrix: [
      [-1, -2, -1],
      [ 0,  0,  0],
      [ 1,  2,  1]
    ]
  },
  blur: {
    name: 'Gaussian Blur (Smoothing)',
    matrix: [
      [0.0625, 0.125, 0.0625],
      [0.125,  0.25,  0.125],
      [0.0625, 0.125, 0.0625]
    ]
  },
  sharpen: {
    name: 'Sharpen (Details)',
    matrix: [
      [ 0, -1,  0],
      [-1,  5, -1],
      [ 0, -1,  0]
    ]
  },
  ridge: {
    name: 'Ridge Detection (Borders)',
    matrix: [
      [-1, -1, -1],
      [-1,  8, -1],
      [-1, -1, -1]
    ]
  }
};

export default function CNNVisualizer() {
  const [activeImage, setActiveImage] = useState('digit7');
  const [activeKernel, setActiveKernel] = useState('sobelV');
  const [isPlaying, setIsPlaying] = useState(false);
  const [slideSpeed, setSlideSpeed] = useState(800); // ms per step

  // Slider indices: input is 10x10, kernel is 3x3, output is 8x8 (no padding)
  // X: 0 to 7, Y: 0 to 7
  const [kernelPos, setKernelPos] = useState({ x: 0, y: 0 });
  const [outputGrid, setOutputGrid] = useState(
    Array(8).fill(null).map(() => Array(8).fill(null))
  );

  const timerRef = useRef(null);

  const currentImage = IMAGE_PRESETS[activeImage];
  const currentKernel = KERNELS[activeKernel].matrix;

  // Initialize/Reset output feature map
  const resetOutput = () => {
    setOutputGrid(Array(8).fill(null).map(() => Array(8).fill(null)));
    setKernelPos({ x: 0, y: 0 });
    setIsPlaying(false);
  };

  useEffect(() => {
    resetOutput();
  }, [activeImage, activeKernel]);

  // Compute convolution at specific position (cx, cy)
  const computeConvolutionVal = (cx, cy) => {
    let sum = 0;
    for (let i = 0; i < 3; i++) {
      for (let j = 0; j < 3; j++) {
        const pixelVal = currentImage[cy + i][cx + j];
        const kernelVal = currentKernel[i][j];
        sum += pixelVal * kernelVal;
      }
    }
    // Round to 1 decimal place
    return Math.round(sum * 10) / 10;
  };

  // Perform single step of sliding window
  const stepSlide = () => {
    setKernelPos((prev) => {
      let nextX = prev.x + 1;
      let nextY = prev.y;

      if (nextX > 7) {
        nextX = 0;
        nextY += 1;
      }

      if (nextY > 7) {
        // Slide completed! Pause
        setIsPlaying(false);
        return { x: 0, y: 0 };
      }

      // Update output grid for this completed cell
      const convVal = computeConvolutionVal(prev.x, prev.y);
      setOutputGrid((grid) => {
        const newGrid = grid.map((row) => [...row]);
        newGrid[prev.y][prev.x] = convVal;
        return newGrid;
      });

      return { x: nextX, y: nextY };
    });
  };

  // Auto-slide loop
  useEffect(() => {
    if (isPlaying) {
      timerRef.current = setInterval(() => {
        stepSlide();
      }, slideSpeed);
    } else {
      clearInterval(timerRef.current);
    }
    return () => clearInterval(timerRef.current);
  }, [isPlaying, slideSpeed, activeImage, activeKernel]);

  // Perform full convolution instantly (Compute entire feature map)
  const computeAll = () => {
    const newGrid = Array(8).fill(null).map((_, y) => {
      return Array(8).fill(null).map((_, x) => computeConvolutionVal(x, y));
    });
    setOutputGrid(newGrid);
    setKernelPos({ x: 7, y: 7 }); // Move to end
    setIsPlaying(false);
  };

  // Math display values for 3x3 sliding region
  const getCalculationSummary = () => {
    const cx = kernelPos.x;
    const cy = kernelPos.y;
    const steps = [];
    let sum = 0;

    for (let i = 0; i < 3; i++) {
      for (let j = 0; j < 3; j++) {
        const pixelVal = currentImage[cy + i][cx + j];
        const kernelVal = currentKernel[i][j];
        const prod = pixelVal * kernelVal;
        sum += prod;
        steps.push({
          px: pixelVal,
          k: kernelVal,
          prod: Math.round(prod * 100) / 100
        });
      }
    }

    return { steps, sum: Math.round(sum * 10) / 10 };
  };

  const mathData = getCalculationSummary();

  // Helper to color output feature map cells dynamically
  const getOutputCellColor = (val) => {
    if (val === null) return 'rgba(255, 255, 255, 0.02)';
    
    // Normalize color output for display (e.g. positive values blue, negative values pink)
    // Blur values go from 0 to 255. Sobel edge values can range from -1000 to +1000.
    const maxVal = activeKernel === 'blur' ? 255 : 510;
    const ratio = Math.min(Math.abs(val) / maxVal, 1.0);

    if (val >= 0) {
      return `rgba(59, 130, 246, ${0.1 + ratio * 0.9})`; // glowing blue for positive features
    } else {
      return `rgba(236, 72, 153, ${0.1 + ratio * 0.9})`; // glowing pink for negative features
    }
  };

  return (
    <div className="animate-fade-in" style={{ display: 'grid', gridTemplateColumns: '1fr', gap: '1.5rem' }}>
      {/* Stats header bar */}
      <div className="stats-grid">
        <div className="stat-card" style={{ borderColor: 'var(--color-cnn-glow)' }}>
          <span className="stat-label">Active Kernel</span>
          <span className="stat-value" style={{ fontSize: '1.05rem', color: 'var(--color-cnn)' }}>
            {KERNELS[activeKernel].name.split(' ')[0]}
          </span>
        </div>
        <div className="stat-card">
          <span className="stat-label">Input Size</span>
          <span className="stat-value">10 x 10</span>
        </div>
        <div className="stat-card">
          <span className="stat-label">Kernel Size</span>
          <span className="stat-value">3 x 3</span>
        </div>
        <div className="stat-card">
          <span className="stat-label">Output Size</span>
          <span className="stat-value">8 x 8</span>
        </div>
      </div>

      <div className="dashboard-layout">
        {/* CNN settings Sidebar */}
        <div className="glass-card flex flex-col gap-4 accent-cnn">
          <h3 style={{ borderBottom: '1px solid var(--border-color)', paddingBottom: '0.5rem' }}>Settings</h3>

          {/* Sample Image presets */}
          <div>
            <span className="form-label" style={{ marginBottom: '0.4rem' }}>
              <Image size={14} /> Input Image
            </span>
            <div style={{ display: 'flex', flexDirection: 'column', gap: '0.4rem' }}>
              <button 
                className={`case-study-btn w-full ${activeImage === 'digit7' ? 'active-cnn' : ''}`}
                onClick={() => setActiveImage('digit7')}
              >
                MNIST Digit 7
              </button>
              <button 
                className={`case-study-btn w-full ${activeImage === 'box' ? 'active-cnn' : ''}`}
                onClick={() => setActiveImage('box')}
              >
                Hollow Box
              </button>
              <button 
                className={`case-study-btn w-full ${activeImage === 'cross' ? 'active-cnn' : ''}`}
                onClick={() => setActiveImage('cross')}
              >
                Cross Pattern
              </button>
            </div>
          </div>

          {/* Kernel Filter Presets */}
          <div className="form-group">
            <span className="form-label">
              <Cpu size={14} /> Convolution Kernel
            </span>
            <select value={activeKernel} onChange={(e) => setActiveKernel(e.target.value)}>
              <option value="sobelV">Sobel Vertical (Detects vertical edges)</option>
              <option value="sobelH">Sobel Horizontal (Detects horizontal edges)</option>
              <option value="blur">Gaussian Blur (Averages neighboring pixels)</option>
              <option value="sharpen">Sharpen Filter (Accentuates details)</option>
              <option value="ridge">Ridge Detection (Highlights bounds)</option>
            </select>
          </div>

          {/* Speed control slider */}
          <div className="form-group">
            <span className="form-label">
              Slide Interval <span className="form-value">{slideSpeed}ms</span>
            </span>
            <input 
              type="range" 
              className="slider-cnn"
              min="200" 
              max="1500" 
              step="100"
              value={slideSpeed} 
              onChange={(e) => setSlideSpeed(parseInt(e.target.value))}
            />
          </div>

          {/* Sliding Controls */}
          <div style={{ display: 'flex', flexDirection: 'column', gap: '0.5rem', marginTop: '0.5rem' }}>
            <button 
              className="btn btn-cnn"
              onClick={() => setIsPlaying(!isPlaying)}
            >
              {isPlaying ? <Pause size={16} /> : <Play size={16} />}
              {isPlaying ? 'Pause Sliding' : 'Play Convolution'}
            </button>
            <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '0.5rem' }}>
              <button className="btn btn-secondary animate-fade-in" onClick={stepSlide} disabled={isPlaying}>
                <SkipForward size={14} /> Step
              </button>
              <button className="btn btn-secondary" onClick={resetOutput}>
                <RotateCcw size={14} /> Reset
              </button>
            </div>
            <button className="btn btn-secondary w-full" onClick={computeAll}>
              Compute Full Map
            </button>
          </div>
        </div>

        {/* Interactive Workspace Grids */}
        <div style={{ display: 'grid', gridTemplateColumns: '1fr', gap: '1.5rem' }}>
          
          {/* Grid Layout of Input vs Kernel vs Output */}
          <div className="glass-card" style={{ padding: '1.25rem' }}>
            <div style={{ display: 'flex', flexWrap: 'wrap', gap: '1.5rem', justifyContent: 'space-around', alignItems: 'center' }}>
              
              {/* Input Image 10x10 */}
              <div className="flex flex-col items-center gap-2">
                <span className="form-label">Input Image (10x10)</span>
                <div style={{
                  display: 'grid',
                  gridTemplateColumns: 'repeat(10, 22px)',
                  gridTemplateRows: 'repeat(10, 22px)',
                  gap: '1px',
                  background: 'rgba(255,255,255,0.05)',
                  padding: '4px',
                  borderRadius: '8px',
                  border: '1px solid var(--border-color)'
                }}>
                  {currentImage.map((row, y) => 
                    row.map((val, x) => {
                      // Check if pixel is inside currently sliding 3x3 window
                      const isInsideWindow = 
                        x >= kernelPos.x && x < kernelPos.x + 3 &&
                        y >= kernelPos.y && y < kernelPos.y + 3;

                      return (
                        <div
                          key={`in-${y}-${x}`}
                          style={{
                            backgroundColor: `rgb(${val}, ${val}, ${val})`,
                            borderRadius: '2px',
                            border: isInsideWindow ? '1.5px solid var(--color-cnn)' : '1px solid transparent',
                            boxShadow: isInsideWindow ? '0 0 4px var(--color-cnn)' : 'none',
                            transition: 'border 0.1s, box-shadow 0.1s'
                          }}
                          title={`Val: ${val}`}
                        />
                      );
                    })
                  )}
                </div>
              </div>

              {/* Convolution sign */}
              <div className="font-mono" style={{ fontSize: '1.5rem', color: 'var(--text-muted)' }}>*</div>

              {/* 3x3 Kernel Grid */}
              <div className="flex flex-col items-center gap-2">
                <span className="form-label">3x3 Kernel Filter</span>
                <div style={{
                  display: 'grid',
                  gridTemplateColumns: 'repeat(3, 36px)',
                  gridTemplateRows: 'repeat(3, 36px)',
                  gap: '2px',
                  background: 'rgba(255,255,255,0.05)',
                  padding: '4px',
                  borderRadius: '8px',
                  border: '1px solid var(--border-color)'
                }}>
                  {currentKernel.map((row, y) =>
                    row.map((val, x) => (
                      <div
                        key={`k-${y}-${x}`}
                        className="font-mono"
                        style={{
                          backgroundColor: '#0b0b0e',
                          color: 'var(--color-cnn)',
                          display: 'flex',
                          alignItems: 'center',
                          justifyContent: 'center',
                          fontSize: '0.75rem',
                          fontWeight: 'bold',
                          borderRadius: '4px',
                          border: '1px solid rgba(255,255,255,0.05)'
                        }}
                      >
                        {val}
                      </div>
                    ))
                  )}
                </div>
              </div>

              {/* Equals sign */}
              <div className="font-mono" style={{ fontSize: '1.5rem', color: 'var(--text-muted)' }}>=</div>

              {/* Output Feature Map 8x8 */}
              <div className="flex flex-col items-center gap-2">
                <span className="form-label">Feature Map (8x8)</span>
                <div style={{
                  display: 'grid',
                  gridTemplateColumns: 'repeat(8, 25px)',
                  gridTemplateRows: 'repeat(8, 25px)',
                  gap: '1px',
                  background: 'rgba(255,255,255,0.05)',
                  padding: '4px',
                  borderRadius: '8px',
                  border: '1px solid var(--border-color)'
                }}>
                  {outputGrid.map((row, y) =>
                    row.map((val, x) => {
                      const isActiveCell = kernelPos.x === x && kernelPos.y === y;
                      return (
                        <div
                          key={`out-${y}-${x}`}
                          className="font-mono"
                          style={{
                            backgroundColor: isActiveCell ? 'rgba(236, 72, 153, 0.2)' : getOutputCellColor(val),
                            border: isActiveCell ? '1.5px solid var(--color-cnn)' : '1px solid rgba(255,255,255,0.02)',
                            boxShadow: isActiveCell ? '0 0 6px var(--color-cnn)' : 'none',
                            display: 'flex',
                            alignItems: 'center',
                            justifyContent: 'center',
                            fontSize: '0.6rem',
                            color: '#fff',
                            borderRadius: '3px',
                            cursor: 'help'
                          }}
                          title={`Val: ${val !== null ? val : 'Uncalculated'}`}
                        >
                          {val !== null ? Math.round(val) : ''}
                        </div>
                      );
                    })
                  )}
                </div>
              </div>

            </div>
          </div>

          {/* Math calculation Panel */}
          <div className="glass-card" style={{ padding: '1.25rem' }}>
            <h4 style={{ fontSize: '0.9rem', color: 'var(--text-secondary)', marginBottom: '0.75rem', borderBottom: '1px solid var(--border-color)', paddingBottom: '0.4rem' }}>
              Sliding Window Math Detail (Output[{kernelPos.y}, {kernelPos.x}])
            </h4>
            <div style={{ display: 'grid', gridTemplateColumns: '1fr', gap: '1rem' }}>
              {/* Equation grid */}
              <div style={{
                display: 'grid',
                gridTemplateColumns: 'repeat(3, 1fr)',
                gap: '0.5rem',
                fontSize: '0.8rem'
              }}>
                {mathData.steps.map((step, idx) => (
                  <div key={idx} style={{ background: 'rgba(0,0,0,0.2)', padding: '0.5rem', borderRadius: '6px', border: '1px solid rgba(255,255,255,0.02)', textAlign: 'center' }}>
                    <div style={{ color: 'var(--text-muted)', fontSize: '0.7rem', marginBottom: '0.2rem' }}>Cell {idx + 1}</div>
                    <div className="font-mono">
                      {step.px} × {step.k} = <strong style={{ color: 'var(--color-cnn)' }}>{step.prod}</strong>
                    </div>
                  </div>
                ))}
              </div>

              {/* Total sum display */}
              <div style={{
                display: 'flex',
                justifyContent: 'space-between',
                alignItems: 'center',
                background: 'rgba(236, 72, 153, 0.06)',
                padding: '0.75rem 1.25rem',
                borderRadius: '8px',
                border: '1px solid rgba(236, 72, 153, 0.15)'
              }}>
                <span style={{ fontSize: '0.85rem', fontWeight: '500' }}>Sum of products (Feature Value):</span>
                <span className="font-mono" style={{ fontSize: '1.2rem', fontWeight: 'bold', color: 'var(--color-cnn)' }}>
                  {mathData.sum}
                </span>
              </div>
            </div>
          </div>

        </div>
      </div>
    </div>
  );
}
