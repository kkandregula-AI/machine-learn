import React, { useState, useEffect } from 'react';
import { Sliders, Activity, GitFork } from 'lucide-react';

// Pre-generated Iris-like dataset (Sepal Length on X [4.5 - 7.5], Petal Width on Y [0.1 - 2.5])
const IRIS_DATA = [
  // Class 0: Versicolor (Orange)
  { x: 5.0, y: 1.0, label: 0 },
  { x: 5.5, y: 1.2, label: 0 },
  { x: 4.9, y: 1.0, label: 0 },
  { x: 6.0, y: 1.5, label: 0 },
  { x: 5.7, y: 1.3, label: 0 },
  { x: 5.5, y: 1.3, label: 0 },
  { x: 5.6, y: 1.1, label: 0 },
  { x: 6.2, y: 1.5, label: 0 },
  { x: 5.0, y: 1.3, label: 0 },
  { x: 5.9, y: 1.5, label: 0 },
  { x: 5.2, y: 1.4, label: 0 },
  { x: 5.0, y: 0.8, label: 0 },
  { x: 6.4, y: 1.5, label: 0 },
  { x: 5.7, y: 1.5, label: 0 },
  { x: 5.4, y: 1.5, label: 0 },
  { x: 6.7, y: 1.5, label: 0 },
  { x: 6.3, y: 1.6, label: 0 },
  { x: 5.6, y: 1.3, label: 0 },
  { x: 5.5, y: 1.0, label: 0 },
  { x: 5.5, y: 1.4, label: 0 },

  // Class 1: Virginica (Blue)
  { x: 6.3, y: 1.8, label: 1 },
  { x: 6.5, y: 2.1, label: 1 },
  { x: 7.6, y: 2.4, label: 1 },
  { x: 4.9, y: 1.7, label: 1 }, // outlier
  { x: 7.3, y: 1.9, label: 1 },
  { x: 6.7, y: 2.0, label: 1 },
  { x: 7.2, y: 2.5, label: 1 },
  { x: 6.5, y: 1.8, label: 1 },
  { x: 6.4, y: 2.0, label: 1 },
  { x: 6.8, y: 2.2, label: 1 },
  { x: 5.7, y: 2.0, label: 1 },
  { x: 5.8, y: 1.9, label: 1 },
  { x: 6.4, y: 1.8, label: 1 },
  { x: 6.5, y: 1.5, label: 1 }, // boundary
  { x: 7.7, y: 2.0, label: 1 },
  { x: 6.3, y: 2.3, label: 1 },
  { x: 6.9, y: 2.3, label: 1 },
  { x: 6.7, y: 2.2, label: 1 },
  { x: 6.9, y: 2.3, label: 1 },
  { x: 5.8, y: 2.2, label: 1 }
];

export default function DecisionTreeVisualizer() {
  const [metric, setMetric] = useState('gini'); // 'gini' or 'entropy'
  
  // Interactive Threshold Sliders
  const [split1, setSplit1] = useState(1.6); // Split 1: Petal Width (Y-axis)
  const [split2, setSplit2] = useState(5.8); // Split 2 (Left): Sepal Length (X-axis)
  const [split3, setSplit3] = useState(6.2); // Split 3 (Right): Sepal Length (X-axis)

  const [treeData, setTreeData] = useState(null);
  const [accuracy, setAccuracy] = useState(0);

  // Re-calculate tree nodes and metrics when thresholds update
  useEffect(() => {
    // 1. Root Node calculation
    const rootPoints = IRIS_DATA;
    const rootMetrics = computeNodeMetrics(rootPoints);

    // 2. Left branch: Petal Width <= split1
    const leftPoints = rootPoints.filter((p) => p.y <= split1);
    const leftMetrics = computeNodeMetrics(leftPoints);

    // 3. Right branch: Petal Width > split1
    const rightPoints = rootPoints.filter((p) => p.y > split1);
    const rightMetrics = computeNodeMetrics(rightPoints);

    // 4. Left-Left leaf: Petal Width <= split1 AND Sepal Length <= split2
    const leafLL = leftPoints.filter((p) => p.x <= split2);
    const metricsLL = computeNodeMetrics(leafLL);

    // 5. Left-Right leaf: Petal Width <= split1 AND Sepal Length > split2
    const leafLR = leftPoints.filter((p) => p.x > split2);
    const metricsLR = computeNodeMetrics(leafLR);

    // 6. Right-Left leaf: Petal Width > split1 AND Sepal Length <= split3
    const leafRL = rightPoints.filter((p) => p.x <= split3);
    const metricsRL = computeNodeMetrics(leafRL);

    // 7. Right-Right leaf: Petal Width > split1 AND Sepal Length > split3
    const leafRR = rightPoints.filter((p) => p.x > split3);
    const metricsRR = computeNodeMetrics(leafRR);

    // Calculate overall accuracy
    let correct = 0;
    leafLL.forEach((p) => { if (p.label === metricsLL.majorityClass) correct++; });
    leafLR.forEach((p) => { if (p.label === metricsLR.majorityClass) correct++; });
    leafRL.forEach((p) => { if (p.label === metricsRL.majorityClass) correct++; });
    leafRR.forEach((p) => { if (p.label === metricsRR.majorityClass) correct++; });
    setAccuracy(correct / IRIS_DATA.length);

    // Construct node hierarchy for SVG rendering
    setTreeData({
      root: { name: `Petal Width <= ${split1}`, count: rootPoints.length, ...rootMetrics },
      left: { name: `Sepal Length <= ${split2}`, count: leftPoints.length, ...leftMetrics },
      right: { name: `Sepal Length <= ${split3}`, count: rightPoints.length, ...rightMetrics },
      leafLL: { name: 'Leaf LL', count: leafLL.length, ...metricsLL },
      leafLR: { name: 'Leaf LR', count: leafLR.length, ...metricsLR },
      leafRL: { name: 'Leaf RL', count: leafRL.length, ...metricsRL },
      leafRR: { name: 'Leaf RR', count: leafRR.length, ...metricsRR }
    });

  }, [split1, split2, split3, metric]);

  // Compute Gini Impurity, Entropy, Count classes, and majority class
  const computeNodeMetrics = (nodePoints) => {
    const total = nodePoints.length;
    if (total === 0) {
      return { impurity: 0, class0: 0, class1: 0, majorityClass: 0 };
    }

    const class0 = nodePoints.filter((p) => p.label === 0).length;
    const class1 = total - class0;
    const p0 = class0 / total;
    const p1 = class1 / total;

    let impurity = 0;
    if (metric === 'gini') {
      impurity = 1 - (Math.pow(p0, 2) + Math.pow(p1, 2));
    } else {
      // Entropy
      const logEps = 1e-15;
      const log2 = (v) => Math.log2(v || logEps);
      impurity = -(p0 * log2(p0) + p1 * log2(p1));
      if (impurity < 0) impurity = 0;
    }

    const majorityClass = class1 > class0 ? 1 : 0;

    return {
      impurity: Math.round(impurity * 1000) / 1000,
      class0,
      class1,
      majorityClass
    };
  };

  // SVG Node Tree diagram generator
  const drawTreeSVG = () => {
    if (!treeData) return null;

    const width = 500;
    const height = 300;

    // Node locations
    const nodes = {
      root: { x: 250, y: 35, data: treeData.root, splitStr: `Petal Width <= ${split1}` },
      left: { x: 120, y: 125, data: treeData.left, splitStr: `Sepal Length <= ${split2}` },
      right: { x: 380, y: 125, data: treeData.right, splitStr: `Sepal Length <= ${split3}` },
      leafLL: { x: 60, y: 240, data: treeData.leafLL, isLeaf: true },
      leafLR: { x: 180, y: 240, data: treeData.leafLR, isLeaf: true },
      leafRL: { x: 320, y: 240, data: treeData.leafRL, isLeaf: true },
      leafRR: { x: 440, y: 240, data: treeData.leafRR, isLeaf: true }
    };

    const lines = [
      // root -> left & right
      { x1: nodes.root.x, y1: nodes.root.y + 20, x2: nodes.left.x, y2: nodes.left.y - 20 },
      { x1: nodes.root.x, y1: nodes.root.y + 20, x2: nodes.right.x, y2: nodes.right.y - 20 },
      // left -> leaves
      { x1: nodes.left.x, y1: nodes.left.y + 20, x2: nodes.leafLL.x, y2: nodes.leafLL.y - 20 },
      { x1: nodes.left.x, y1: nodes.left.y + 20, x2: nodes.leafLR.x, y2: nodes.leafLR.y - 20 },
      // right -> leaves
      { x1: nodes.right.x, y1: nodes.right.y + 20, x2: nodes.leafRL.x, y2: nodes.leafRL.y - 20 },
      { x1: nodes.right.x, y1: nodes.right.y + 20, x2: nodes.leafRR.x, y2: nodes.leafRR.y - 20 }
    ];

    return (
      <svg width="100%" height="100%" viewBox="0 0 500 300" style={{ background: '#0b0b0e', borderRadius: '12px', border: '1px solid var(--border-color)' }}>
        {/* Connection lines */}
        {lines.map((line, idx) => (
          <line
            key={`l-${idx}`}
            x1={line.x1}
            y1={line.y1}
            x2={line.x2}
            y2={line.y2}
            stroke="rgba(255,255,255,0.08)"
            strokeWidth="2.5"
          />
        ))}

        {/* Nodes */}
        {Object.entries(nodes).map(([key, n]) => {
          const isLeaf = n.isLeaf;
          const classColor = n.data.count === 0 ? 'var(--text-muted)' : (n.data.majorityClass === 1 ? 'var(--color-neural)' : 'var(--color-warning)');
          
          return (
            <g key={key}>
              {/* Outer box */}
              <rect
                x={n.x - 50}
                y={n.y - 20}
                width={100}
                height={42}
                rx={6}
                fill={isLeaf ? 'rgba(255,255,255,0.02)' : 'rgba(255,255,255,0.05)'}
                stroke={classColor}
                strokeWidth={isLeaf ? '1.5' : '2.5'}
                style={{ filter: `drop-shadow(0 2px 4px rgba(0,0,0,0.5))` }}
              />

              {/* Node texts */}
              <text
                x={n.x}
                y={n.y - 6}
                textAnchor="middle"
                fill="#ffffff"
                fontSize="7.5px"
                fontWeight="600"
              >
                {isLeaf ? (n.data.majorityClass === 1 ? 'Virginica' : 'Versicolor') : n.splitStr}
              </text>
              <text
                x={n.x}
                y={n.y + 6}
                textAnchor="middle"
                fill="var(--text-secondary)"
                fontSize="6.5px"
              >
                Samples: {n.data.count}
              </text>
              <text
                x={n.x}
                y={n.y + 16}
                textAnchor="middle"
                fill="var(--text-muted)"
                fontSize="6px"
                fontFamily="var(--font-mono)"
              >
                {metric.toUpperCase()}: {n.data.impurity}
              </text>
            </g>
          );
        })}
      </svg>
    );
  };

  // Helper to draw 2D decision partitions on scatter plot SVG
  const drawScatterPlot = () => {
    // Sepal Length: X goes from 4.5 to 7.8
    // Petal Width: Y goes from 0.5 to 2.8
    const mapX = (x) => ((x - 4.5) / 3.3) * 100; // percent
    const mapY = (y) => (100 - ((y - 0.5) / 2.3) * 100); // percent

    return (
      <div style={{ position: 'relative', width: '100%', aspectRatio: '1.4', background: '#0b0b0e', borderRadius: '12px', border: '1px solid var(--border-color)', overflow: 'hidden' }}>
        
        {/* Background decision regions (colored using divs absolute positioning) */}
        {/* Region 1: Petal Width <= split1 (Versicolor region unless overridden by Split 2) */}
        <div style={{
          position: 'absolute',
          left: '0%',
          right: '100%',
          top: `${mapY(split1)}%`,
          bottom: '0%',
          background: 'rgba(245, 158, 11, 0.04)',
          borderTop: '1px dashed rgba(255,255,255,0.15)'
        }} />

        {/* Region LL: Petal Width <= split1 AND Sepal Length <= split2 */}
        <div style={{
          position: 'absolute',
          left: '0%',
          width: `${mapX(split2)}%`,
          top: `${mapY(split1)}%`,
          bottom: '0%',
          background: 'rgba(245, 158, 11, 0.08)',
          transition: 'all 0.1s'
        }} />

        {/* Region LR: Petal Width <= split1 AND Sepal Length > split2 */}
        <div style={{
          position: 'absolute',
          left: `${mapX(split2)}%`,
          right: '0%',
          top: `${mapY(split1)}%`,
          bottom: '0%',
          background: 'rgba(59, 130, 246, 0.04)', // Virginica light tint
          transition: 'all 0.1s'
        }} />

        {/* Region RL: Petal Width > split1 AND Sepal Length <= split3 */}
        <div style={{
          position: 'absolute',
          left: '0%',
          width: `${mapX(split3)}%`,
          top: '0%',
          height: `${mapY(split1)}%`,
          background: 'rgba(245, 158, 11, 0.04)',
          transition: 'all 0.1s'
        }} />

        {/* Region RR: Petal Width > split1 AND Sepal Length > split3 */}
        <div style={{
          position: 'absolute',
          left: `${mapX(split3)}%`,
          right: '0%',
          top: '0%',
          height: `${mapY(split1)}%`,
          background: 'rgba(59, 130, 246, 0.09)',
          transition: 'all 0.1s'
        }} />

        {/* Grid lines inside plot */}
        <svg width="100%" height="100%" style={{ position: 'absolute', left: 0, top: 0, pointerEvents: 'none' }}>
          {/* Split 1 line (Petal Width) horizontal */}
          <line
            x1="0"
            y1={`${mapY(split1)}%`}
            x2="100%"
            y2={`${mapY(split1)}%`}
            stroke="var(--color-tree)"
            strokeWidth="2.5"
            strokeDasharray="4 2"
          />
          <text x="10" y={`${mapY(split1) - 1.5}%`} fill="var(--color-tree)" fontSize="8.5px" fontWeight="bold">Split 1 (Petal Width = {split1})</text>

          {/* Split 2 line (Left Sepal Length) vertical */}
          <line
            x1={`${mapX(split2)}%`}
            y1={`${mapY(split1)}%`}
            x2={`${mapX(split2)}%`}
            y2="100%"
            stroke="var(--color-regression)"
            strokeWidth="2"
            strokeDasharray="4 2"
          />
          <text x={`${mapX(split2) + 1}%`} y="95%" fill="var(--color-regression)" fontSize="8px">Split 2 ({split2})</text>

          {/* Split 3 line (Right Sepal Length) vertical */}
          <line
            x1={`${mapX(split3)}%`}
            y1="0"
            x2={`${mapX(split3)}%`}
            y2={`${mapY(split1)}%`}
            stroke="var(--color-cnn)"
            strokeWidth="2"
            strokeDasharray="4 2"
          />
          <text x={`${mapX(split3) + 1}%`} y="15%" fill="var(--color-cnn)" fontSize="8px">Split 3 ({split3})</text>

          {/* Draw data points */}
          {IRIS_DATA.map((p, idx) => (
            <circle
              key={idx}
              cx={`${mapX(p.x)}%`}
              cy={`${mapY(p.y)}%`}
              r="6"
              fill={p.label === 1 ? 'var(--color-neural)' : 'var(--color-warning)'}
              stroke="#ffffff"
              strokeWidth="1.5"
              style={{ filter: 'drop-shadow(0 1px 3px rgba(0,0,0,0.5))' }}
            />
          ))}
        </svg>

        {/* Axes titles */}
        <div style={{ position: 'absolute', bottom: '8px', left: '50%', transform: 'translateX(-50%)', fontSize: '9px', color: 'var(--text-muted)', fontWeight: 'bold' }}>
          Sepal Length (X-axis)
        </div>
        <div style={{ position: 'absolute', left: '8px', top: '50%', transform: 'translateY(-50%) rotate(-90deg)', transformOrigin: 'left center', fontSize: '9px', color: 'var(--text-muted)', fontWeight: 'bold' }}>
          Petal Width (Y-axis)
        </div>
      </div>
    );
  };

  return (
    <div className="animate-fade-in" style={{ display: 'grid', gridTemplateColumns: '1fr', gap: '1.5rem' }}>
      {/* Metrics Row */}
      <div className="stats-grid">
        <div className="stat-card" style={{ borderColor: 'var(--color-tree-glow)' }}>
          <span className="stat-label">Model Accuracy</span>
          <span className="stat-value" style={{ color: accuracy > 0.85 ? '#6ee7b7' : '#fcd34d' }}>
            {(accuracy * 100).toFixed(1)}%
          </span>
        </div>
        <div className="stat-card">
          <span className="stat-label">Impurity Metric</span>
          <span className="stat-value" style={{ textTransform: 'capitalize', fontSize: '1.2rem' }}>
            {metric} Impurity
          </span>
        </div>
        <div className="stat-card">
          <span className="stat-label">Dataset</span>
          <span className="stat-value" style={{ fontSize: '1.2rem' }}>Iris Classification</span>
        </div>
        <div className="stat-card">
          <span className="stat-label">Tree Leaves</span>
          <span className="stat-value">4 Leafs</span>
        </div>
      </div>

      <div className="dashboard-layout">
        {/* Sliders Control Sidebar */}
        <div className="glass-card flex flex-col gap-4 accent-tree">
          <h3 style={{ borderBottom: '1px solid var(--border-color)', paddingBottom: '0.5rem' }}>Splits Control</h3>

          {/* Metric Selector Toggle */}
          <div className="form-group">
            <span className="form-label">Impurity Calculation</span>
            <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '0.25rem' }}>
              <button
                className="case-study-btn"
                style={{
                  backgroundColor: metric === 'gini' ? 'rgba(16, 185, 129, 0.15)' : 'rgba(255,255,255,0.02)',
                  borderColor: metric === 'gini' ? 'var(--color-tree)' : 'var(--border-color)',
                  color: metric === 'gini' ? '#fff' : 'var(--text-secondary)'
                }}
                onClick={() => setMetric('gini')}
              >
                Gini Impurity
              </button>
              <button
                className="case-study-btn"
                style={{
                  backgroundColor: metric === 'entropy' ? 'rgba(16, 185, 129, 0.15)' : 'rgba(255,255,255,0.02)',
                  borderColor: metric === 'entropy' ? 'var(--color-tree)' : 'var(--border-color)',
                  color: metric === 'entropy' ? '#fff' : 'var(--text-secondary)'
                }}
                onClick={() => setMetric('entropy')}
              >
                Shannon Entropy
              </button>
            </div>
          </div>

          {/* Split 1 slider */}
          <div className="form-group">
            <span className="form-label">
              Root Split (Petal Width) <span className="form-value" style={{ color: 'var(--color-tree)' }}>{split1}</span>
            </span>
            <input
              type="range"
              className="slider-tree"
              min="0.6"
              max="2.4"
              step="0.1"
              value={split1}
              onChange={(e) => setSplit1(parseFloat(e.target.value))}
            />
          </div>

          {/* Split 2 slider */}
          <div className="form-group">
            <span className="form-label">
              Left Split (Sepal Length) <span className="form-value" style={{ color: 'var(--color-regression)' }}>{split2}</span>
            </span>
            <input
              type="range"
              className="slider-tree"
              min="4.6"
              max="7.4"
              step="0.1"
              value={split2}
              onChange={(e) => setSplit2(parseFloat(e.target.value))}
            />
          </div>

          {/* Split 3 slider */}
          <div className="form-group">
            <span className="form-label">
              Right Split (Sepal Length) <span className="form-value" style={{ color: 'var(--color-cnn)' }}>{split3}</span>
            </span>
            <input
              type="range"
              className="slider-tree"
              min="4.6"
              max="7.4"
              step="0.1"
              value={split3}
              onChange={(e) => setSplit3(parseFloat(e.target.value))}
            />
          </div>

          {/* Reset buttons */}
          <div style={{ display: 'flex', flexDirection: 'column', gap: '0.5rem', marginTop: '0.5rem' }}>
            <button className="btn btn-secondary" onClick={() => { setSplit1(1.6); setSplit2(5.8); setSplit3(6.2); }}>
              Reset Thresholds
            </button>
          </div>
          
          <div style={{ fontSize: '0.75rem', color: 'var(--text-muted)', lineHeight: '1.4' }}>
            <span style={{ fontWeight: 'bold', color: 'var(--text-secondary)' }}>Note:</span> Adjust the sliders to see how partitions slide on the scatter plot and how Gini values at each node recalculate. Aim for 100% accuracy.
          </div>
        </div>

        {/* Center workspace grid */}
        <div style={{ display: 'grid', gridTemplateColumns: '1fr', gap: '1.5rem' }}>
          {/* Decision Scatter Grid */}
          {drawScatterPlot()}

          {/* Tree SVG Node Diagram */}
          <div className="flex flex-col gap-2">
            <span className="form-label" style={{ display: 'flex', gap: '0.25rem', alignItems: 'center' }}>
              <GitFork size={14} /> Generated Hierarchical Tree
            </span>
            <div style={{ height: '240px', width: '100%' }}>
              {drawTreeSVG()}
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}
