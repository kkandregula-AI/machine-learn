import React, { useState, useEffect } from 'react';
import Header from './components/Header';
import ExplainerPanel from './components/ExplainerPanel';
import RegressionVisualizer from './visualizers/RegressionVisualizer';
import NeuralNetworkVisualizer from './visualizers/NeuralNetworkVisualizer';
import CNNVisualizer from './visualizers/CNNVisualizer';
import DecisionTreeVisualizer from './visualizers/DecisionTreeVisualizer';
import * as serviceWorker from './sw-register';

export default function App() {
  const [activeTab, setActiveTab] = useState('regression'); // 'regression', 'neural', 'cnn', 'tree'

  // Register service worker on mount
  useEffect(() => {
    serviceWorker.register();
  }, []);

  const renderActiveVisualizer = () => {
    switch (activeTab) {
      case 'regression':
        return <RegressionVisualizer />;
      case 'neural':
        return <NeuralNetworkVisualizer />;
      case 'cnn':
        return <CNNVisualizer />;
      case 'tree':
        return <DecisionTreeVisualizer />;
      default:
        return <RegressionVisualizer />;
    }
  };

  return (
    <div className="app-container">
      {/* Header component */}
      <Header activeTab={activeTab} />

      {/* Main navigation Tabs */}
      <nav className="tabs-nav" style={{ marginTop: '0.5rem' }}>
        <button
          className={`tab-btn tab-regression ${activeTab === 'regression' ? 'active' : ''}`}
          onClick={() => setActiveTab('regression')}
        >
          Regression Analysis
        </button>
        <button
          className={`tab-btn tab-neural ${activeTab === 'neural' ? 'active' : ''}`}
          onClick={() => setActiveTab('neural')}
        >
          Neural Networks (MLP)
        </button>
        <button
          className={`tab-btn tab-cnn ${activeTab === 'cnn' ? 'active' : ''}`}
          onClick={() => setActiveTab('cnn')}
        >
          CNN Feature Extraction
        </button>
        <button
          className={`tab-btn tab-tree ${activeTab === 'tree' ? 'active' : ''}`}
          onClick={() => setActiveTab('tree')}
        >
          Decision Trees
        </button>
      </nav>

      {/* Main workspace (split between visualizer and educational sidebar) */}
      <main className="dashboard-layout with-explainer" style={{ flexGrow: 1 }}>
        {/* Left Side: Active Visualizer */}
        <section style={{ display: 'flex', flexDirection: 'column', minWidth: 0 }}>
          {renderActiveVisualizer()}
        </section>

        {/* Right Side: Explainer panel */}
        <aside>
          <ExplainerPanel activeTab={activeTab} />
        </aside>
      </main>

      {/* Subtle footer */}
      <footer style={{ 
        textAlign: 'center', 
        fontSize: '0.75rem', 
        color: 'var(--text-muted)', 
        borderTop: '1px solid var(--border-color)', 
        paddingTop: '1rem',
        marginTop: '2rem'
      }}>
        MLPlayground PWA • Created for students to understand ML & Deep Learning concepts easily.
      </footer>
    </div>
  );
}
