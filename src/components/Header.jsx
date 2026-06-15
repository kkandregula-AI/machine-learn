import React, { useState, useEffect } from 'react';
import { Sparkles, Download, CheckCircle, BrainCircuit } from 'lucide-react';

export default function Header({ activeTab }) {
  const [deferredPrompt, setDeferredPrompt] = useState(null);
  const [isInstalled, setIsInstalled] = useState(false);

  useEffect(() => {
    // Check if app is already running in standalone (PWA) mode
    if (window.matchMedia('(display-mode: standalone)').matches || window.navigator.standalone) {
      setIsInstalled(true);
    }

    const handleBeforeInstallPrompt = (e) => {
      // Prevent Chrome 67 and earlier from automatically showing the prompt
      e.preventDefault();
      // Stash the event so it can be triggered later.
      setDeferredPrompt(e);
    };

    window.addEventListener('beforeinstallprompt', handleBeforeInstallPrompt);

    const handleAppInstalled = () => {
      setIsInstalled(true);
      setDeferredPrompt(null);
      console.log('MLPlayground was installed successfully.');
    };

    window.addEventListener('appinstalled', handleAppInstalled);

    return () => {
      window.removeEventListener('beforeinstallprompt', handleBeforeInstallPrompt);
      window.removeEventListener('appinstalled', handleAppInstalled);
    };
  }, []);

  const handleInstallClick = async () => {
    if (!deferredPrompt) return;
    
    // Show the install prompt
    deferredPrompt.prompt();
    
    // Wait for the user to respond to the prompt
    const { outcome } = await deferredPrompt.userChoice;
    console.log(`User response to install prompt: ${outcome}`);
    
    // We've used the prompt, and can't use it again
    setDeferredPrompt(null);
  };

  // Get active tab color for accents
  const getAccentColor = () => {
    switch (activeTab) {
      case 'regression': return 'var(--color-regression)';
      case 'neural': return 'var(--color-neural)';
      case 'cnn': return 'var(--color-cnn)';
      case 'tree': return 'var(--color-tree)';
      default: return 'var(--color-regression)';
    }
  };

  return (
    <header className="glass-card" style={{ padding: '1.25rem 1.75rem', borderBottom: `2px solid ${getAccentColor()}` }}>
      <div className="flex justify-between items-center" style={{ flexWrap: 'wrap', gap: '1rem' }}>
        <div className="flex items-center gap-4">
          <div style={{
            background: 'rgba(255,255,255,0.04)',
            padding: '0.6rem',
            borderRadius: '12px',
            border: '1px solid rgba(255,255,255,0.08)',
            display: 'flex',
            alignItems: 'center',
            justifyContent: 'center',
            boxShadow: `0 0 15px ${getAccentColor()}33`
          }}>
            <BrainCircuit size={28} style={{ color: getAccentColor() }} />
          </div>
          <div>
            <h1 style={{ fontSize: '1.75rem', letterSpacing: '-0.03em', display: 'flex', alignItems: 'center', gap: '0.5rem' }}>
              MLPlayground 
              <Sparkles size={16} style={{ color: getAccentColor(), animation: 'pulseGlow 2s infinite ease-in-out' }} />
            </h1>
            <p style={{ fontSize: '0.85rem', color: 'var(--text-secondary)' }}>
              Interactive Explainer & Visualizer for Machine Learning
            </p>
          </div>
        </div>

        <div className="flex items-center gap-4">
          {/* PWA Installation Badge/Button */}
          {isInstalled ? (
            <div style={{
              display: 'flex',
              alignItems: 'center',
              gap: '0.35rem',
              background: 'rgba(16, 185, 129, 0.1)',
              border: '1px solid rgba(16, 185, 129, 0.2)',
              color: '#6ee7b7',
              fontSize: '0.8rem',
              padding: '0.4rem 0.8rem',
              borderRadius: '20px',
              fontWeight: '500'
            }}>
              <CheckCircle size={14} />
              Offline Ready
            </div>
          ) : deferredPrompt ? (
            <button 
              className="btn btn-secondary" 
              onClick={handleInstallClick}
              style={{
                fontSize: '0.8rem',
                padding: '0.4rem 0.9rem',
                display: 'flex',
                alignItems: 'center',
                gap: '0.4rem',
                borderColor: getAccentColor(),
                boxShadow: `0 0 10px ${getAccentColor()}33`,
                animation: 'fadeIn 0.5s ease'
              }}
            >
              <Download size={14} style={{ color: getAccentColor() }} />
              Install PWA App
            </button>
          ) : (
            <div style={{
              display: 'flex',
              alignItems: 'center',
              gap: '0.35rem',
              background: 'rgba(255,255,255,0.03)',
              border: '1px solid var(--border-color)',
              color: 'var(--text-muted)',
              fontSize: '0.8rem',
              padding: '0.4rem 0.8rem',
              borderRadius: '20px'
            }}>
              <Sparkles size={12} style={{ color: getAccentColor() }} />
              Web Mode
            </div>
          )}
        </div>
      </div>
    </header>
  );
}
