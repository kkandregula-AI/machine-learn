import React from 'react';
import { BookOpen, Lightbulb, PlayCircle, HelpCircle } from 'lucide-react';

export default function ExplainerPanel({ activeTab }) {
  const getPanelData = () => {
    switch (activeTab) {
      case 'regression':
        return {
          title: 'Regression Analysis',
          subtitle: 'Predicting Continuous Values',
          color: 'var(--color-regression)',
          concept: 'Regression models find relationships between input features (like size of a house) and a continuous target value (like its price). By calculating gradients, the model iteratively shifts weights to fit the dataset.',
          equations: [
            { label: 'Linear Model', formula: 'y = wx + b' },
            { label: 'Mean Squared Error (Loss)', formula: 'L = (1/N) * Σ (y_pred - y_actual)²' },
            { label: 'Gradient Descent (Weight Update)', formula: 'w ← w - η * (∂L/∂w)' }
          ],
          terms: [
            { name: 'Weight (w)', desc: 'The slope of the line. It adjusts the strength of the input variable.' },
            { name: 'Bias (b)', desc: 'The y-intercept. It shifts the entire fitting line up or down.' },
            { name: 'Learning Rate (η)', desc: 'The size of the steps the model takes when updating parameters. Too high causes oscillation; too low makes learning extremely slow.' },
            { name: 'Regularization (L1/L2)', desc: 'Penalties added to the loss function to prevent overfitting by keeping weights small. L1 (Lasso) drives weights to zero, while L2 (Ridge) shrinks them smoothly.' }
          ],
          challenges: [
            'Load the "Salary vs Experience" dataset and train it using a Linear model. Observe the high loss since the relationship is non-linear.',
            'Switch to a Polynomial model of degree 2 or 3. Notice how the fitting curve bends to capture the salary curve with a much lower loss.',
            'Load the "House Size vs Price" dataset, click to add an outlier high above the trend line, and compare L1 vs L2 regularization. See how L1 is less sensitive to the outlier!'
          ]
        };

      case 'neural':
        return {
          title: 'Neural Networks (MLP)',
          subtitle: 'Deep Learning Decision Boundaries',
          color: 'var(--color-neural)',
          concept: 'Multi-Layer Perceptrons mimic biological brains by passing inputs through layers of interconnected neurons. Each neuron computes a weighted sum, applies a non-linear activation function, and feeds it forward to classify complex, non-linear boundaries.',
          equations: [
            { label: 'Neuron Activation', formula: 'a = σ( Σ (w_i * x_i) + b )' },
            { label: 'Sigmoid Function', formula: 'σ(z) = 1 / (1 + e^-z)' },
            { label: 'ReLU Function', formula: 'f(z) = max(0, z)' }
          ],
          terms: [
            { name: 'Hidden Layers', desc: 'Layers between inputs and outputs that extract abstract representations of the input features.' },
            { name: 'Activation Functions', desc: 'Functions like ReLU or Tanh that introduce non-linearity. Without them, neural networks would only be capable of learning linear splits.' },
            { name: 'Backpropagation', desc: 'The algorithm that calculates gradients of the loss function starting from the output layer and working backward to update the network weights.' },
            { name: 'Epoch', desc: 'One complete pass of the training dataset through the neural network.' }
          ],
          challenges: [
            'Select the XOR dataset and set Hidden Layers to 0 (Linear model). Notice how it is mathematically impossible for the line to split the quadrants.',
            'Add 1 hidden layer with 3 neurons, and set Activation to ReLU. Click Train and watch the decision boundary dynamically morph to solve XOR!',
            'Load the Spiral dataset. Try different combinations of learning rates and network depths. Can you classify it cleanly with a 3-layer neural network?'
          ]
        };

      case 'cnn':
        return {
          title: 'Convolutional Networks',
          subtitle: 'Visual Feature Map Extractors',
          color: 'var(--color-cnn)',
          concept: 'Convolutional Neural Networks (CNNs) process spatial data like images using small grids called kernels/filters. As the kernel slides (convolves) across pixels, it performs element-wise multiplications, highlighting visual features like edges, textures, or blurs.',
          equations: [
            { label: '2D Convolution Math', formula: 'S(i,j) = Σ_m Σ_n I(i+m, j+n) * K(m, n)' },
            { label: 'Filter Dimension', formula: 'Usually odd matrices like 3x3 or 5x5' }
          ],
          terms: [
            { name: 'Kernel (Filter)', desc: 'A matrix of weights designed to detect specific image features (e.g., Sobel filter for vertical edges).' },
            { name: 'Feature Map', desc: 'The output grid of a convolution layer showing where specific visual features were detected.' },
            { name: 'Stride', desc: 'The step size of the kernel as it slides across the input image. A stride of 1 moves one pixel at a time.' },
            { name: 'Padding', desc: 'Adding boundary pixels (often zeros) to the input edges so the kernel can fit over edge pixels, keeping the output size constant.' }
          ],
          challenges: [
            'Select the "MNIST 7 Digit" image and choose the "Edge Detection (Sobel)" filter. Watch how the filter highlights the outlines of the digit.',
            'Click "Step" or "Auto-Slide" to watch the kernel slide pixel-by-pixel. Observe the panel calculations to understand how dot-product multiplication is calculated.',
            'Select "Gaussian Blur" and inspect the outputs. Notice how high-frequency sharp differences (like edges) are smoothed out by the weighted averaging.'
          ]
        };

      case 'tree':
        return {
          title: 'Decision Trees',
          subtitle: 'Hierarchical Rule-Based Classification',
          color: 'var(--color-tree)',
          concept: 'Decision trees partition data by asking a series of binary yes/no questions. At each branch, the algorithm selects the feature split that maximizes information gain, reducing disorder in the remaining subgroups.',
          equations: [
            { label: 'Entropy (Disorder)', formula: 'H(S) = - Σ p_i * log₂(p_i)' },
            { label: 'Gini Impurity', formula: 'Gini = 1 - Σ (p_i)²' },
            { label: 'Information Gain', formula: 'IG(S, A) = H(S) - Σ (|S_v|/|S|) * H(S_v)' }
          ],
          terms: [
            { name: 'Root Node', desc: 'The topmost node in a decision tree representing the first feature split.' },
            { name: 'Split Threshold', desc: 'The boundary condition (e.g., Age > 30) that separates the samples.' },
            { name: 'Leaf Node', desc: 'A terminal node that does not split further, representing the final class prediction.' },
            { name: 'Gini Impurity', desc: 'A measure of how often a randomly chosen element from the set would be incorrectly labeled. A Gini of 0 represents a perfectly pure group.' }
          ],
          challenges: [
            'Look at the Iris Flower data splits on the scatter plot. Adjust the threshold slider for the root split and see how the dividing line moves.',
            'Increase the Maximum Tree Depth. Watch how the node graph expands, creating smaller sub-boxes in the plot to classify every point.',
            'Change the splitting criteria from "Gini Impurity" to "Entropy" and check if the root split changes. Observe how decision paths are traced for single samples.'
          ]
        };

      default:
        return null;
    }
  };

  const data = getPanelData();
  if (!data) return null;

  return (
    <div className="glass-card flex flex-col gap-4 animate-fade-in" style={{ height: 'fit-content' }}>
      <div>
        <div style={{ display: 'flex', alignItems: 'center', gap: '0.5rem', marginBottom: '0.25rem' }}>
          <BookOpen size={18} style={{ color: data.color }} />
          <h2 style={{ fontSize: '1.25rem' }}>{data.title}</h2>
        </div>
        <p style={{ fontSize: '0.8rem', color: data.color, fontWeight: '600', textTransform: 'uppercase', letterSpacing: '0.05em' }}>
          {data.subtitle}
        </p>
      </div>

      <div style={{ background: 'rgba(255,255,255,0.02)', padding: '0.85rem', borderRadius: '10px', border: '1px solid var(--border-color)' }}>
        <p style={{ fontSize: '0.85rem', lineHeight: '1.5' }}>{data.concept}</p>
      </div>

      {/* Equations Section */}
      <div>
        <h3 style={{ fontSize: '0.9rem', marginBottom: '0.5rem', display: 'flex', alignItems: 'center', gap: '0.35rem' }}>
          <span style={{ color: data.color }}>■</span> Mathematical Models
        </h3>
        <div style={{ display: 'flex', flexCol: 'column', gap: '0.5rem' }}>
          {data.equations.map((eq, i) => (
            <div key={i} style={{ 
              background: 'rgba(0,0,0,0.25)', 
              padding: '0.6rem 0.8rem', 
              borderRadius: '6px', 
              border: '1px solid rgba(255,255,255,0.03)' 
            }}>
              <div style={{ fontSize: '0.7rem', color: 'var(--text-muted)', fontWeight: '600' }}>{eq.label}</div>
              <div className="font-mono" style={{ fontSize: '0.8rem', marginTop: '0.2rem', color: '#cbd5e1', overflowX: 'auto', whiteSpace: 'nowrap' }}>
                {eq.formula}
              </div>
            </div>
          ))}
        </div>
      </div>

      {/* Terms Section */}
      <div>
        <h3 style={{ fontSize: '0.9rem', marginBottom: '0.5rem', display: 'flex', alignItems: 'center', gap: '0.35rem' }}>
          <span style={{ color: data.color }}>■</span> Key Vocabulary
        </h3>
        <div style={{ display: 'flex', flexDirection: 'column', gap: '0.6rem' }}>
          {data.terms.map((term, i) => (
            <div key={i} style={{ fontSize: '0.8rem' }}>
              <strong style={{ color: 'var(--text-primary)', display: 'block', marginBottom: '0.1rem' }}>{term.name}</strong>
              <span style={{ color: 'var(--text-secondary)' }}>{term.desc}</span>
            </div>
          ))}
        </div>
      </div>

      {/* Guided Exercises Section */}
      <div style={{ 
        borderTop: '1px solid var(--border-color)', 
        paddingTop: '1rem', 
        marginTop: '0.5rem'
      }}>
        <h3 style={{ fontSize: '0.9rem', marginBottom: '0.6rem', display: 'flex', alignItems: 'center', gap: '0.4rem', color: '#f1f5f9' }}>
          <Lightbulb size={16} style={{ color: 'var(--color-warning)' }} />
          Guided Challenges
        </h3>
        <ol style={{ paddingLeft: '1.2rem', display: 'flex', flexDirection: 'column', gap: '0.5rem' }}>
          {data.challenges.map((challenge, i) => (
            <li key={i} style={{ fontSize: '0.8rem', color: 'var(--text-secondary)', lineHeight: '1.4' }}>
              {challenge}
            </li>
          ))}
        </ol>
      </div>
    </div>
  );
}
