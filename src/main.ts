import './style.css';
import { AuroraRenderer } from './aurora-renderer';

async function main() {
  // Create fullscreen canvas
  const canvas = document.createElement('canvas');
  canvas.id = 'aurora-canvas';
  document.body.appendChild(canvas);

  // Set canvas size to window size
  const resize = () => {
    canvas.width = window.innerWidth * window.devicePixelRatio;
    canvas.height = window.innerHeight * window.devicePixelRatio;
    canvas.style.width = `${window.innerWidth}px`;
    canvas.style.height = `${window.innerHeight}px`;
  };
  resize();
  window.addEventListener('resize', resize);

  // Initialize renderer
  const renderer = new AuroraRenderer(canvas);
  const success = await renderer.initialize();

  if (!success) {
    // Show error message if WebGPU not available
    const errorDiv = document.createElement('div');
    errorDiv.className = 'error-message';
    errorDiv.innerHTML = `
            <h1>WebGPU Not Supported</h1>
            <p>Your browser doesn't support WebGPU yet.</p>
            <p>Try using Chrome 113+ or Edge 113+ with WebGPU enabled.</p>
        `;
    document.body.appendChild(errorDiv);
    return;
  }

  // Animation loop
  function animate() {
    renderer.render();
    requestAnimationFrame(animate);
  }
  animate();
}

main();
