import './style.css';
import { AuroraRenderer } from './aurora-renderer';
import { SparkleRenderer } from './sparkle-renderer';
import { CanvasManager } from './canvas-manager';
import { Game } from './game';
import { InputHandler } from './input';

async function main() {
  const app = document.querySelector<HTMLDivElement>('#app')!;

  // Create fullscreen aurora canvas (WebGPU background)
  const auroraCanvas = document.createElement('canvas');
  auroraCanvas.id = 'aurora-canvas';
  app.appendChild(auroraCanvas);

  // Create sparkle canvas (WebGPU effects layer)
  const sparkleCanvas = document.createElement('canvas');
  sparkleCanvas.id = 'sparkle-canvas';
  app.appendChild(sparkleCanvas);

  // Create overlay canvas for game interactions (2D)
  const gameCanvas = document.createElement('canvas');
  gameCanvas.id = 'game-canvas';
  app.appendChild(gameCanvas);

  // Set canvas sizes to window size with device pixel ratio
  let renderer: AuroraRenderer | null = null;
  let sparkleRenderer: SparkleRenderer | null = null;
  const canvasManager = new CanvasManager(auroraCanvas, gameCanvas, sparkleCanvas);
  const gameContext = gameCanvas.getContext('2d');

  if (!gameContext) {
    throw new Error('Failed to get 2D context for game canvas.');
  }

  const resize = () => {
    const { width, height, dpr } = canvasManager.resize();
    canvasManager.configureGameContext(gameContext);

    // Update renderers with new size (if initialized)
    if (renderer) {
      renderer.resize(width * dpr, height * dpr);
    }
    if (sparkleRenderer) {
      sparkleRenderer.resize(width * dpr, height * dpr);
    }
  };
  resize();
  window.addEventListener('resize', resize);

  // Initialize aurora renderer
  renderer = new AuroraRenderer(auroraCanvas);
  const auroraSuccess = await renderer.initialize();

  // Initialize sparkle renderer
  sparkleRenderer = new SparkleRenderer(sparkleCanvas);
  const sparkleSuccess = await sparkleRenderer.initialize();

  // Call resize after initialization to ensure proper WebGPU context configuration
  resize();

  if (!auroraSuccess || !sparkleSuccess) {
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

  const input = new InputHandler(gameCanvas, (event) =>
    canvasManager.getPointFromEvent(gameCanvas, event)
  );
  const game = new Game(input);

  // Main animation loop
  let lastTime = performance.now();
  function animate(currentTime: number) {
    const deltaTime = (currentTime - lastTime) / 1000; // seconds
    lastTime = currentTime;

    game.update(deltaTime);

    // Get drawing state for sparkle renderer
    const drawingState = game.getDrawingState();
    sparkleRenderer.updateMouseState(
      drawingState.isDrawing,
      drawingState.mouseX,
      drawingState.mouseY,
      drawingState.trailPoints
    );

    // Render aurora background
    renderer.render();

    // Render sparkle effects
    sparkleRenderer.render();

    game.render(gameContext);

    requestAnimationFrame(animate);
  }
  animate(performance.now());
}

main();
