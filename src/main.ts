import './style.css';
import { AuroraRenderer } from './aurora-renderer';
import { SparkleRenderer } from './sparkle-renderer';
import { CanvasManager } from './canvas-manager';
import { Game } from './game';
import { InputHandler } from './input';

async function main() {
  const app = document.querySelector<HTMLDivElement>('#app')!;
  const auroraAudioUrl = new URL('./assets/aurora.mp3', import.meta.url).toString();
  const auroraAudio = new Audio(auroraAudioUrl);
  auroraAudio.loop = true;
  auroraAudio.preload = 'auto';
  auroraAudio.volume = 1;
  const chimeAudioUrl = new URL('./assets/chime.mp3', import.meta.url).toString();
  const chimeAudio = new Audio(chimeAudioUrl);
  chimeAudio.preload = 'auto';
  let audioContext: AudioContext | null = null;
  let gainNode: GainNode | null = null;

  const ensureAudioContext = () => {
    if (audioContext) return;
    audioContext = new AudioContext();
    const audioSource = audioContext.createMediaElementSource(auroraAudio);
    gainNode = audioContext.createGain();
    gainNode.gain.value = 0.33;
    audioSource.connect(gainNode).connect(audioContext.destination);
  };

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

  const audioAttribution = document.createElement('div');
  audioAttribution.className = 'audio-attribution';
  audioAttribution.innerHTML = `
    <span>Music: </span>
    <a href="https://freemusicarchive.org/music/Kevin_Hartnell/Umbra_1955/Kevin_Hartnell_-_Umbra_-_10_Aurora/" target="_blank" rel="noopener noreferrer">Aurora</a>
    <span> by Kevin Hartnell — </span>
    <a href="https://creativecommons.org/licenses/by/4.0/" target="_blank" rel="noopener noreferrer">CC BY 4.0</a>
    <span> (no changes)</span>
  `;
  app.appendChild(audioAttribution);

  const muteButton = document.createElement('button');
  muteButton.className = 'mute-button';
  muteButton.type = 'button';
  muteButton.textContent = 'Mute';
  muteButton.setAttribute('aria-pressed', 'false');
  app.appendChild(muteButton);

  const topControls = document.createElement('div');
  topControls.className = 'top-controls';

  const clearButton = document.createElement('button');
  clearButton.className = 'control-button';
  clearButton.type = 'button';
  clearButton.textContent = 'Clear';
  topControls.appendChild(clearButton);

  const resetButton = document.createElement('button');
  resetButton.className = 'control-button';
  resetButton.type = 'button';
  resetButton.textContent = 'Reset';
  topControls.appendChild(resetButton);

  app.appendChild(topControls);

  const volumeControl = document.createElement('div');
  volumeControl.className = 'volume-control';
  volumeControl.innerHTML = `
    <label for="volume-slider">Volume</label>
    <input id="volume-slider" type="range" min="0" max="2" step="0.05" value="0.33" />
  `;
  app.appendChild(volumeControl);
  const volumeSlider = volumeControl.querySelector<HTMLInputElement>('#volume-slider')!;
  chimeAudio.volume = Math.min(1, Number(volumeSlider.value));

  const updateMuteButton = () => {
    const isMuted = auroraAudio.muted;
    muteButton.textContent = isMuted ? 'Unmute' : 'Mute';
    muteButton.setAttribute('aria-pressed', String(isMuted));
    chimeAudio.muted = isMuted;
  };
  updateMuteButton();

  let userHasInteracted = false;

  const attemptPlay = () => {
    if (!userHasInteracted) return;
    ensureAudioContext();
    if (audioContext && audioContext.state === 'suspended') {
      audioContext.resume().catch(() => {
        // Resume can fail until user gesture; playback will retry.
      });
    }
    auroraAudio.play().catch(() => {
      // Autoplay restrictions are expected; user gesture will retry.
    });
  };

  const startAudioOnFirstInteraction = () => {
    userHasInteracted = true;
    attemptPlay();
  };
  const firstInteractionOptions: AddEventListenerOptions = { once: true, capture: true };
  document.addEventListener('pointerdown', startAudioOnFirstInteraction, firstInteractionOptions);
  document.addEventListener('touchstart', startAudioOnFirstInteraction, firstInteractionOptions);
  document.addEventListener('click', startAudioOnFirstInteraction, firstInteractionOptions);
  document.addEventListener('keydown', startAudioOnFirstInteraction, firstInteractionOptions);

  muteButton.addEventListener('click', (event) => {
    event.stopPropagation();
    auroraAudio.muted = !auroraAudio.muted;
    if (!auroraAudio.muted) {
      attemptPlay();
    }
    updateMuteButton();
  });

  volumeSlider.addEventListener('input', () => {
    ensureAudioContext();
    if (gainNode) {
      gainNode.gain.value = Number(volumeSlider.value);
    }
    chimeAudio.volume = Math.min(1, Number(volumeSlider.value));
  });

  // Set canvas sizes to window size with device pixel ratio
  let renderer: AuroraRenderer | null = null;
  let sparkleRenderer: SparkleRenderer | null = null;
  let game: Game | null = null;
  const canvasManager = new CanvasManager(auroraCanvas, gameCanvas, sparkleCanvas);
  const gameContext = gameCanvas.getContext('2d');

  if (!gameContext) {
    throw new Error('Failed to get 2D context for game canvas.');
  }

  const resize = () => {
    const { width, height, dpr } = canvasManager.resize();
    canvasManager.configureGameContext(gameContext);
    if (game) {
      game.setViewportSize(width, height);
    }

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
  const sparkleSuccess = await sparkleRenderer.initialize(renderer!.getDevice());

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
  game = new Game(input, {
    onPointHit: () => {
      chimeAudio.currentTime = 0;
      chimeAudio.play().catch(() => {
        // Autoplay restrictions are expected; next gesture will retry.
      });
    },
  });

  clearButton.addEventListener('click', (event) => {
    event.stopPropagation();
    game?.clearUserPaths();
  });

  resetButton.addEventListener('click', (event) => {
    event.stopPropagation();
    game?.resetCurrentLetter();
  });
  const size = canvasManager.getSize();
  game.setViewportSize(size.width, size.height);

  // Main animation loop
  let lastTime = performance.now();
  function animate(currentTime: number) {
    const deltaTime = (currentTime - lastTime) / 1000; // seconds
    lastTime = currentTime;

    if (!game || !renderer || !sparkleRenderer) return;

    game.update(deltaTime);

    // Get drawing state for sparkle renderer
    const drawingState = game.getDrawingState();
    sparkleRenderer.updateMouseState(
      drawingState.isDrawing,
      drawingState.mouseX,
      drawingState.mouseY,
      drawingState.trailPoints
    );

    const dotState = game.getDotState();
    sparkleRenderer.setDots(dotState.dots, dotState.currentIndex, dotState.radius);

    // Render aurora background
    renderer.render();

    // Render sparkle effects
    sparkleRenderer.render();

    if (gameContext) {
      game.render(gameContext);
    }

    requestAnimationFrame(animate);
  }
  animate(performance.now());
}

main();
