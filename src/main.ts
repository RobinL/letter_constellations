import './style.css';
import { AuroraRenderer } from './aurora-renderer';
import { SparkleRenderer } from './sparkle-renderer';
import { CanvasManager } from './canvas-manager';
import { Game } from './game';
import { InputHandler } from './input';
import { PerformanceMonitor } from './performance-monitor';

type ItemEntry = {
  name: string;
  label: string;
  imageUrl: string;
  audioUrl?: string;
};

const itemImageModules = import.meta.glob('./assets/items/*/*.png', {
  eager: true,
  query: { as: 'url' },
});
const itemVoiceModules = import.meta.glob('./assets/voice/*/*.webm', {
  eager: true,
  query: { as: 'url' },
});
const alphabetVoiceModules = import.meta.glob('./assets/voice/alphabet/*.webm', {
  eager: true,
  query: { as: 'url' },
});

const toAssetUrl = (mod: unknown): string => {
  if (typeof mod === 'string') {
    return mod;
  }
  if (mod && typeof mod === 'object' && 'default' in mod) {
    const value = (mod as { default?: unknown }).default;
    return typeof value === 'string' ? value : '';
  }
  return '';
};

const parseAssetInfo = (path: string): { folder: string; name: string } => {
  const parts = path.split('/');
  const fileName = parts[parts.length - 1] ?? '';
  const folder = parts[parts.length - 2] ?? '';
  const name = fileName.replace(/\.[^.]+$/, '');
  return { folder, name };
};

const itemAudioByLetter = new Map<string, Map<string, string>>();
for (const [path, mod] of Object.entries(itemVoiceModules)) {
  const { folder, name } = parseAssetInfo(path);
  if (folder === 'alphabet') {
    continue;
  }
  const normalizedUrl = toAssetUrl(mod);
  if (!normalizedUrl) {
    continue;
  }
  if (!itemAudioByLetter.has(folder)) {
    itemAudioByLetter.set(folder, new Map());
  }
  itemAudioByLetter.get(folder)!.set(name, normalizedUrl);
}

const itemImagesByLetter = new Map<string, ItemEntry[]>();
for (const [path, mod] of Object.entries(itemImageModules)) {
  const { folder, name } = parseAssetInfo(path);
  const label = name.replace(/_/g, ' ');
  const imageUrl = toAssetUrl(mod);
  if (!imageUrl) {
    continue;
  }
  const audioUrl = itemAudioByLetter.get(folder)?.get(name);
  if (!itemImagesByLetter.has(folder)) {
    itemImagesByLetter.set(folder, []);
  }
  itemImagesByLetter.get(folder)!.push({
    name,
    label,
    imageUrl,
    audioUrl,
  });
}

const alphabetAudioByLetter = new Map<string, string>();
for (const [path, mod] of Object.entries(alphabetVoiceModules)) {
  const { name } = parseAssetInfo(path);
  const audioUrl = toAssetUrl(mod);
  if (!audioUrl) {
    continue;
  }
  alphabetAudioByLetter.set(name, audioUrl);
}

const pickRandomItems = <T,>(items: T[], count: number): T[] => {
  const copy = [...items];
  for (let i = copy.length - 1; i > 0; i -= 1) {
    const j = Math.floor(Math.random() * (i + 1));
    [copy[i], copy[j]] = [copy[j], copy[i]];
  }
  return copy.slice(0, Math.min(count, copy.length));
};

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
  let currentVolume = 0.33;

  const ensureAudioContext = () => {
    if (audioContext) return;
    audioContext = new AudioContext();
    const audioSource = audioContext.createMediaElementSource(auroraAudio);
    gainNode = audioContext.createGain();
    gainNode.gain.value = 0.25 * currentVolume;
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
  currentVolume = Number(volumeSlider.value);
  chimeAudio.volume = Math.min(1, currentVolume);
  const voiceAudioCache = new Map<string, HTMLAudioElement>();
  const voiceAudioElements = new Set<HTMLAudioElement>();
  let activeVoiceAudio: HTMLAudioElement | null = null;
  let voiceSequenceId = 0;
  let pendingLetterAudioUrl: string | null = null;

  const getVoiceAudio = (url: string): HTMLAudioElement => {
    let audio = voiceAudioCache.get(url);
    if (!audio) {
      audio = new Audio(url);
      audio.preload = 'auto';
      voiceAudioCache.set(url, audio);
      voiceAudioElements.add(audio);
    }
    audio.volume = Math.min(1, Number(volumeSlider.value));
    audio.muted = auroraAudio.muted;
    return audio;
  };

  const stopActiveVoice = () => {
    if (activeVoiceAudio) {
      activeVoiceAudio.pause();
      activeVoiceAudio.currentTime = 0;
    }
  };

  const playVoiceSequence = async (urls: string[]) => {
    if (urls.length === 0) {
      return;
    }
    const sequenceId = ++voiceSequenceId;
    stopActiveVoice();
    for (const url of urls) {
      if (sequenceId !== voiceSequenceId) {
        return;
      }
      const audio = getVoiceAudio(url);
      activeVoiceAudio = audio;
      audio.currentTime = 0;
      try {
        await audio.play();
      } catch {
        return;
      }
      await new Promise<void>((resolve) => {
        const handleEnded = () => {
          audio.removeEventListener('ended', handleEnded);
          audio.removeEventListener('error', handleEnded);
          resolve();
        };
        audio.addEventListener('ended', handleEnded);
        audio.addEventListener('error', handleEnded);
      });
    }
  };

  const requestLetterSound = (letter: string) => {
    const letterUrl = alphabetAudioByLetter.get(letter);
    if (!letterUrl) {
      return;
    }
    if (!userHasInteracted) {
      pendingLetterAudioUrl = letterUrl;
      return;
    }
    void playVoiceSequence([letterUrl]);
  };

  const updateMuteButton = () => {
    const isMuted = auroraAudio.muted;
    muteButton.textContent = isMuted ? 'Unmute' : 'Mute';
    muteButton.setAttribute('aria-pressed', String(isMuted));
    chimeAudio.muted = isMuted;
    for (const audio of voiceAudioElements) {
      audio.muted = isMuted;
    }
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
    if (pendingLetterAudioUrl) {
      const letterUrl = pendingLetterAudioUrl;
      pendingLetterAudioUrl = null;
      void playVoiceSequence([letterUrl]);
    }
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
    const sliderValue = Number(volumeSlider.value);
    currentVolume = sliderValue;
    if (gainNode) {
      gainNode.gain.value = 0.25 * sliderValue;
    }
    chimeAudio.volume = Math.min(1, sliderValue);
    const voiceVolume = Math.min(1, sliderValue);
    for (const audio of voiceAudioElements) {
      audio.volume = voiceVolume;
    }
  });

  const letterPanel = document.createElement('div');
  letterPanel.className = 'letter-panel';

  const letterDisplay = document.createElement('div');
  letterDisplay.className = 'letter-display';
  letterDisplay.setAttribute('aria-hidden', 'true');

  const itemTray = document.createElement('div');
  itemTray.className = 'item-tray letter-panel__item';

  letterPanel.appendChild(letterDisplay);
  letterPanel.appendChild(itemTray);
  app.appendChild(letterPanel);

  const renderItemsForLetter = (letter: string) => {
    itemTray.replaceChildren();
    letterDisplay.textContent = letter;
    const items = itemImagesByLetter.get(letter) ?? [];
    if (items.length === 0) {
      const empty = document.createElement('div');
      empty.className = 'item-empty';
      empty.textContent = `No ${letter.toUpperCase()} items yet.`;
      itemTray.appendChild(empty);
      return;
    }

    const selection = pickRandomItems(items, 1);
    for (const item of selection) {
      const button = document.createElement('button');
      button.type = 'button';
      button.className = 'item-card';
      button.setAttribute('aria-label', item.label);
      button.addEventListener('click', (event) => {
        event.stopPropagation();
        const urls: string[] = [];
        const letterUrl = alphabetAudioByLetter.get(letter);
        if (letterUrl) {
          urls.push(letterUrl);
        }
        if (item.audioUrl) {
          urls.push(item.audioUrl);
        }
        void playVoiceSequence(urls);
      });

      const image = document.createElement('img');
      image.src = item.imageUrl;
      image.alt = item.label;
      image.loading = 'lazy';
      image.decoding = 'async';

      button.appendChild(image);
      itemTray.appendChild(button);
    }
  };

  // Set canvas sizes to window size with device pixel ratio
  let renderer: AuroraRenderer | null = null;
  let sparkleRenderer: SparkleRenderer | null = null;
  let game: Game | null = null;
  const performanceMonitor = new PerformanceMonitor();
  const canvasManager = new CanvasManager(auroraCanvas, gameCanvas, sparkleCanvas);

  // Apply initial quality settings
  canvasManager.updateQuality(performanceMonitor.getQualitySettings());

  const gameContext = gameCanvas.getContext('2d');

  if (!gameContext) {
    throw new Error('Failed to get 2D context for game canvas.');
  }

  const resize = () => {
    const { width, height } = canvasManager.resize();
    canvasManager.configureGameContext(gameContext);
    if (game) {
      game.setViewportSize(width, height);
    }

    // Update renderers with new size (if initialized)
    if (renderer) {
      renderer.resize(auroraCanvas.width, auroraCanvas.height);
    }
    if (sparkleRenderer) {
      sparkleRenderer.resize(sparkleCanvas.width, sparkleCanvas.height);
    }
  };

  // Set up quality change handler
  performanceMonitor.setOnQualityChange((settings) => {
    canvasManager.updateQuality(settings);
    resize(); // Reapply sizes with new quality
  });

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
    onLetterChange: (letter) => {
      renderItemsForLetter(letter);
      requestLetterSound(letter);
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
  let auroraFrame = 0;
  function animate(currentTime: number) {
    // Record frame for performance monitoring
    performanceMonitor.recordFrame();

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

    // Render aurora background every other frame
    if ((auroraFrame++ & 1) === 0) {
      renderer.render();
    }

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
