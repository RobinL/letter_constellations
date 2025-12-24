import './style.css';
import { AuroraRenderer } from './aurora-renderer';
import { SparkleRenderer } from './sparkle-renderer';
import { CanvasManager } from './canvas-manager';
import { Game, availableLetters } from './game';
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

  const controlsBar = document.createElement('div');
  controlsBar.className = 'controls-bar';

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

  const muteButton = document.createElement('button');
  muteButton.className = 'mute-button';
  muteButton.type = 'button';
  muteButton.textContent = 'Mute';
  muteButton.setAttribute('aria-pressed', 'false');

  controlsBar.appendChild(topControls);

  const volumeControl = document.createElement('div');
  volumeControl.className = 'volume-control';
  volumeControl.innerHTML = `
    <label for="volume-slider">Volume</label>
    <input id="volume-slider" type="range" min="0" max="2" step="0.05" value="0.33" />
  `;
  controlsBar.appendChild(volumeControl);
  controlsBar.appendChild(muteButton);
  app.appendChild(controlsBar);

  // Settings button
  const settingsButton = document.createElement('button');
  settingsButton.className = 'settings-button';
  settingsButton.type = 'button';
  settingsButton.textContent = 'Letters';
  app.appendChild(settingsButton);

  // Settings overlay and modal
  const settingsOverlay = document.createElement('div');
  settingsOverlay.className = 'settings-overlay';

  const settingsModal = document.createElement('div');
  settingsModal.className = 'settings-modal';

  const settingsTitle = document.createElement('h2');
  settingsTitle.textContent = 'Select Letters';
  settingsModal.appendChild(settingsTitle);

  const settingsError = document.createElement('div');
  settingsError.className = 'settings-error';
  settingsError.textContent = 'Please select at least one letter';
  settingsModal.appendChild(settingsError);

  // Letter grid
  const letterGrid = document.createElement('div');
  letterGrid.className = 'letter-grid';

  // All 26 letters for the grid
  const allLetters = 'abcdefghijklmnopqrstuvwxyz'.split('');
  const availableLetterSet = new Set(availableLetters);

  // Load saved selection from localStorage or default to all available
  const STORAGE_KEY = 'letterConstellations_enabledLetters';
  const loadEnabledLetters = (): Set<string> => {
    try {
      const saved = localStorage.getItem(STORAGE_KEY);
      if (saved) {
        const parsed = JSON.parse(saved) as string[];
        const validLetters = parsed.filter((l) => availableLetterSet.has(l));
        if (validLetters.length > 0) {
          return new Set(validLetters);
        }
      }
    } catch {
      // Ignore parse errors
    }
    // Default to all available letters
    return new Set(availableLetters);
  };

  const saveEnabledLetters = (letters: Set<string>): void => {
    try {
      localStorage.setItem(STORAGE_KEY, JSON.stringify([...letters]));
    } catch {
      // Ignore storage errors
    }
  };

  let enabledLetters = loadEnabledLetters();
  const letterCheckboxes = new Map<string, HTMLInputElement>();

  for (const letter of allLetters) {
    const isAvailable = availableLetterSet.has(letter);

    const checkbox = document.createElement('input');
    checkbox.type = 'checkbox';
    checkbox.className = 'letter-checkbox';
    checkbox.id = `letter-${letter}`;
    checkbox.checked = enabledLetters.has(letter);
    checkbox.disabled = !isAvailable;

    const label = document.createElement('label');
    label.className = 'letter-label';
    label.htmlFor = `letter-${letter}`;
    label.textContent = letter;
    if (!isAvailable) {
      label.title = 'Coming soon';
    }

    letterGrid.appendChild(checkbox);
    letterGrid.appendChild(label);

    if (isAvailable) {
      letterCheckboxes.set(letter, checkbox);
    }
  }
  settingsModal.appendChild(letterGrid);

  // Settings buttons
  const settingsButtons = document.createElement('div');
  settingsButtons.className = 'settings-buttons';

  const selectAllBtn = document.createElement('button');
  selectAllBtn.className = 'settings-btn';
  selectAllBtn.type = 'button';
  selectAllBtn.textContent = 'Select All';

  const selectNoneBtn = document.createElement('button');
  selectNoneBtn.className = 'settings-btn';
  selectNoneBtn.type = 'button';
  selectNoneBtn.textContent = 'Select None';

  const doneBtn = document.createElement('button');
  doneBtn.className = 'settings-btn primary';
  doneBtn.type = 'button';
  doneBtn.textContent = 'Done';

  settingsButtons.appendChild(selectAllBtn);
  settingsButtons.appendChild(selectNoneBtn);
  settingsButtons.appendChild(doneBtn);
  settingsModal.appendChild(settingsButtons);

  settingsOverlay.appendChild(settingsModal);
  app.appendChild(settingsOverlay);

  const updateEnabledFromCheckboxes = (): void => {
    enabledLetters = new Set<string>();
    for (const [letter, checkbox] of letterCheckboxes) {
      if (checkbox.checked) {
        enabledLetters.add(letter);
      }
    }
  };

  const showError = (show: boolean): void => {
    if (show) {
      settingsError.classList.add('visible');
    } else {
      settingsError.classList.remove('visible');
    }
  };

  const openSettings = (): void => {
    // Reset checkboxes to current enabled state
    for (const [letter, checkbox] of letterCheckboxes) {
      checkbox.checked = enabledLetters.has(letter);
    }
    showError(false);
    settingsOverlay.classList.add('visible');
  };

  const closeSettings = (save: boolean): void => {
    if (save) {
      updateEnabledFromCheckboxes();
      if (enabledLetters.size === 0) {
        showError(true);
        return;
      }
      saveEnabledLetters(enabledLetters);
      // Update the game with new enabled letters
      if (game) {
        game.setEnabledLetters(enabledLetters);
      }
    }
    showError(false);
    settingsOverlay.classList.remove('visible');
  };

  settingsButton.addEventListener('click', (event) => {
    event.stopPropagation();
    openSettings();
  });

  selectAllBtn.addEventListener('click', (event) => {
    event.stopPropagation();
    for (const checkbox of letterCheckboxes.values()) {
      checkbox.checked = true;
    }
    showError(false);
  });

  selectNoneBtn.addEventListener('click', (event) => {
    event.stopPropagation();
    for (const checkbox of letterCheckboxes.values()) {
      checkbox.checked = false;
    }
  });

  doneBtn.addEventListener('click', (event) => {
    event.stopPropagation();
    closeSettings(true);
  });

  // Close on overlay click (but not modal click)
  settingsOverlay.addEventListener('click', (event) => {
    if (event.target === settingsOverlay) {
      closeSettings(false);
    }
  });

  // Close on Escape key
  document.addEventListener('keydown', (event) => {
    if (event.key === 'Escape' && settingsOverlay.classList.contains('visible')) {
      closeSettings(false);
    }
  });

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

  const letterDisplay = document.createElement('button');
  letterDisplay.className = 'letter-display';
  letterDisplay.type = 'button';
  letterDisplay.setAttribute('aria-label', 'Play letter sound');

  const itemTray = document.createElement('div');
  itemTray.className = 'item-tray letter-panel__item';

  letterPanel.appendChild(letterDisplay);
  letterPanel.appendChild(itemTray);
  app.appendChild(letterPanel);

  let currentLetter = '';
  const renderItemsForLetter = (letter: string) => {
    itemTray.replaceChildren();
    letterDisplay.textContent = letter;
    currentLetter = letter;
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

  // Set initial enabled letters from saved settings
  game.setEnabledLetters(enabledLetters);

  letterDisplay.addEventListener('click', (event) => {
    event.stopPropagation();
    if (!currentLetter) {
      return;
    }
    requestLetterSound(currentLetter);
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
