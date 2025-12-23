# Architecture Review - Letter Constellations

## Summary
The current codebase is a clean starter: a focused `AuroraRenderer` for WebGPU and a lightweight `main.ts` that owns setup and the render loop. This is a reasonable foundation for the upcoming letter-constellation game, but the gameplay features will quickly outgrow the single-file orchestration. A few small refactors now will keep the game logic isolated and make input/rendering easier to extend.

## Findings
- `src/main.ts` currently owns DOM setup, lifecycle, resize handling, and the frame loop. This will become a bottleneck as soon as we add gameplay state, star generation, and input handling.
- The overlay `#game-canvas` is present but has no 2D context setup or coordinate mapping. When drawing on a high-DPI canvas, we will need a consistent "CSS pixel" coordinate system for pointer input and drawing.
- Asset loading is embedded in `AuroraRenderer` (`background.png`). The game will likely need star sprites, letter templates, and sound; a central loader would avoid scattering fetch/bitmap logic.
- `AuroraRenderer` does not handle device loss or destruction, which may matter in a longer-running interactive game session.
- Minor cleanups: `src/counter.ts` and `src/typescript.svg` look unused; leaving them is harmless but they can be removed to reduce noise.

## Refactors To Do Before Adding Gameplay
1) Introduce a `Game` module/class
   - Owns game state, letter targets, and child-friendly rules.
   - Exposes `update(dt)` and `render(ctx)` for the 2D overlay.
2) Add a lightweight input handler
   - Centralize pointer events, hit-testing, and "snap-to-star" logic.
   - Provide a consistent coordinate transform based on DPR.
3) Centralize resize logic
   - A small `CanvasManager` that sets `width/height`, tracks DPR, and provides helpers to map screen -> canvas coordinates.
   - For the 2D overlay, set `ctx.setTransform(dpr, 0, 0, dpr, 0, 0)` on resize to draw in CSS pixels.
4) Create a simple asset loader
   - One place for `fetch`/`createImageBitmap` and future sounds/JSON.
   - Avoids pushing asset concerns into renderers or game state.
5) Render loop orchestration
   - Keep `AuroraRenderer.render()` and `Game.render(ctx)` both driven from a single `requestAnimationFrame` to avoid drift.
   - Inject delta time once, in one place.

## Optional Improvements (Nice-to-Have)
- Add device-lost handling in `AuroraRenderer` and a clean shutdown hook for future scene changes.
- Consider moving shader params into a small config file if we plan to tune aurora intensity or respond to gameplay.

## Conclusion
No major architectural blocker is present. The split between WebGPU background and 2D overlay is a good fit for the upcoming "draw constellations" gameplay. The main work before adding gameplay is to modularize `main.ts` into game state, input, and resize utilities so the feature work stays clean and testable.
