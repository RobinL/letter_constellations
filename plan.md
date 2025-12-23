Aim of the app

A browser-based “Letter Constellations” experience: a hand-painted Iceland/northern-lights scene as a backdrop, with a high-end animated sky (twinkling stars + ribboning aurora). Eventually, it becomes a simple children’s game where the child draws star constellations that form letters.

Recommended implementation
	•	App shell: statically hosted Vite + TypeScript.
	•	Rendering: WebGPU for the sky effects (stars + aurora) and any post (soft glow/bloom). Keep the painted background as a texture layer.
	•	Performance: render the aurora/glow at a slightly reduced internal resolution and composite over the full-res background.