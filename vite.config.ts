import { defineConfig } from 'vite';

const base = '/letter_constellations/';

export default defineConfig({
    base,
    server: {
        port: 3000,
        open: true
    },
    build: {
        outDir: 'dist'
    }
});
