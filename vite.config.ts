import { defineConfig } from 'vite';
import { imagetools } from 'vite-imagetools';

const base = '/letter_constellations/';

export default defineConfig({
    base,
    plugins: [
        imagetools({
            defaultDirectives: (url) => {
                // Keep background images at native resolution with higher quality
                if (url.searchParams.has('background') || url.pathname.includes('background')) {
                    return new URLSearchParams({
                        format: 'webp',
                        quality: '80',
                        as: 'url'
                    });
                }
                // All other images use standard settings
                return new URLSearchParams({
                    format: 'webp',
                    w: '1600',
                    quality: '60',
                    as: 'url'
                });
            }
        })
    ],
    server: {
        port: 3000,
        open: true
    },
    build: {
        outDir: 'dist'
    }
});
