export type QualityLevel = 'low' | 'medium' | 'high' | 'ultra';

export type QualitySettings = {
    auroraScale: number;
    sparkleScale: number;
    level: QualityLevel;
};

const QUALITY_PRESETS: Record<QualityLevel, QualitySettings> = {
    low: { auroraScale: 0.33, sparkleScale: 0.5, level: 'low' },
    medium: { auroraScale: 0.5, sparkleScale: 0.75, level: 'medium' },
    high: { auroraScale: 0.75, sparkleScale: 1.0, level: 'high' },
    ultra: { auroraScale: 1.0, sparkleScale: 1.0, level: 'ultra' },
};

export class PerformanceMonitor {
    private frameTimes: number[] = [];
    private lastFrameTime = performance.now();
    private readonly sampleSize = 90; // 1.5 seconds at 60fps
    private readonly stabilizationFrames = 60; // Wait before adjusting
    private frameCount = 0;
    private currentQuality: QualityLevel;
    private onQualityChange?: (settings: QualitySettings) => void;
    private lastAdjustmentFrame = 0;
    private readonly adjustmentCooldown = 120; // Frames between adjustments

    constructor(initialQuality?: QualityLevel) {
        this.currentQuality = initialQuality ?? this.detectInitialQuality();
    }

    private detectInitialQuality(): QualityLevel {
        // Use device hints for initial quality estimate
        const isMobile = /iPhone|iPad|iPod|Android/i.test(navigator.userAgent);
        const isLowEndMobile = /iPhone [5-8]|iPad (Mini|Air(?! [3-9]))|Android [4-7]/i.test(
            navigator.userAgent
        );
        const memory = (navigator as { deviceMemory?: number }).deviceMemory;
        const cores = navigator.hardwareConcurrency || 2;

        if (isLowEndMobile) {
            return 'low';
        }

        if (isMobile) {
            // Modern mobile devices can handle medium quality
            return memory && memory >= 4 ? 'medium' : 'low';
        }

        // Desktop heuristics
        if (memory && memory >= 8 && cores >= 8) {
            return 'high';
        }

        if (memory && memory >= 4 && cores >= 4) {
            return 'medium';
        }

        return 'medium';
    }

    setOnQualityChange(callback: (settings: QualitySettings) => void): void {
        this.onQualityChange = callback;
    }

    recordFrame(): void {
        const now = performance.now();
        const frameTime = now - this.lastFrameTime;
        this.lastFrameTime = now;
        this.frameCount++;

        // Ignore extremely long frames (tab was backgrounded, etc.)
        if (frameTime > 100) {
            return;
        }

        this.frameTimes.push(frameTime);
        if (this.frameTimes.length > this.sampleSize) {
            this.frameTimes.shift();
        }

        // Check for quality adjustment periodically
        if (
            this.frameCount > this.stabilizationFrames &&
            this.frameCount - this.lastAdjustmentFrame > this.adjustmentCooldown &&
            this.frameTimes.length >= this.sampleSize / 2
        ) {
            this.maybeAdjustQuality();
        }
    }

    private getAverageFPS(): number {
        if (this.frameTimes.length === 0) return 60;
        const avgFrameTime =
            this.frameTimes.reduce((a, b) => a + b, 0) / this.frameTimes.length;
        return 1000 / avgFrameTime;
    }

    private getPercentileFPS(percentile: number): number {
        if (this.frameTimes.length === 0) return 60;
        const sorted = [...this.frameTimes].sort((a, b) => b - a); // Worst first
        const index = Math.floor(sorted.length * (1 - percentile / 100));
        return 1000 / sorted[index];
    }

    private maybeAdjustQuality(): void {
        const avgFPS = this.getAverageFPS();
        const lowPercentileFPS = this.getPercentileFPS(5); // 5th percentile (worst frames)

        const levels: QualityLevel[] = ['low', 'medium', 'high', 'ultra'];
        const currentIndex = levels.indexOf(this.currentQuality);

        let newQuality = this.currentQuality;

        // Downgrade if performance is poor
        if (lowPercentileFPS < 25 || avgFPS < 40) {
            if (currentIndex > 0) {
                newQuality = levels[currentIndex - 1];
            }
        }
        // Upgrade if performance is excellent and sustained
        else if (avgFPS > 58 && lowPercentileFPS > 50) {
            if (currentIndex < levels.length - 1) {
                newQuality = levels[currentIndex + 1];
            }
        }

        if (newQuality !== this.currentQuality) {
            console.info(
                `Quality adjusted: ${this.currentQuality} → ${newQuality} (avg: ${avgFPS.toFixed(1)} fps, 5th%: ${lowPercentileFPS.toFixed(1)} fps)`
            );
            this.currentQuality = newQuality;
            this.lastAdjustmentFrame = this.frameCount;
            this.frameTimes = []; // Reset samples after adjustment
            this.onQualityChange?.(this.getQualitySettings());
        }
    }

    getQualitySettings(): QualitySettings {
        return QUALITY_PRESETS[this.currentQuality];
    }

    getCurrentLevel(): QualityLevel {
        return this.currentQuality;
    }

    // Allow manual override
    setQuality(level: QualityLevel): void {
        if (level !== this.currentQuality) {
            this.currentQuality = level;
            this.lastAdjustmentFrame = this.frameCount;
            this.frameTimes = [];
            this.onQualityChange?.(this.getQualitySettings());
        }
    }

    getDebugInfo(): string {
        const avgFPS = this.getAverageFPS();
        const lowFPS = this.getPercentileFPS(5);
        return `${this.currentQuality} | ${avgFPS.toFixed(0)} fps (5%: ${lowFPS.toFixed(0)})`;
    }
}
