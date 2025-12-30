import type { PointerPoint } from './input';
import { InputHandler } from './input';
const letterModules = import.meta.glob('./assets/letters_json/*.json', {
  eager: true,
});

type PlotPoint = {
  order: number;
  x: number;
  y: number;
};

type PlotBounds = {
  minX: number;
  maxX: number;
  minY: number;
  maxY: number;
};

type StrokeSegment = {
  from: number;
  to: number;
};

type Point2D = {
  x: number;
  y: number;
};

type LetterPlot = {
  name: string;
  points: PlotPoint[];
};

const letterData = Object.entries(letterModules)
  .map(([path, mod]) => {
    const points = (mod as { default: PlotPoint[] }).default;
    if (!Array.isArray(points)) {
      return null;
    }
    const fileName = path.split('/').pop() ?? 'unknown';
    const name = fileName.replace(/\.json$/i, '');
    return { name, points };
  })
  .filter((entry): entry is LetterPlot => entry !== null);

// Export available letters for the settings UI
export const availableLetters = letterData.map((entry) => entry.name).sort();

const loadRandomPlotPoints = (enabledLetters?: Set<string>): { name: string; points: PlotPoint[] } => {
  if (letterData.length === 0) {
    return { name: 'unknown', points: [] };
  }

  // Filter to only enabled letters if provided
  let filteredData = letterData;
  if (enabledLetters && enabledLetters.size > 0) {
    filteredData = letterData.filter((entry) => enabledLetters.has(entry.name));
  }

  // Fallback to all letters if filter results in empty set
  if (filteredData.length === 0) {
    filteredData = letterData;
  }

  const selected = filteredData[Math.floor(Math.random() * filteredData.length)];
  console.info('Selected letter:', selected.name);
  return {
    name: selected.name,
    points: selected.points
      .map((point) => ({ order: point.order, x: point.x, y: point.y }))
      .sort((a, b) => a.order - b.order),
  };
};

const computeBounds = (points: PlotPoint[]): PlotBounds => {
  if (points.length === 0) {
    return { minX: 0, maxX: 0, minY: 0, maxY: 0 };
  }

  let minX = points[0].x;
  let maxX = points[0].x;
  let minY = points[0].y;
  let maxY = points[0].y;

  for (const point of points) {
    if (point.x < minX) minX = point.x;
    if (point.x > maxX) maxX = point.x;
    if (point.y < minY) minY = point.y;
    if (point.y > maxY) maxY = point.y;
  }

  return { minX, maxX, minY, maxY };
};

const PEN_UP_DISTANCE_MULTIPLIER = 3.5;

const computeMedian = (values: number[]): number => {
  const sorted = [...values].sort((a, b) => a - b);
  const mid = Math.floor(sorted.length / 2);
  if (sorted.length % 2 === 0) {
    return (sorted[mid - 1] + sorted[mid]) / 2;
  }
  return sorted[mid];
};

const computePenUpThreshold = (points: PlotPoint[]): number => {
  if (points.length < 2) {
    return Number.POSITIVE_INFINITY;
  }
  const distances: number[] = [];
  for (let i = 0; i < points.length - 1; i += 1) {
    const dx = points[i + 1].x - points[i].x;
    const dy = points[i + 1].y - points[i].y;
    distances.push(Math.hypot(dx, dy));
  }
  const nonZeroDistances = distances.filter((distance) => distance > 0);
  const sample = nonZeroDistances.length > 0 ? nonZeroDistances : distances;
  if (sample.length === 0) {
    return Number.POSITIVE_INFINITY;
  }
  const median = computeMedian(sample);
  return median * PEN_UP_DISTANCE_MULTIPLIER;
};

const buildStrokeSegments = (points: PlotPoint[], threshold: number): StrokeSegment[] => {
  const segments: StrokeSegment[] = [];
  if (points.length < 2) {
    return segments;
  }
  for (let i = 0; i < points.length - 1; i += 1) {
    const dx = points[i + 1].x - points[i].x;
    const dy = points[i + 1].y - points[i].y;
    if (Math.hypot(dx, dy) > threshold) {
      continue;
    }
    segments.push({ from: i, to: i + 1 });
  }
  return segments;
};

export type PathPoint = {
  x: number;
  y: number;
  time: number;
};

type GameCallbacks = {
  onPointHit?: () => void;
  onLetterChange?: (letter: string) => void;
};

export class Game {
  private plotPoints: PlotPoint[] = [];
  private paths: PathPoint[][] = [];
  private currentPath: PathPoint[] | null = null;
  private fadeSeconds = 9;
  private maxPoints = 600;
  private isDrawing = false;
  private currentMousePos = { x: 0, y: 0 };
  private plotBounds: PlotBounds = { minX: 0, maxX: 0, minY: 0, maxY: 0 };
  private scaledPlotPoints: PlotPoint[] = [];
  private lastPlotSize = { width: 0, height: 0 };
  private needsRescale = true;
  private currentTargetIndex = 0;
  private dotRadius = 20;
  private hitRadiusScale = 1.5;
  private penUpDistanceThreshold = Number.POSITIVE_INFINITY;
  private lineSegmentIndex = 0;
  private lineSegmentT = 0;
  private linePauseRemaining = 0;
  private readonly lineSegmentSeconds = 0.4;
  private readonly lineLoopPauseSeconds = 0.4;
  private completionMessageUntil = 0;
  private pendingLetterReset = false;
  private readonly completionMessageSeconds = 2;
  private completionGraceUntil = 0;
  private readonly completionGraceSeconds = 0.25;
  private callbacks: GameCallbacks;
  private currentLetterName = 'unknown';
  private enabledLetters: Set<string> | null = null;

  constructor(input: InputHandler, callbacks: GameCallbacks = {}) {
    const selection = loadRandomPlotPoints();
    this.plotPoints = selection.points;
    this.plotBounds = computeBounds(this.plotPoints);
    this.callbacks = callbacks;
    this.currentLetterName = selection.name;
    this.callbacks.onLetterChange?.(this.currentLetterName);
    input.setCallbacks({
      onStart: (point) => this.startPath(point),
      onMove: (point) => this.extendPath(point),
      onEnd: (point) => this.endPath(point),
    });
  }

  setEnabledLetters(letters: Set<string> | null): void {
    this.enabledLetters = letters;
  }

  // Expose state for sparkle renderer
  getDrawingState(): {
    isDrawing: boolean;
    mouseX: number;
    mouseY: number;
    trailPoints: PathPoint[];
  } {
    const trailPoints = this.paths.flatMap((path) => path);
    return {
      isDrawing: this.isDrawing,
      mouseX: this.currentMousePos.x,
      mouseY: this.currentMousePos.y,
      trailPoints,
    };
  }

  getDotState(): {
    dots: PlotPoint[];
    currentIndex: number;
    radius: number;
  } {
    return {
      dots: this.scaledPlotPoints,
      currentIndex: this.currentTargetIndex,
      radius: this.dotRadius,
    };
  }

  clearUserPaths(): void {
    this.paths = [];
    this.currentPath = null;
    this.isDrawing = false;
  }

  resetCurrentLetter(): void {
    this.clearUserPaths();
    this.currentTargetIndex = 0;
    this.lineSegmentIndex = 0;
    this.lineSegmentT = 0;
    this.linePauseRemaining = 0;
    this.pendingLetterReset = false;
    this.completionMessageUntil = 0;
    this.completionGraceUntil = 0;
  }

  setViewportSize(width: number, height: number): void {
    if (this.plotPoints.length === 0) {
      return;
    }

    const viewportWidth = Math.max(1, width);
    const viewportHeight = Math.max(1, height);

    if (
      viewportWidth === this.lastPlotSize.width &&
      viewportHeight === this.lastPlotSize.height &&
      !this.needsRescale
    ) {
      return;
    }

    this.lastPlotSize = { width: viewportWidth, height: viewportHeight };
    this.needsRescale = false;

    const boundsWidth = this.plotBounds.maxX - this.plotBounds.minX;
    const boundsHeight = this.plotBounds.maxY - this.plotBounds.minY;
    if (boundsWidth <= 0 || boundsHeight <= 0) {
      return;
    }

    const targetWidth = viewportWidth * 0.5;
    const targetHeight = viewportHeight * 0.5;
    const scale = Math.min(targetWidth / boundsWidth, targetHeight / boundsHeight);
    const offsetX =
      (viewportWidth - boundsWidth * scale) / 2 - this.plotBounds.minX * scale;
    const offsetY =
      (viewportHeight - boundsHeight * scale) / 2 - this.plotBounds.minY * scale;

    this.scaledPlotPoints = this.plotPoints.map((point) => ({
      order: point.order,
      x: point.x * scale + offsetX,
      y: point.y * scale + offsetY,
    }));
    this.penUpDistanceThreshold = computePenUpThreshold(this.scaledPlotPoints);
  }

  update(deltaTime: number): void {
    this.advancePlotAnimation(deltaTime);

    const now = performance.now() / 1000;
    if (this.pendingLetterReset) {
      this.lockInputAfterCompletion(now);
    }

    if (this.paths.length > 0) {
      const cutoff = now - this.fadeSeconds;
      for (const path of this.paths) {
        let keepIndex = 0;
        while (keepIndex < path.length && path[keepIndex].time < cutoff) {
          keepIndex += 1;
        }
        if (keepIndex > 0) {
          path.splice(0, keepIndex);
        }
        if (path.length > this.maxPoints) {
          path.splice(0, path.length - this.maxPoints);
        }
      }
      this.paths = this.paths.filter((path) => path.length > 0);
    }

    if (this.pendingLetterReset && now >= this.completionMessageUntil) {
      this.resetForNewLetter();
      this.pendingLetterReset = false;
      this.completionMessageUntil = 0;
      this.completionGraceUntil = 0;
    }
  }

  render(ctx: CanvasRenderingContext2D): void {
    ctx.save();
    ctx.setTransform(1, 0, 0, 1, 0, 0);
    ctx.clearRect(0, 0, ctx.canvas.width, ctx.canvas.height);
    ctx.restore();

    this.renderPlotLines(ctx);

    if (this.paths.length > 0) {
      ctx.strokeStyle = 'rgba(200, 230, 255, 0.9)';
      ctx.lineWidth = 15;
      ctx.lineJoin = 'round';
      ctx.lineCap = 'round';

      for (const path of this.paths) {
        if (path.length === 0) {
          continue;
        }
        ctx.beginPath();
        path.forEach((point, index) => {
          if (index === 0) {
            ctx.moveTo(point.x, point.y);
          } else {
            ctx.lineTo(point.x, point.y);
          }
        });
        ctx.stroke();
      }
    }

    this.renderCompletionMessage(ctx);

  }

  private startPath(point: PointerPoint): void {
    if (this.pendingLetterReset) {
      return;
    }
    this.isDrawing = true;
    this.currentMousePos = { x: point.x, y: point.y };
    this.currentPath = [{ x: point.x, y: point.y, time: point.time }];
    this.paths.push(this.currentPath);
    this.tryAdvanceTarget(point);
  }

  private extendPath(point: PointerPoint): void {
    this.currentMousePos = { x: point.x, y: point.y };
    if (this.pendingLetterReset && !this.isCompletionGraceActive(point.time)) {
      this.lockInputAfterCompletion(point.time);
      return;
    }
    if (!this.currentPath) {
      return;
    }
    this.currentPath.push({ x: point.x, y: point.y, time: point.time });
    this.tryAdvanceTarget(point);
  }

  private endPath(point: PointerPoint): void {
    this.currentMousePos = { x: point.x, y: point.y };
    if (this.pendingLetterReset && !this.isCompletionGraceActive(point.time)) {
      this.lockInputAfterCompletion(point.time);
      return;
    }
    if (this.currentPath) {
      this.currentPath.push({ x: point.x, y: point.y, time: point.time });
    }
    this.tryAdvanceTarget(point);
    this.isDrawing = false;
    this.currentPath = null;
  }

  private isCompletionGraceActive(now: number): boolean {
    return this.pendingLetterReset && now < this.completionGraceUntil;
  }

  private lockInputAfterCompletion(now: number): void {
    if (!this.pendingLetterReset || now < this.completionGraceUntil) {
      return;
    }
    this.isDrawing = false;
    this.currentPath = null;
  }

  private renderPlotLines(ctx: CanvasRenderingContext2D): void {
    if (this.scaledPlotPoints.length < 2) {
      return;
    }

    ctx.save();
    ctx.strokeStyle = 'rgba(255, 220, 0, 0.5)';
    ctx.lineWidth = 9;
    ctx.lineJoin = 'round';
    ctx.lineCap = 'round';

    const startIndex = Math.min(
      Math.max(0, this.currentTargetIndex - 1),
      this.scaledPlotPoints.length - 1
    );
    const points = this.scaledPlotPoints.slice(startIndex);
    const segments = buildStrokeSegments(points, this.penUpDistanceThreshold);
    if (segments.length === 0) {
      ctx.restore();
      return;
    }

    if (this.linePauseRemaining > 0) {
      this.strokeLineSegments(ctx, points, segments, segments.length - 1);
      this.drawLineNib(ctx, points, segments, segments.length - 1, 1);
    } else {
      const clampedIndex = Math.min(this.lineSegmentIndex, segments.length - 1);
      this.strokeLineSegments(ctx, points, segments, clampedIndex - 1);
      this.strokePartialLineSegment(ctx, points, segments, clampedIndex, this.lineSegmentT);
      this.drawLineNib(ctx, points, segments, clampedIndex, this.lineSegmentT);
    }
    ctx.restore();
  }

  private strokeLineSegments(
    ctx: CanvasRenderingContext2D,
    points: PlotPoint[],
    segments: StrokeSegment[],
    lastIndex: number
  ): void {
    if (lastIndex < 0) {
      return;
    }
    const cappedIndex = Math.min(lastIndex, segments.length - 1);
    for (let i = 0; i <= cappedIndex; i += 1) {
      const segment = segments[i];
      const from = points[segment.from];
      const to = points[segment.to];
      this.strokeLineSegment(ctx, from, to);
    }
  }

  private strokePartialLineSegment(
    ctx: CanvasRenderingContext2D,
    points: PlotPoint[],
    segments: StrokeSegment[],
    index: number,
    t: number
  ): void {
    if (index < 0 || index >= segments.length) {
      return;
    }
    const segment = segments[index];
    const from = points[segment.from];
    const to = points[segment.to];
    const x = from.x + (to.x - from.x) * t;
    const y = from.y + (to.y - from.y) * t;
    this.strokeLineSegment(ctx, from, { x, y });
  }

  private strokeLineSegment(
    ctx: CanvasRenderingContext2D,
    from: Point2D,
    to: Point2D
  ): void {
    ctx.beginPath();
    ctx.moveTo(from.x, from.y);
    ctx.lineTo(to.x, to.y);
    ctx.stroke();
  }

  private drawLineNib(
    ctx: CanvasRenderingContext2D,
    points: PlotPoint[],
    segments: StrokeSegment[],
    index: number,
    t: number
  ): void {
    if (index < 0 || index >= segments.length) {
      return;
    }
    const segment = segments[index];
    const from = points[segment.from];
    const to = points[segment.to];
    const clampedT = Math.max(0, Math.min(1, t));
    const x = from.x + (to.x - from.x) * clampedT;
    const y = from.y + (to.y - from.y) * clampedT;
    const nibRadius = Math.max(5, ctx.lineWidth * 0.7);
    ctx.fillStyle = '#ffe000';
    ctx.beginPath();
    ctx.arc(x, y, nibRadius, 0, Math.PI * 2);
    ctx.fill();
  }

  private tryAdvanceTarget(point: PointerPoint): void {
    if (!this.isDrawing || this.scaledPlotPoints.length === 0) {
      return;
    }

    if (this.currentTargetIndex >= this.scaledPlotPoints.length) {
      return;
    }

    const target = this.scaledPlotPoints[this.currentTargetIndex];
    const dx = point.x - target.x;
    const dy = point.y - target.y;
    const radius = this.getTargetDotRadius() * this.hitRadiusScale;
    if (dx * dx + dy * dy <= radius * radius) {
      this.currentTargetIndex = Math.min(
        this.currentTargetIndex + 1,
        this.scaledPlotPoints.length
      );
      this.callbacks.onPointHit?.();
      this.lineSegmentIndex = 0;
      this.lineSegmentT = 0;
      this.linePauseRemaining = 0;
      if (this.currentTargetIndex >= this.scaledPlotPoints.length) {
        this.handleLetterComplete();
      }
    }
  }

  private getTargetDotRadius(): number {
    // Keep in sync with size scaling in sparkle-renderer.ts.
    const baseRadius = Math.max(10, this.dotRadius * 1.3);
    return baseRadius * 1.45;
  }

  private advancePlotAnimation(deltaTime: number): void {
    if (deltaTime <= 0 || this.scaledPlotPoints.length < 2) {
      return;
    }

    const startIndex = Math.min(
      Math.max(0, this.currentTargetIndex - 1),
      this.scaledPlotPoints.length - 1
    );
    const points = this.scaledPlotPoints.slice(startIndex);
    const segments = buildStrokeSegments(points, this.penUpDistanceThreshold);
    const segmentCount = segments.length;
    if (segmentCount === 0) {
      return;
    }
    const segmentLengths = segments.map((segment) => {
      const from = points[segment.from];
      const to = points[segment.to];
      return Math.hypot(to.x - from.x, to.y - from.y);
    });
    const nonZeroLengths = segmentLengths.filter((length) => length > 0);
    if (nonZeroLengths.length === 0) {
      return;
    }
    const baseLength = computeMedian(nonZeroLengths);
    const lineSpeed = baseLength / this.lineSegmentSeconds;
    this.lineSegmentIndex = Math.min(this.lineSegmentIndex, segmentCount - 1);

    if (this.linePauseRemaining > 0) {
      this.linePauseRemaining = Math.max(0, this.linePauseRemaining - deltaTime);
      if (this.linePauseRemaining > 0) {
        return;
      }
      this.lineSegmentIndex = 0;
      this.lineSegmentT = 0;
    }

    let remainingDistance = deltaTime * lineSpeed;
    while (remainingDistance > 0) {
      if (this.lineSegmentIndex >= segmentCount) {
        this.linePauseRemaining = this.lineLoopPauseSeconds;
        this.lineSegmentIndex = 0;
        this.lineSegmentT = 0;
        break;
      }
      const segmentLength = segmentLengths[this.lineSegmentIndex] ?? 0;
      if (segmentLength <= 0) {
        this.lineSegmentIndex += 1;
        this.lineSegmentT = 0;
        continue;
      }
      const distanceToEnd = segmentLength * (1 - this.lineSegmentT);
      if (remainingDistance < distanceToEnd) {
        this.lineSegmentT += remainingDistance / segmentLength;
        remainingDistance = 0;
      } else {
        remainingDistance -= distanceToEnd;
        this.lineSegmentIndex += 1;
        this.lineSegmentT = 0;
      }
    }
  }

  private handleLetterComplete(): void {
    if (this.pendingLetterReset) {
      return;
    }
    const now = performance.now() / 1000;
    this.completionMessageUntil = now + this.completionMessageSeconds;
    this.completionGraceUntil = now + this.completionGraceSeconds;
    this.pendingLetterReset = true;
  }

  private resetForNewLetter(): void {
    const selection = loadRandomPlotPoints(this.enabledLetters ?? undefined);
    this.plotPoints = selection.points;
    this.plotBounds = computeBounds(this.plotPoints);
    this.currentLetterName = selection.name;
    this.callbacks.onLetterChange?.(this.currentLetterName);
    this.scaledPlotPoints = [];
    this.currentTargetIndex = 0;
    this.lineSegmentIndex = 0;
    this.lineSegmentT = 0;
    this.linePauseRemaining = 0;
    this.paths = [];
    this.currentPath = null;
    this.isDrawing = false;
    this.needsRescale = true;
    this.completionGraceUntil = 0;
    if (this.lastPlotSize.width > 0 && this.lastPlotSize.height > 0) {
      this.setViewportSize(this.lastPlotSize.width, this.lastPlotSize.height);
    }
  }

  private renderCompletionMessage(ctx: CanvasRenderingContext2D): void {
    const now = performance.now() / 1000;
    if (now >= this.completionMessageUntil) {
      return;
    }

    const width = ctx.canvas.width / (window.devicePixelRatio || 1);
    const height = ctx.canvas.height / (window.devicePixelRatio || 1);

    ctx.save();
    ctx.textAlign = 'center';
    ctx.textBaseline = 'middle';
    ctx.fillStyle = 'rgba(255, 244, 210, 0.95)';
    ctx.strokeStyle = 'rgba(0, 0, 0, 0.45)';
    ctx.lineWidth = 6;
    ctx.font = '700 48px system-ui, -apple-system, sans-serif';
    const message = 'Well done!';
    ctx.strokeText(message, width / 2, height * 0.45);
    ctx.fillText(message, width / 2, height * 0.45);
    ctx.restore();
  }
}
