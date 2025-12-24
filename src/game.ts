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

const letterData = Object.values(letterModules)
  .map((mod) => (mod as { default: PlotPoint[] }).default)
  .filter((points): points is PlotPoint[] => Array.isArray(points));

const loadRandomPlotPoints = (): PlotPoint[] => {
  if (letterData.length === 0) {
    return [];
  }

  const data = letterData[Math.floor(Math.random() * letterData.length)] as PlotPoint[];
  return data
    .map((point) => ({ order: point.order, x: point.x, y: point.y }))
    .sort((a, b) => a.order - b.order);
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

export type PathPoint = {
  x: number;
  y: number;
  time: number;
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
  private lineSegmentIndex = 0;
  private lineSegmentT = 0;
  private linePauseRemaining = 0;
  private readonly lineSegmentSeconds = 0.123;
  private readonly lineLoopPauseSeconds = 0.4;
  private completionMessageUntil = 0;
  private pendingLetterReset = false;
  private readonly completionMessageSeconds = 2;

  constructor(input: InputHandler) {
    this.plotPoints = loadRandomPlotPoints();
    this.plotBounds = computeBounds(this.plotPoints);
    input.setCallbacks({
      onStart: (point) => this.startPath(point),
      onMove: (point) => this.extendPath(point),
      onEnd: (point) => this.endPath(point),
    });
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
  }

  update(deltaTime: number): void {
    this.advancePlotAnimation(deltaTime);

    const now = performance.now() / 1000;

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
    if (this.pendingLetterReset) {
      return;
    }
    this.currentMousePos = { x: point.x, y: point.y };
    if (!this.currentPath) {
      return;
    }
    this.currentPath.push({ x: point.x, y: point.y, time: point.time });
    this.tryAdvanceTarget(point);
  }

  private endPath(point: PointerPoint): void {
    if (this.pendingLetterReset) {
      return;
    }
    this.currentMousePos = { x: point.x, y: point.y };
    if (this.currentPath) {
      this.currentPath.push({ x: point.x, y: point.y, time: point.time });
    }
    this.tryAdvanceTarget(point);
    this.isDrawing = false;
    this.currentPath = null;
  }

  private renderPlotLines(ctx: CanvasRenderingContext2D): void {
    if (this.scaledPlotPoints.length < 2) {
      return;
    }

    ctx.strokeStyle = 'rgba(255, 220, 0, 0.5)';
    ctx.lineWidth = 9;
    ctx.lineJoin = 'round';
    ctx.lineCap = 'round';

    ctx.beginPath();
    const startIndex = Math.min(
      Math.max(0, this.currentTargetIndex - 1),
      this.scaledPlotPoints.length - 1
    );
    const points = this.scaledPlotPoints.slice(startIndex);
    if (points.length < 2) {
      return;
    }
    ctx.moveTo(points[0].x, points[0].y);

    if (this.linePauseRemaining > 0) {
      for (let i = 1; i < points.length; i += 1) {
        ctx.lineTo(points[i].x, points[i].y);
      }
    } else {
      const lastIndex = Math.min(this.lineSegmentIndex, points.length - 1);
      for (let i = 1; i <= lastIndex; i += 1) {
        ctx.lineTo(points[i].x, points[i].y);
      }

      const nextIndex = Math.min(this.lineSegmentIndex + 1, points.length - 1);
      if (nextIndex > this.lineSegmentIndex) {
        const from = points[this.lineSegmentIndex];
        const to = points[nextIndex];
        const x = from.x + (to.x - from.x) * this.lineSegmentT;
        const y = from.y + (to.y - from.y) * this.lineSegmentT;
        ctx.lineTo(x, y);
      }
    }
    ctx.stroke();
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

    const remainingPoints = Math.max(
      0,
      this.scaledPlotPoints.length - Math.max(0, this.currentTargetIndex - 1)
    );
    if (remainingPoints < 2) {
      return;
    }
    const segmentCount = remainingPoints - 1;
    this.lineSegmentIndex = Math.min(this.lineSegmentIndex, segmentCount - 1);

    if (this.linePauseRemaining > 0) {
      this.linePauseRemaining = Math.max(0, this.linePauseRemaining - deltaTime);
      if (this.linePauseRemaining > 0) {
        return;
      }
      this.lineSegmentIndex = 0;
      this.lineSegmentT = 0;
    }

    this.lineSegmentT += deltaTime / this.lineSegmentSeconds;
    while (this.lineSegmentT >= 1) {
      this.lineSegmentT -= 1;
      this.lineSegmentIndex += 1;

      if (this.lineSegmentIndex >= segmentCount) {
        this.linePauseRemaining = this.lineLoopPauseSeconds;
        this.lineSegmentIndex = 0;
        this.lineSegmentT = 0;
        break;
      }
    }
  }

  private handleLetterComplete(): void {
    if (this.pendingLetterReset) {
      return;
    }
    const now = performance.now() / 1000;
    this.completionMessageUntil = now + this.completionMessageSeconds;
    this.pendingLetterReset = true;
    this.isDrawing = false;
    this.currentPath = null;
  }

  private resetForNewLetter(): void {
    this.plotPoints = loadRandomPlotPoints();
    this.plotBounds = computeBounds(this.plotPoints);
    this.scaledPlotPoints = [];
    this.currentTargetIndex = 0;
    this.lineSegmentIndex = 0;
    this.lineSegmentT = 0;
    this.linePauseRemaining = 0;
    this.paths = [];
    this.currentPath = null;
    this.isDrawing = false;
    this.needsRescale = true;
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
