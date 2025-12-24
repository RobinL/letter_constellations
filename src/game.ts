import type { PointerPoint } from './input';
import { InputHandler } from './input';
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

const letterJsonModules = import.meta.glob('./assets/letters_json/*.json', {
  as: 'raw',
  eager: true,
});

const loadRandomPlotPoints = (): PlotPoint[] => {
  const letterJsons = Object.values(letterJsonModules) as string[];
  if (letterJsons.length === 0) {
    return [];
  }

  const raw = letterJsons[Math.floor(Math.random() * letterJsons.length)];
  try {
    return (JSON.parse(raw) as PlotPoint[])
      .map((point) => ({ order: point.order, x: point.x, y: point.y }))
      .sort((a, b) => a.order - b.order);
  } catch {
    return [];
  }
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
  private currentPath: PathPoint[] = [];
  private fadeSeconds = 9;
  private maxPoints = 600;
  private isDrawing = false;
  private currentMousePos = { x: 0, y: 0 };
  private plotBounds: PlotBounds = { minX: 0, maxX: 0, minY: 0, maxY: 0 };
  private scaledPlotPoints: PlotPoint[] = [];
  private lastPlotSize = { width: 0, height: 0 };
  private currentTargetIndex = 0;
  private dotRadius = 20;
  private hitRadiusScale = 1.5;

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
    return {
      isDrawing: this.isDrawing,
      mouseX: this.currentMousePos.x,
      mouseY: this.currentMousePos.y,
      trailPoints: [...this.currentPath],
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
      viewportHeight === this.lastPlotSize.height
    ) {
      return;
    }

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
    this.lastPlotSize = { width: viewportWidth, height: viewportHeight };
  }

  update(deltaTime: number): void {
    void deltaTime;
    if (this.currentPath.length === 0) {
      return;
    }

    const now = performance.now() / 1000;
    const cutoff = now - this.fadeSeconds;
    this.currentPath = this.currentPath.filter((point) => point.time >= cutoff);

    if (this.currentPath.length > this.maxPoints) {
      this.currentPath = this.currentPath.slice(-this.maxPoints);
    }
  }

  render(ctx: CanvasRenderingContext2D): void {
    ctx.save();
    ctx.setTransform(1, 0, 0, 1, 0, 0);
    ctx.clearRect(0, 0, ctx.canvas.width, ctx.canvas.height);
    ctx.restore();

    this.renderPlotLines(ctx);

    if (this.currentPath.length === 0) {
      return;
    }

    ctx.strokeStyle = 'rgba(200, 230, 255, 0.9)';
    ctx.lineWidth = 15;
    ctx.lineJoin = 'round';
    ctx.lineCap = 'round';

    ctx.beginPath();
    this.currentPath.forEach((point, index) => {
      if (index === 0) {
        ctx.moveTo(point.x, point.y);
      } else {
        ctx.lineTo(point.x, point.y);
      }
    });
    ctx.stroke();

  }

  private startPath(point: PointerPoint): void {
    this.isDrawing = true;
    this.currentMousePos = { x: point.x, y: point.y };
    this.currentPath = [{ x: point.x, y: point.y, time: point.time }];
    this.tryAdvanceTarget(point);
  }

  private extendPath(point: PointerPoint): void {
    this.currentMousePos = { x: point.x, y: point.y };
    this.currentPath.push({ x: point.x, y: point.y, time: point.time });
    this.tryAdvanceTarget(point);
  }

  private endPath(point: PointerPoint): void {
    this.currentMousePos = { x: point.x, y: point.y };
    this.currentPath.push({ x: point.x, y: point.y, time: point.time });
    this.tryAdvanceTarget(point);
    this.isDrawing = false;
  }

  private renderPlotLines(ctx: CanvasRenderingContext2D): void {
    if (this.scaledPlotPoints.length < 2) {
      return;
    }

    ctx.strokeStyle = 'rgba(255, 220, 0, 0.5)';
    ctx.lineWidth = 3;
    ctx.lineJoin = 'round';
    ctx.lineCap = 'round';

    ctx.beginPath();
    this.scaledPlotPoints.forEach((point, index) => {
      if (index === 0) {
        ctx.moveTo(point.x, point.y);
      } else {
        ctx.lineTo(point.x, point.y);
      }
    });
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
    const radius = this.dotRadius * this.hitRadiusScale;
    if (dx * dx + dy * dy <= radius * radius) {
      this.currentTargetIndex = Math.min(
        this.currentTargetIndex + 1,
        this.scaledPlotPoints.length
      );
    }
  }
}
