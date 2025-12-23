import type { PointerPoint } from './input';
import { InputHandler } from './input';

export type PathPoint = {
  x: number;
  y: number;
  time: number;
};

export class Game {
  private currentPath: PathPoint[] = [];
  private fadeSeconds = 3;
  private maxPoints = 600;
  private isDrawing = false;
  private currentMousePos = { x: 0, y: 0 };

  constructor(input: InputHandler) {
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

    if (this.currentPath.length === 0) {
      return;
    }

    ctx.strokeStyle = 'rgba(200, 230, 255, 0.9)';
    ctx.lineWidth = 3;
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

    for (const point of this.currentPath) {
      this.drawStar(ctx, point.x, point.y);
    }
  }

  private startPath(point: PointerPoint): void {
    this.isDrawing = true;
    this.currentMousePos = { x: point.x, y: point.y };
    this.currentPath = [{ x: point.x, y: point.y, time: point.time }];
  }

  private extendPath(point: PointerPoint): void {
    this.currentMousePos = { x: point.x, y: point.y };
    this.currentPath.push({ x: point.x, y: point.y, time: point.time });
  }

  private endPath(point: PointerPoint): void {
    this.isDrawing = false;
    this.currentMousePos = { x: point.x, y: point.y };
    this.currentPath.push({ x: point.x, y: point.y, time: point.time });
  }

  private drawStar(ctx: CanvasRenderingContext2D, x: number, y: number): void {
    ctx.fillStyle = 'rgba(255, 255, 255, 0.9)';
    ctx.beginPath();
    ctx.arc(x, y, 2.5, 0, Math.PI * 2);
    ctx.fill();
  }
}
