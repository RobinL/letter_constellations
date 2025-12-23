export type CanvasSize = {
  width: number;
  height: number;
  dpr: number;
};

export class CanvasManager {
  private size: CanvasSize;

  constructor(
    private auroraCanvas: HTMLCanvasElement,
    private gameCanvas: HTMLCanvasElement
  ) {
    this.size = { width: 0, height: 0, dpr: 1 };
  }

  resize(): CanvasSize {
    const width = window.innerWidth;
    const height = window.innerHeight;
    const dpr = Math.max(1, window.devicePixelRatio || 1);

    this.applySize(this.auroraCanvas, width, height, dpr);
    this.applySize(this.gameCanvas, width, height, dpr);

    this.size = { width, height, dpr };
    return this.size;
  }

  configureGameContext(ctx: CanvasRenderingContext2D): void {
    const { dpr } = this.size;
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    ctx.imageSmoothingEnabled = true;
  }

  getPointFromEvent(
    canvas: HTMLCanvasElement,
    event: PointerEvent
  ): { x: number; y: number } {
    const rect = canvas.getBoundingClientRect();
    return {
      x: event.clientX - rect.left,
      y: event.clientY - rect.top,
    };
  }

  getSize(): CanvasSize {
    return this.size;
  }

  private applySize(
    canvas: HTMLCanvasElement,
    width: number,
    height: number,
    dpr: number
  ): void {
    canvas.width = Math.max(1, Math.floor(width * dpr));
    canvas.height = Math.max(1, Math.floor(height * dpr));
    canvas.style.width = `${width}px`;
    canvas.style.height = `${height}px`;
  }
}
