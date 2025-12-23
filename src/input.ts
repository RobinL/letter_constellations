export type PointerPoint = {
  x: number;
  y: number;
  id: number;
  time: number;
};

export type PointerCallbacks = {
  onStart?: (point: PointerPoint) => void;
  onMove?: (point: PointerPoint) => void;
  onEnd?: (point: PointerPoint) => void;
};

export class InputHandler {
  private callbacks: PointerCallbacks = {};
  private activePointerId: number | null = null;

  constructor(
    private canvas: HTMLCanvasElement,
    private getPoint: (event: PointerEvent) => { x: number; y: number }
  ) {
    this.onPointerDown = this.onPointerDown.bind(this);
    this.onPointerMove = this.onPointerMove.bind(this);
    this.onPointerUp = this.onPointerUp.bind(this);

    this.canvas.addEventListener('pointerdown', this.onPointerDown);
    this.canvas.addEventListener('pointermove', this.onPointerMove);
    this.canvas.addEventListener('pointerup', this.onPointerUp);
    this.canvas.addEventListener('pointercancel', this.onPointerUp);
    this.canvas.addEventListener('pointerleave', this.onPointerUp);
  }

  setCallbacks(callbacks: PointerCallbacks): void {
    this.callbacks = callbacks;
  }

  dispose(): void {
    this.canvas.removeEventListener('pointerdown', this.onPointerDown);
    this.canvas.removeEventListener('pointermove', this.onPointerMove);
    this.canvas.removeEventListener('pointerup', this.onPointerUp);
    this.canvas.removeEventListener('pointercancel', this.onPointerUp);
    this.canvas.removeEventListener('pointerleave', this.onPointerUp);
  }

  private onPointerDown(event: PointerEvent): void {
    if (this.activePointerId !== null) {
      return;
    }
    this.activePointerId = event.pointerId;
    this.canvas.setPointerCapture(event.pointerId);
    event.preventDefault();
    const point = this.buildPoint(event);
    this.callbacks.onStart?.(point);
  }

  private onPointerMove(event: PointerEvent): void {
    if (this.activePointerId !== event.pointerId) {
      return;
    }
    event.preventDefault();
    const point = this.buildPoint(event);
    this.callbacks.onMove?.(point);
  }

  private onPointerUp(event: PointerEvent): void {
    if (this.activePointerId !== event.pointerId) {
      return;
    }
    event.preventDefault();
    const point = this.buildPoint(event);
    this.callbacks.onEnd?.(point);
    this.canvas.releasePointerCapture(event.pointerId);
    this.activePointerId = null;
  }

  private buildPoint(event: PointerEvent): PointerPoint {
    const { x, y } = this.getPoint(event);
    return {
      x,
      y,
      id: event.pointerId,
      time: performance.now() / 1000,
    };
  }
}
