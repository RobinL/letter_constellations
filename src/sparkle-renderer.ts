// WebGPU Sparkle Renderer
// Creates magical twinkling sparkles that follow the cursor with momentum

import sparkleShaderCode from './shaders/sparkle.wgsl?raw';

export type TrailPoint = {
    x: number;
    y: number;
    time: number;
};

const MAX_TRAIL_POINTS = 64;

export class SparkleRenderer {
    private canvas: HTMLCanvasElement;
    private device!: GPUDevice;
    private context!: GPUCanvasContext;
    private pipeline!: GPURenderPipeline;
    private uniformBuffer!: GPUBuffer;
    private trailBuffer!: GPUBuffer;
    private bindGroup!: GPUBindGroup;
    private startTime: number;

    // Mouse state
    private mouseActive = false;
    private mousePos = { x: 0.5, y: 0.5 };
    private mouseVelocity = { x: 0, y: 0 };
    private lastMousePos = { x: 0.5, y: 0.5 };
    private lastMouseTime = 0;

    // Trail data
    private trailPoints: TrailPoint[] = [];

    constructor(canvas: HTMLCanvasElement) {
        this.canvas = canvas;
        this.startTime = performance.now();
    }

    async initialize(existingDevice?: GPUDevice): Promise<boolean> {
        // Use existing device or create new one
        if (existingDevice) {
            this.device = existingDevice;
        } else {
            if (!navigator.gpu) {
                console.error('WebGPU not supported');
                return false;
            }

            const adapter = await navigator.gpu.requestAdapter();
            if (!adapter) {
                console.error('Failed to get GPU adapter');
                return false;
            }

            this.device = await adapter.requestDevice();
        }

        // Configure canvas context with alpha for compositing
        this.context = this.canvas.getContext('webgpu') as GPUCanvasContext;
        const presentationFormat = navigator.gpu.getPreferredCanvasFormat();

        this.context.configure({
            device: this.device,
            format: presentationFormat,
            alphaMode: 'premultiplied',
        });

        // Create uniform buffer
        // Struct layout:
        // time: f32 (4) + padding (4) + resolution: vec2<f32> (8) +
        // mouse_active: f32 (4) + padding (4) + mouse_pos: vec2<f32> (8) +
        // mouse_velocity: vec2<f32> (8) + trail_count: f32 (4) + _padding: f32 (4)
        // Total: 48 bytes (aligned to 16)
        this.uniformBuffer = this.device.createBuffer({
            size: 48,
            usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
        });

        // Create trail buffer (storage buffer for trail points)
        // Each point: pos (vec2: 8) + time (f32: 4) + padding (4) = 16 bytes
        const trailBufferSize = MAX_TRAIL_POINTS * 16;
        this.trailBuffer = this.device.createBuffer({
            size: trailBufferSize,
            usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
        });

        // Create shader module
        const shaderModule = this.device.createShaderModule({
            code: sparkleShaderCode,
        });

        // Create bind group layout
        const bindGroupLayout = this.device.createBindGroupLayout({
            entries: [
                {
                    binding: 0,
                    visibility: GPUShaderStage.FRAGMENT,
                    buffer: { type: 'uniform' },
                },
                {
                    binding: 1,
                    visibility: GPUShaderStage.FRAGMENT,
                    buffer: { type: 'read-only-storage' },
                },
            ],
        });

        // Create bind group
        this.bindGroup = this.device.createBindGroup({
            layout: bindGroupLayout,
            entries: [
                { binding: 0, resource: { buffer: this.uniformBuffer } },
                { binding: 1, resource: { buffer: this.trailBuffer } },
            ],
        });

        // Create pipeline layout
        const pipelineLayout = this.device.createPipelineLayout({
            bindGroupLayouts: [bindGroupLayout],
        });

        // Create render pipeline with alpha blending
        this.pipeline = this.device.createRenderPipeline({
            layout: pipelineLayout,
            vertex: {
                module: shaderModule,
                entryPoint: 'vertexMain',
            },
            fragment: {
                module: shaderModule,
                entryPoint: 'fragmentMain',
                targets: [
                    {
                        format: presentationFormat,
                        blend: {
                            color: {
                                srcFactor: 'src-alpha',
                                dstFactor: 'one-minus-src-alpha',
                                operation: 'add',
                            },
                            alpha: {
                                srcFactor: 'one',
                                dstFactor: 'one-minus-src-alpha',
                                operation: 'add',
                            },
                        },
                    },
                ],
            },
            primitive: {
                topology: 'triangle-list',
            },
        });

        return true;
    }

    // Update mouse state from game input
    updateMouseState(
        active: boolean,
        x: number,
        y: number,
        trailPoints: TrailPoint[]
    ): void {
        const currentTime = performance.now() / 1000;

        // Convert coordinates to normalized UV space (0-1)
        const normalizedX = x / (this.canvas.width / window.devicePixelRatio);
        const normalizedY = y / (this.canvas.height / window.devicePixelRatio);

        // Calculate velocity
        const dt = currentTime - this.lastMouseTime;
        if (dt > 0 && dt < 0.1) {
            // Smooth velocity with exponential moving average
            const newVelX = (normalizedX - this.lastMousePos.x) / dt;
            const newVelY = (normalizedY - this.lastMousePos.y) / dt;
            this.mouseVelocity.x = this.mouseVelocity.x * 0.7 + newVelX * 0.3;
            this.mouseVelocity.y = this.mouseVelocity.y * 0.7 + newVelY * 0.3;
        }

        this.mouseActive = active;
        this.lastMousePos = { ...this.mousePos };
        this.mousePos = { x: normalizedX, y: normalizedY };
        this.lastMouseTime = currentTime;

        // Convert trail points to normalized coordinates
        this.trailPoints = trailPoints.slice(-MAX_TRAIL_POINTS).map((p) => ({
            x: p.x / (this.canvas.width / window.devicePixelRatio),
            y: p.y / (this.canvas.height / window.devicePixelRatio),
            time: p.time,
        }));

        // Apply velocity decay when not active
        if (!active) {
            this.mouseVelocity.x *= 0.95;
            this.mouseVelocity.y *= 0.95;
        }
    }

    render(): void {
        const time = (performance.now() - this.startTime) / 1000;

        // Convert trail times to be relative to our startTime (same time base as shader)
        const adjustedTrailPoints = this.trailPoints.map((p) => ({
            ...p,
            time: p.time - this.startTime / 1000,
        }));

        // Update uniform buffer
        const uniformData = new Float32Array([
            time, // offset 0: time
            0, // offset 4: padding for vec2 alignment
            this.canvas.width, // offset 8: resolution.x
            this.canvas.height, // offset 12: resolution.y
            this.mouseActive ? 1.0 : 0.0, // offset 16: mouse_active
            0, // offset 20: padding for vec2 alignment
            this.mousePos.x, // offset 24: mouse_pos.x
            this.mousePos.y, // offset 28: mouse_pos.y
            this.mouseVelocity.x, // offset 32: mouse_velocity.x
            this.mouseVelocity.y, // offset 36: mouse_velocity.y
            adjustedTrailPoints.length, // offset 40: trail_count
            0, // offset 44: _padding
        ]);
        this.device.queue.writeBuffer(this.uniformBuffer, 0, uniformData);

        // Update trail buffer
        const trailData = new Float32Array(MAX_TRAIL_POINTS * 4);
        for (let i = 0; i < adjustedTrailPoints.length; i++) {
            const point = adjustedTrailPoints[i];
            trailData[i * 4 + 0] = point.x;
            trailData[i * 4 + 1] = point.y;
            trailData[i * 4 + 2] = point.time;
            trailData[i * 4 + 3] = 0; // padding
        }
        this.device.queue.writeBuffer(this.trailBuffer, 0, trailData);

        // Create command encoder
        const commandEncoder = this.device.createCommandEncoder();

        // Begin render pass
        const renderPass = commandEncoder.beginRenderPass({
            colorAttachments: [
                {
                    view: this.context.getCurrentTexture().createView(),
                    clearValue: { r: 0, g: 0, b: 0, a: 0 },
                    loadOp: 'clear',
                    storeOp: 'store',
                },
            ],
        });

        renderPass.setPipeline(this.pipeline);
        renderPass.setBindGroup(0, this.bindGroup);
        renderPass.draw(3);
        renderPass.end();

        // Submit commands
        this.device.queue.submit([commandEncoder.finish()]);
    }

    resize(width: number, height: number): void {
        this.canvas.width = width;
        this.canvas.height = height;

        // Reconfigure WebGPU context
        const presentationFormat = navigator.gpu.getPreferredCanvasFormat();
        this.context.configure({
            device: this.device,
            format: presentationFormat,
            alphaMode: 'premultiplied',
        });
    }
}
