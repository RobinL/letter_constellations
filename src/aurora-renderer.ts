// WebGPU Aurora Renderer
// Sets up WebGPU pipeline and renders aurora effect over background

import auroraShaderCode from './shaders/aurora.wgsl?raw';
import backgroundUrl from './assets/background.png';

export class AuroraRenderer {
    private canvas: HTMLCanvasElement;
    private device!: GPUDevice;
    private context!: GPUCanvasContext;
    private pipeline!: GPURenderPipeline;
    private uniformBuffer!: GPUBuffer;
    private bindGroup!: GPUBindGroup;
    private startTime: number;

    constructor(canvas: HTMLCanvasElement) {
        this.canvas = canvas;
        this.startTime = performance.now();
    }

    async initialize(): Promise<boolean> {
        // Check WebGPU support
        if (!navigator.gpu) {
            console.error('WebGPU not supported in this browser');
            return false;
        }

        const adapter = await navigator.gpu.requestAdapter();
        if (!adapter) {
            console.error('Failed to get GPU adapter');
            return false;
        }

        this.device = await adapter.requestDevice();

        // Configure canvas context
        this.context = this.canvas.getContext('webgpu') as GPUCanvasContext;
        const presentationFormat = navigator.gpu.getPreferredCanvasFormat();

        this.context.configure({
            device: this.device,
            format: presentationFormat,
            alphaMode: 'premultiplied',
        });

        // Load background texture
        const backgroundTexture = await this.loadTexture(backgroundUrl);

        // Create sampler
        const sampler = this.device.createSampler({
            magFilter: 'linear',
            minFilter: 'linear',
            mipmapFilter: 'linear',
        });

        // Create uniform buffer (time, resolution, padding - with proper alignment)
        // WGSL alignment: time(4) + pad(4) + resolution(8) + padding(4) + struct_pad(4) = 24 bytes
        this.uniformBuffer = this.device.createBuffer({
            size: 32, // Must be multiple of 16 and accommodate alignment
            usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
        });

        // Create shader module
        const shaderModule = this.device.createShaderModule({
            code: auroraShaderCode,
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
                    texture: { sampleType: 'float' },
                },
                {
                    binding: 2,
                    visibility: GPUShaderStage.FRAGMENT,
                    sampler: { type: 'filtering' },
                },
            ],
        });

        // Create bind group
        this.bindGroup = this.device.createBindGroup({
            layout: bindGroupLayout,
            entries: [
                { binding: 0, resource: { buffer: this.uniformBuffer } },
                { binding: 1, resource: backgroundTexture.createView() },
                { binding: 2, resource: sampler },
            ],
        });

        // Create pipeline layout
        const pipelineLayout = this.device.createPipelineLayout({
            bindGroupLayouts: [bindGroupLayout],
        });

        // Create render pipeline
        this.pipeline = this.device.createRenderPipeline({
            layout: pipelineLayout,
            vertex: {
                module: shaderModule,
                entryPoint: 'vertexMain',
            },
            fragment: {
                module: shaderModule,
                entryPoint: 'fragmentMain',
                targets: [{ format: presentationFormat }],
            },
            primitive: {
                topology: 'triangle-list',
            },
        });

        return true;
    }

    private async loadTexture(url: string): Promise<GPUTexture> {
        const response = await fetch(url);
        const blob = await response.blob();
        const imageBitmap = await createImageBitmap(blob);

        const texture = this.device.createTexture({
            size: [imageBitmap.width, imageBitmap.height, 1],
            format: 'rgba8unorm',
            usage:
                GPUTextureUsage.TEXTURE_BINDING |
                GPUTextureUsage.COPY_DST |
                GPUTextureUsage.RENDER_ATTACHMENT,
        });

        this.device.queue.copyExternalImageToTexture(
            { source: imageBitmap },
            { texture: texture },
            [imageBitmap.width, imageBitmap.height]
        );

        return texture;
    }

    render(): void {
        // Update uniforms with proper WGSL alignment
        // Layout: time(f32) at 0, padding at 4, resolution(vec2) at 8, _padding(f32) at 16
        const time = (performance.now() - this.startTime) / 1000;
        const uniformData = new Float32Array([
            time,           // offset 0: time
            0,              // offset 4: padding for vec2 alignment
            this.canvas.width,  // offset 8: resolution.x
            this.canvas.height, // offset 12: resolution.y
            0,              // offset 16: _padding
            0,              // offset 20: struct padding
            0,              // offset 24: struct padding
            0,              // offset 28: struct padding
        ]);
        this.device.queue.writeBuffer(this.uniformBuffer, 0, uniformData);

        // Create command encoder
        const commandEncoder = this.device.createCommandEncoder();

        // Begin render pass
        const renderPass = commandEncoder.beginRenderPass({
            colorAttachments: [
                {
                    view: this.context.getCurrentTexture().createView(),
                    clearValue: { r: 0, g: 0, b: 0, a: 1 },
                    loadOp: 'clear',
                    storeOp: 'store',
                },
            ],
        });

        renderPass.setPipeline(this.pipeline);
        renderPass.setBindGroup(0, this.bindGroup);
        renderPass.draw(3); // Fullscreen triangle
        renderPass.end();

        // Submit commands
        this.device.queue.submit([commandEncoder.finish()]);
    }

    resize(width: number, height: number): void {
        this.canvas.width = width;
        this.canvas.height = height;
    }
}
