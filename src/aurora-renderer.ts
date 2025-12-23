// WebGPU Aurora Renderer
// Sets up WebGPU pipeline and renders aurora effect over background

import auroraShaderCode from './shaders/aurora.wgsl?raw';
import backgroundUrl from './assets/background.png';
import { loadImageBitmap } from './assets';

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
        const backgroundImage = await loadImageBitmap(backgroundUrl);
        const backgroundTexture = this.createTextureFromImage(backgroundImage);

        // Create sampler (no mipmaps needed for fullscreen background)
        const sampler = this.device.createSampler({
            magFilter: 'linear',
            minFilter: 'linear',
        });

        // Create uniform buffer with proper WGSL struct alignment:
        // struct Uniforms { time: f32, resolution: vec2<f32>, _padding: f32 }
        // Memory layout: time(4) + pad(4) + resolution.x(4) + resolution.y(4) + _padding(4) + struct_pad(12) = 32 bytes
        // (structs are 16-byte aligned, vec2 requires 8-byte alignment)
        this.uniformBuffer = this.device.createBuffer({
            size: 32,
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

    private createTextureFromImage(imageBitmap: ImageBitmap): GPUTexture {
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
        // Update uniforms - must match WGSL struct layout
        const time = (performance.now() - this.startTime) / 1000;
        const uniformData = new Float32Array([
            time,               // offset 0: time
            0,                  // offset 4: padding (vec2 needs 8-byte alignment)
            this.canvas.width,  // offset 8: resolution.x
            this.canvas.height, // offset 12: resolution.y
            0,                  // offset 16: _padding
            0, 0, 0,           // offset 20-28: struct padding to 32 bytes
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

        // Reconfigure WebGPU context with new drawable size
        const presentationFormat = navigator.gpu.getPreferredCanvasFormat();
        this.context.configure({
            device: this.device,
            format: presentationFormat,
            alphaMode: 'premultiplied',
        });
    }

    getDevice(): GPUDevice {
        return this.device;
    }
}
