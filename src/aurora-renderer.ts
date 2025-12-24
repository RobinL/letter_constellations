// WebGPU Aurora Renderer
// Sets up WebGPU pipeline and renders aurora effect over background

import auroraShaderCode from './shaders/aurora.wgsl?raw';
import backgroundLandscapeUrl from './assets/background_landscape.png?url';
import backgroundPortraitUrl from './assets/background_portrait.png?url';
import { loadImageBitmap } from './assets';

export class AuroraRenderer {
    private canvas: HTMLCanvasElement;
    private device!: GPUDevice;
    private context!: GPUCanvasContext;
    private pipeline!: GPURenderPipeline;
    private uniformBuffer!: GPUBuffer;
    private bindGroup!: GPUBindGroup;
    private bindGroupLayout!: GPUBindGroupLayout;
    private backgroundLandscapeImage!: ImageBitmap;
    private backgroundPortraitImage!: ImageBitmap;
    private backgroundLandscapeTexture?: GPUTexture;
    private backgroundPortraitTexture?: GPUTexture;
    private backgroundLandscapeView?: GPUTextureView;
    private backgroundPortraitView?: GPUTextureView;
    private sampler!: GPUSampler;
    private currentBackground: 'landscape' | 'portrait' = 'landscape';
    private backgroundTextureWidth = 0;
    private backgroundTextureHeight = 0;
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

        // Load background textures
        const [backgroundLandscape, backgroundPortrait] = await Promise.all([
            loadImageBitmap(backgroundLandscapeUrl),
            loadImageBitmap(backgroundPortraitUrl),
        ]);
        this.backgroundLandscapeImage = backgroundLandscape;
        this.backgroundPortraitImage = backgroundPortrait;

        // Create sampler (no mipmaps needed for fullscreen background)
        this.sampler = this.device.createSampler({
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
        this.bindGroupLayout = this.device.createBindGroupLayout({
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
        this.updateBackgroundBindGroup();

        // Create pipeline layout
        const pipelineLayout = this.device.createPipelineLayout({
            bindGroupLayouts: [this.bindGroupLayout],
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

    private createCoveredTexture(
        imageBitmap: ImageBitmap,
        width: number,
        height: number
    ): GPUTexture {
        const canvas = document.createElement('canvas');
        canvas.width = width;
        canvas.height = height;
        const ctx = canvas.getContext('2d');
        if (!ctx) {
            throw new Error('Failed to get 2D context for background');
        }

        ctx.fillStyle = '#000';
        ctx.fillRect(0, 0, width, height);

        const imageAspect = imageBitmap.width / imageBitmap.height;
        const canvasAspect = width / height;
        let drawWidth = width;
        let drawHeight = height;
        let offsetX = 0;
        let offsetY = 0;

        if (canvasAspect > imageAspect) {
            drawWidth = width;
            drawHeight = Math.round(width / imageAspect);
            offsetY = Math.floor((height - drawHeight) / 2);
        } else {
            drawHeight = height;
            drawWidth = Math.round(height * imageAspect);
            offsetX = Math.floor((width - drawWidth) / 2);
        }

        ctx.drawImage(imageBitmap, offsetX, offsetY, drawWidth, drawHeight);

        const texture = this.device.createTexture({
            size: [width, height, 1],
            format: 'rgba8unorm',
            usage:
                GPUTextureUsage.TEXTURE_BINDING |
                GPUTextureUsage.COPY_DST |
                GPUTextureUsage.RENDER_ATTACHMENT,
        });

        this.device.queue.copyExternalImageToTexture(
            { source: canvas },
            { texture: texture },
            [width, height]
        );

        return texture;
    }

    private refreshBackgroundTextures(width: number, height: number): void {
        if (width === 0 || height === 0) {
            return;
        }

        if (this.backgroundLandscapeTexture) {
            this.backgroundLandscapeTexture.destroy();
        }
        if (this.backgroundPortraitTexture) {
            this.backgroundPortraitTexture.destroy();
        }

        this.backgroundLandscapeTexture = this.createCoveredTexture(
            this.backgroundLandscapeImage,
            width,
            height
        );
        this.backgroundPortraitTexture = this.createCoveredTexture(
            this.backgroundPortraitImage,
            width,
            height
        );
        this.backgroundLandscapeView =
            this.backgroundLandscapeTexture.createView();
        this.backgroundPortraitView =
            this.backgroundPortraitTexture.createView();
        this.backgroundTextureWidth = width;
        this.backgroundTextureHeight = height;
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

        this.updateBackgroundBindGroup();

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

    private updateBackgroundBindGroup(): void {
        if (!this.uniformBuffer || !this.bindGroupLayout) {
            return;
        }

        let refreshed = false;
        if (
            this.canvas.width !== this.backgroundTextureWidth ||
            this.canvas.height !== this.backgroundTextureHeight
        ) {
            this.refreshBackgroundTextures(this.canvas.width, this.canvas.height);
            refreshed = true;
        }

        const isPortrait = this.canvas.height > this.canvas.width;
        const desired = isPortrait ? 'portrait' : 'landscape';
        if (desired === this.currentBackground && this.bindGroup && !refreshed) {
            return;
        }

        this.currentBackground = desired;
        const view =
            desired === 'portrait'
                ? this.backgroundPortraitView
                : this.backgroundLandscapeView;
        if (!view) {
            return;
        }

        this.bindGroup = this.device.createBindGroup({
            layout: this.bindGroupLayout,
            entries: [
                { binding: 0, resource: { buffer: this.uniformBuffer } },
                { binding: 1, resource: view },
                { binding: 2, resource: this.sampler },
            ],
        });
    }
}
