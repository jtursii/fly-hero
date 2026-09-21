/** The connectome panel (PLAN Phase 7 task 3, site plan §4).
 *
 *  139,241 neurons as one `THREE.Points`. The 4,096 recorded slots flash
 *  from `activity.bin`, interpolated in the vertex shader between the two
 *  20 fps frames that bracket the master playhead; every other neuron is a
 *  dim static cloud, because nothing was recorded for it and the site never
 *  invents activity it does not have. Anchor-positioned neurons are dimmer
 *  again (D15).
 *
 *  The signal lines are real FlyWire edges (`flow_edges.bin`, written by
 *  export_web.py): optic lobes -> central brain -> descending neurons, red
 *  where the edge is excitatory and gold where it is inhibitory, brightness
 *  from the recorded activity of the neuron the signal leaves from.
 *
 *  Everything reads the master playhead, so scrubbing and rate changes need
 *  no special handling here.
 */

import * as THREE from "three";
import { OrbitControls } from "three/examples/jsm/controls/OrbitControls.js";
import type { BrainAssets, Manifest, SongHeavy } from "./types.ts";

/** One muted colour per FlyWire super_class, in brain.json's order. */
const CLASS_COLORS = [
  0x5b8cff, 0x7ce3c4, 0xffb45b, 0xff7ba8, 0xb08cff, 0x8fd0ff, 0xffe08a, 0x7affb0,
  0xff9d7a, 0x9aa7c7,
];

const EXCITATORY = new THREE.Color(0xff7a5c);
const INHIBITORY = new THREE.Color(0xffc64d);

/** 4,096 slots as a 64x64 R8 texture the vertex shader can sample. */
const TEX = 64;
/** Seconds of stillness after a drag before the slow spin comes back. */
const AUTO_ROTATE_DELAY_S = 3;

const POINT_VS = /* glsl */ `
  attribute vec3 aColor;
  attribute float aSlot;   // -1 for the neurons nothing was recorded for
  attribute float aDim;    // D15: anchor positions are drawn dimmer
  uniform sampler2D uActA;
  uniform sampler2D uActB;
  uniform float uMix;
  uniform float uScale;
  varying vec3 vColor;
  varying float vGlow;

  float slotLevel(float slot) {
    vec2 uv = vec2(mod(slot, ${TEX}.0) + 0.5, floor(slot / ${TEX}.0) + 0.5) / ${TEX}.0;
    return mix(texture2D(uActA, uv).r, texture2D(uActB, uv).r, uMix);
  }

  void main() {
    float level = aSlot < 0.0 ? 0.0 : slotLevel(aSlot);
    vGlow = level;
    // Unrecorded neurons keep a flat base; recorded ones ride their activity.
    float bright = aSlot < 0.0 ? 0.30 : 0.22 + 2.4 * level;
    vColor = aColor * aDim * bright;

    vec4 mv = modelViewMatrix * vec4(position, 1.0);
    gl_Position = projectionMatrix * mv;
    // World-radius -> pixels: uScale is drawingBufferHeight / 2 tan(fov/2).
    float size = aSlot < 0.0 ? 0.0030 : 0.0038 + 0.0130 * level;
    gl_PointSize = max(1.0, size * uScale / max(0.15, -mv.z));
  }
`;

const POINT_FS = /* glsl */ `
  varying vec3 vColor;
  varying float vGlow;
  void main() {
    // Soft round sprite; recorded neurons get a hotter core as they fire.
    float d = length(gl_PointCoord - 0.5) * 2.0;
    float a = smoothstep(1.0, 0.25, d);
    if (a <= 0.003) discard;
    gl_FragColor = vec4(vColor * (1.0 + 1.6 * vGlow * (1.0 - d)), a);
  }
`;

const LINE_VS = /* glsl */ `
  attribute vec3 aColor;
  attribute float aSlot;   // the presynaptic neuron, for both endpoints
  attribute float aEnd;    // 0 at the presynaptic end, 1 at the postsynaptic
  uniform sampler2D uActA;
  uniform sampler2D uActB;
  uniform float uMix;
  varying vec3 vColor;
  varying float vAlpha;

  void main() {
    vec2 uv = vec2(mod(aSlot, ${TEX}.0) + 0.5, floor(aSlot / ${TEX}.0) + 0.5) / ${TEX}.0;
    float level = mix(texture2D(uActA, uv).r, texture2D(uActB, uv).r, uMix);
    vColor = aColor;
    // Fades along the edge, so the direction of flow reads at a glance.
    vAlpha = level * mix(0.55, 0.04, aEnd);
    gl_Position = projectionMatrix * modelViewMatrix * vec4(position, 1.0);
  }
`;

const LINE_FS = /* glsl */ `
  varying vec3 vColor;
  varying float vAlpha;
  void main() {
    if (vAlpha <= 0.004) discard;
    gl_FragColor = vec4(vColor * vAlpha, vAlpha);
  }
`;

/** What the panel's readouts show for the current playhead. */
export interface BrainStats {
  /** Recorded slots above the 4-bit quantizer's first level. */
  active: number;
  /** Mean activation over all 139,241 neurons, from scope channel "all". */
  meanActivation: number;
}

export class BrainView {
  private renderer: THREE.WebGLRenderer;
  private scene = new THREE.Scene();
  private camera: THREE.PerspectiveCamera;
  private controls: OrbitControls;
  private group = new THREE.Group();
  private lines: THREE.LineSegments | null = null;
  private texA: THREE.DataTexture;
  private texB: THREE.DataTexture;
  private uniforms: Record<string, THREE.IUniform>;
  private slotOfNeuron: Int32Array;
  private lastIdleAt = 0;
  private dragging = false;

  /** Set by `setSong`; null until a song's heavy files have arrived. */
  private song: { heavy: SongHeavy; manifest: Manifest; scopeAll: number } | null = null;
  /** The activity frame currently in texA, so it is unpacked only on change. */
  private frameA = -1;
  readonly stats: BrainStats = { active: 0, meanActivation: 0 };

  constructor(canvas: HTMLCanvasElement, assets: BrainAssets) {
    const { brain, positions, classes, posSource } = assets;
    const n = brain.n_neurons;

    this.renderer = new THREE.WebGLRenderer({ canvas, antialias: false, alpha: true });
    this.renderer.setPixelRatio(Math.min(window.devicePixelRatio || 1, 2));
    this.camera = new THREE.PerspectiveCamera(40, 1, 0.01, 100);
    this.camera.position.set(0, 0, 1.9);

    this.controls = new OrbitControls(this.camera, canvas);
    this.controls.enableDamping = true;
    this.controls.dampingFactor = 0.08;
    this.controls.enablePan = false; // panning is how a brain gets lost off-screen
    // Clamped so the brain can never be zoomed into nothing or lost in the
    // distance; panning is off for the same reason.
    this.controls.minDistance = 1.1;
    this.controls.maxDistance = 3.2;
    this.controls.rotateSpeed = 0.8;
    this.controls.zoomSpeed = 0.7;
    this.controls.autoRotate = true;
    this.controls.autoRotateSpeed = 0.35;
    this.controls.addEventListener("start", () => {
      this.dragging = true;
      this.controls.autoRotate = false;
    });
    this.controls.addEventListener("end", () => {
      this.dragging = false;
      this.lastIdleAt = performance.now();
    });

    this.texA = makeActivityTexture();
    this.texB = makeActivityTexture();
    this.uniforms = {
      uActA: { value: this.texA },
      uActB: { value: this.texB },
      uMix: { value: 0 },
      uScale: { value: 1 },
    };

    // slot lookup, and the per-neuron attributes the shaders read.
    this.slotOfNeuron = new Int32Array(n).fill(-1);
    brain.slot_neuron_idx.forEach((neuron, slot) => (this.slotOfNeuron[neuron] = slot));

    const colors = new Float32Array(n * 3);
    const dim = new Float32Array(n);
    const slot = new Float32Array(n);
    const c = new THREE.Color();
    for (let i = 0; i < n; i++) {
      c.setHex(CLASS_COLORS[classes[i] % CLASS_COLORS.length]);
      colors[i * 3] = c.r;
      colors[i * 3 + 1] = c.g;
      colors[i * 3 + 2] = c.b;
      // D15: 21,141 neurons sit at an anchor point rather than a true soma.
      dim[i] = posSource[i] === 1 ? 0.5 : 1.0;
      slot[i] = this.slotOfNeuron[i];
    }

    const geom = new THREE.BufferGeometry();
    geom.setAttribute("position", new THREE.BufferAttribute(positions, 3));
    geom.setAttribute("aColor", new THREE.BufferAttribute(colors, 3));
    geom.setAttribute("aDim", new THREE.BufferAttribute(dim, 1));
    geom.setAttribute("aSlot", new THREE.BufferAttribute(slot, 1));
    const points = new THREE.Points(
      geom,
      new THREE.ShaderMaterial({
        uniforms: this.uniforms,
        vertexShader: POINT_VS,
        fragmentShader: POINT_FS,
        transparent: true,
        depthWrite: false,
        blending: THREE.AdditiveBlending,
      }),
    );
    this.group.add(points);

    if (assets.flowEdges) this.buildFlowLines(assets.flowEdges, positions);

    // FlyWire's axes put the brain on its side for a screen-facing view;
    // the small tilt keeps it from reading as a flat sheet.
    this.group.rotation.x = -Math.PI / 2 + 0.28;
    this.scene.add(this.group);
    this.resize();
  }

  /** Real edges from `flow_edges.bin`: Int32 [E, 3] of presynaptic slot,
   *  postsynaptic neuron, sign. */
  private buildFlowLines(edges: Int32Array, positions: Float32Array): void {
    const e = edges.length / 3;
    const pos = new Float32Array(e * 6);
    const col = new Float32Array(e * 6);
    const slot = new Float32Array(e * 2);
    const end = new Float32Array(e * 2);
    for (let i = 0; i < e; i++) {
      const preSlot = edges[i * 3];
      const preNeuron = this.slotNeuron(preSlot);
      const postNeuron = edges[i * 3 + 1];
      const c = edges[i * 3 + 2] >= 0 ? EXCITATORY : INHIBITORY;
      for (let v = 0; v < 2; v++) {
        const src = (v === 0 ? preNeuron : postNeuron) * 3;
        const dst = (i * 2 + v) * 3;
        pos[dst] = positions[src];
        pos[dst + 1] = positions[src + 1];
        pos[dst + 2] = positions[src + 2];
        col[dst] = c.r;
        col[dst + 1] = c.g;
        col[dst + 2] = c.b;
        slot[i * 2 + v] = preSlot;
        end[i * 2 + v] = v;
      }
    }
    const g = new THREE.BufferGeometry();
    g.setAttribute("position", new THREE.BufferAttribute(pos, 3));
    g.setAttribute("aColor", new THREE.BufferAttribute(col, 3));
    g.setAttribute("aSlot", new THREE.BufferAttribute(slot, 1));
    g.setAttribute("aEnd", new THREE.BufferAttribute(end, 1));
    this.lines = new THREE.LineSegments(
      g,
      new THREE.ShaderMaterial({
        uniforms: this.uniforms,
        vertexShader: LINE_VS,
        fragmentShader: LINE_FS,
        transparent: true,
        depthWrite: false,
        blending: THREE.AdditiveBlending,
      }),
    );
    this.group.add(this.lines);
  }

  /** slot -> neuron index, by scanning the lookup once at build time. */
  private slotNeuronCache: Int32Array | null = null;
  private slotNeuron(slot: number): number {
    if (!this.slotNeuronCache) {
      this.slotNeuronCache = new Int32Array(TEX * TEX).fill(0);
      for (let i = 0; i < this.slotOfNeuron.length; i++) {
        const s = this.slotOfNeuron[i];
        if (s >= 0) this.slotNeuronCache[s] = i;
      }
    }
    return this.slotNeuronCache[slot];
  }

  /** Hand the panel a song's recording, or null while one is loading. */
  setSong(heavy: SongHeavy | null, manifest: Manifest | null, scopeAllChannel = 0): void {
    this.song = heavy && manifest ? { heavy, manifest, scopeAll: scopeAllChannel } : null;
    this.frameA = -1;
    if (!this.song) {
      this.texA.image.data.fill(0);
      this.texB.image.data.fill(0);
      this.texA.needsUpdate = this.texB.needsUpdate = true;
      this.stats.active = 0;
      this.stats.meanActivation = 0;
    }
  }

  /** Camera state, for the interaction checks in scripts/check_brain.mjs. */
  get cameraState(): { distance: number; azimuth: number; autoRotate: boolean } {
    return {
      distance: this.camera.position.length(),
      azimuth: Math.atan2(this.camera.position.x, this.camera.position.z),
      autoRotate: this.controls.autoRotate,
    };
  }

  resize(): void {
    const c = this.renderer.domElement;
    const w = Math.max(1, c.clientWidth);
    const h = Math.max(1, c.clientHeight);
    this.renderer.setSize(w, h, false);
    this.camera.aspect = w / h;
    this.camera.updateProjectionMatrix();
    const dpr = this.renderer.getPixelRatio();
    this.uniforms.uScale.value = (h * dpr) / (2 * Math.tan((this.camera.fov * Math.PI) / 360));
  }

  /** Unpack one 4-bit activity frame into a texture (level/15 -> 0..255). */
  private fillFrame(tex: THREE.DataTexture, frame: number): void {
    const { heavy, manifest } = this.song!;
    const k = manifest.activity_slots;
    const half = k >> 1;
    const base = Math.min(Math.max(frame, 0), manifest.activity_frames - 1) * half;
    const out = tex.image.data as Uint8Array;
    const src = heavy.activity;
    for (let i = 0; i < half; i++) {
      const byte = src[base + i];
      out[i * 2] = (byte & 0x0f) * 17;
      out[i * 2 + 1] = (byte >> 4) * 17;
    }
    tex.needsUpdate = true;
  }

  private updateStats(frame: number): void {
    const { heavy, manifest, scopeAll } = this.song!;
    const data = this.texA.image.data as Uint8Array;
    let active = 0;
    for (let i = 0; i < manifest.activity_slots; i++) if (data[i] > 0) active++;
    this.stats.active = active;

    const channels = heavy.scope.length / manifest.frames;
    const f = Math.min(Math.max(Math.round(frame * (manifest.fps / manifest.activity_fps)), 0), manifest.frames - 1);
    this.stats.meanActivation = heavy.scope[f * channels + scopeAll];
  }

  /** `t` is the master playhead in seconds; everything here is a pure
   *  function of it, so scrubbing and rate changes need no extra work. */
  render(t: number): void {
    if (this.song) {
      const { manifest } = this.song;
      const exact = t * manifest.activity_fps;
      const i0 = Math.min(Math.max(Math.floor(exact), 0), manifest.activity_frames - 1);
      if (i0 !== this.frameA) {
        this.fillFrame(this.texA, i0);
        this.fillFrame(this.texB, i0 + 1);
        this.frameA = i0;
        this.updateStats(i0);
      }
      this.uniforms.uMix.value = Math.min(Math.max(exact - i0, 0), 1);
    }

    if (!this.dragging && !this.controls.autoRotate &&
        performance.now() - this.lastIdleAt > AUTO_ROTATE_DELAY_S * 1000) {
      this.controls.autoRotate = true;
    }
    this.controls.update();
    this.renderer.render(this.scene, this.camera);
  }
}

function makeActivityTexture(): THREE.DataTexture {
  const tex = new THREE.DataTexture(new Uint8Array(TEX * TEX), TEX, TEX, THREE.RedFormat);
  tex.minFilter = tex.magFilter = THREE.NearestFilter;
  tex.needsUpdate = true;
  return tex;
}
