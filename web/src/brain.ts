/** PLACEHOLDER (Session 2 finishes this).
 *
 *  All this does today is prove the three.js scaffold and `positions.bin`
 *  load: 139,241 real neuron positions as a dim, slowly drifting point cloud.
 *  It is NOT driven by the recording -- `activity.bin` is not even fetched
 *  yet. Session 2 adds the two `DataTexture`s, the vertex-shader lerp between
 *  activity frames, bloom and the input/DN highlight toggle (PLAN task 3).
 */

import * as THREE from "three";
import type { BrainAssets } from "./types.ts";

/** One muted colour per FlyWire super_class, in brain.json's order. */
const CLASS_COLORS = [
  0x5b8cff, 0x7ce3c4, 0xffb45b, 0xff7ba8, 0xb08cff, 0x8fd0ff, 0xffe08a, 0x7affb0,
  0xff9d7a, 0x9aa7c7,
];

export class BrainView {
  private renderer: THREE.WebGLRenderer;
  private scene = new THREE.Scene();
  private camera: THREE.PerspectiveCamera;
  private points: THREE.Points;

  constructor(canvas: HTMLCanvasElement, assets: BrainAssets) {
    this.renderer = new THREE.WebGLRenderer({ canvas, antialias: false, alpha: true });
    this.renderer.setPixelRatio(Math.min(window.devicePixelRatio || 1, 2));
    this.camera = new THREE.PerspectiveCamera(40, 1, 0.01, 100);
    this.camera.position.set(0, 0, 2.5);

    const n = assets.brain.n_neurons;
    const colors = new Float32Array(n * 3);
    const c = new THREE.Color();
    for (let i = 0; i < n; i++) {
      c.setHex(CLASS_COLORS[assets.classes[i] % CLASS_COLORS.length]);
      // D15: 21,141 neurons sit at an anchor point rather than a true soma.
      // Drawing those dimmer is the honest visual cue (site plan §5 item 8).
      const k = assets.posSource[i] === 1 ? 0.45 : 1.0;
      colors[i * 3] = c.r * k;
      colors[i * 3 + 1] = c.g * k;
      colors[i * 3 + 2] = c.b * k;
    }

    const geom = new THREE.BufferGeometry();
    geom.setAttribute("position", new THREE.BufferAttribute(assets.positions, 3));
    geom.setAttribute("color", new THREE.BufferAttribute(colors, 3));
    const mat = new THREE.PointsMaterial({
      size: 0.006,
      vertexColors: true,
      transparent: true,
      opacity: 0.32,
      blending: THREE.AdditiveBlending,
      depthWrite: false,
      sizeAttenuation: true,
    });
    this.points = new THREE.Points(geom, mat);
    // FlyWire's axes put the brain on its side for a screen-facing view.
    this.points.rotation.x = -Math.PI / 2;
    this.scene.add(this.points);
    this.resize();
  }

  resize(): void {
    const c = this.renderer.domElement;
    const w = Math.max(1, c.clientWidth);
    const h = Math.max(1, c.clientHeight);
    this.renderer.setSize(w, h, false);
    this.camera.aspect = w / h;
    this.camera.updateProjectionMatrix();
  }

  /** `t` is the master playhead, so even the placeholder's drift is on the
   *  one clock rather than a timer of its own. */
  render(t: number): void {
    this.points.rotation.z = t * 0.06;
    this.renderer.render(this.scene, this.camera);
  }
}
