import * as THREE from "three";
import { OrbitControls } from "three/addons/controls/OrbitControls.js";
import { SparkRenderer, SplatMesh } from "@sparkjsdev/spark";

export class SplatViewer {
  constructor(container) {
    this.container = container;
    this.scene = new THREE.Scene();
    this.camera = new THREE.PerspectiveCamera(60, 1, 0.01, 1000);
    this.camera.position.set(0, 0, 3);

    this.renderer = new THREE.WebGLRenderer({ antialias: true, alpha: false });
    this.renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
    this.renderer.setClearColor(0x111312, 1);
    container.appendChild(this.renderer.domElement);

    this.spark = new SparkRenderer({ renderer: this.renderer });
    this.scene.add(this.spark);

    this.controls = new OrbitControls(this.camera, this.renderer.domElement);
    this.controls.enableDamping = true;
    this.controls.dampingFactor = 0.08;
    this.controls.target.set(0, 0, 0);

    this.splat = null;
    this.resizeObserver = new ResizeObserver(() => this.resize());
    this.resizeObserver.observe(container);
    this.resize();

    this.renderer.setAnimationLoop(() => {
      this.controls.update();
      this.renderer.render(this.scene, this.camera);
    });
  }

  resize() {
    const width = Math.max(1, this.container.clientWidth);
    const height = Math.max(1, this.container.clientHeight);
    this.camera.aspect = width / height;
    this.camera.updateProjectionMatrix();
    this.renderer.setSize(width, height, false);
  }

  async load(url) {
    if (this.splat) {
      this.scene.remove(this.splat);
      this.splat.dispose();
      this.splat = null;
    }

    const cacheBustedUrl = `${url}?t=${Date.now()}`;
    const splat = new SplatMesh({ url: cacheBustedUrl, lod: true });
    splat.quaternion.set(1, 0, 0, 0);
    splat.position.set(0, 0, -1.5);
    this.scene.add(splat);
    this.splat = splat;
    await splat.initialized;
    this.frameMesh(splat);
  }

  frameMesh(mesh) {
    const box = mesh.getBoundingBox?.(true);
    if (!box || !Number.isFinite(box.min.x) || !Number.isFinite(box.max.x)) {
      return;
    }
    const center = new THREE.Vector3();
    const size = new THREE.Vector3();
    box.getCenter(center);
    box.getSize(size);

    mesh.position.sub(center);
    const radius = Math.max(size.x, size.y, size.z, 0.5);
    this.camera.position.set(0, 0, Math.max(2.2, radius * 1.8));
    this.controls.target.set(0, 0, 0);
    this.controls.update();
  }

  dispose() {
    this.renderer.setAnimationLoop(null);
    this.resizeObserver.disconnect();
    if (this.splat) {
      this.splat.dispose();
    }
    this.renderer.dispose();
  }
}
