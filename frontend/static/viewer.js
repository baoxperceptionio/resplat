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
    this.renderer.domElement.tabIndex = 0;
    container.appendChild(this.renderer.domElement);

    this.spark = new SparkRenderer({ renderer: this.renderer });
    this.scene.add(this.spark);

    this.controls = new OrbitControls(this.camera, this.renderer.domElement);
    this.controls.enableDamping = true;
    this.controls.dampingFactor = 0.08;
    this.controls.target.set(0, 0, 0);

    this.splat = null;
    this.keys = new Set();
    this.clock = new THREE.Clock();
    this.defaultView = {
      position: this.camera.position.clone(),
      target: this.controls.target.clone(),
    };
    this.baseMoveSpeed = 1.0;

    this.onKeyDown = (event) => this.handleKeyDown(event);
    this.onKeyUp = (event) => this.handleKeyUp(event);
    this.onPointerDown = () => this.renderer.domElement.focus();
    this.onBlur = () => this.keys.clear();
    this.renderer.domElement.addEventListener("keydown", this.onKeyDown);
    this.renderer.domElement.addEventListener("keyup", this.onKeyUp);
    this.renderer.domElement.addEventListener("pointerdown", this.onPointerDown);
    this.renderer.domElement.addEventListener("blur", this.onBlur);

    this.resizeObserver = new ResizeObserver(() => this.resize());
    this.resizeObserver.observe(container);
    this.resize();

    this.renderer.setAnimationLoop(() => {
      this.updateKeyboardMovement();
      this.controls.update();
      this.renderer.render(this.scene, this.camera);
    });
  }

  handleKeyDown(event) {
    const key = event.key.toLowerCase();
    if (!["w", "a", "s", "d", "q", "e", "r"].includes(key)) {
      return;
    }
    event.preventDefault();
    if (key === "r") {
      this.resetView();
      return;
    }
    this.keys.add(key);
  }

  handleKeyUp(event) {
    this.keys.delete(event.key.toLowerCase());
  }

  updateKeyboardMovement() {
    const delta = Math.min(this.clock.getDelta(), 0.05);
    if (this.keys.size === 0) {
      return;
    }

    const forward = new THREE.Vector3();
    this.camera.getWorldDirection(forward);
    forward.y = 0;
    if (forward.lengthSq() < 1e-6) {
      forward.set(0, 0, -1);
    }
    forward.normalize();

    const right = new THREE.Vector3().crossVectors(forward, this.camera.up).normalize();
    const up = new THREE.Vector3(0, 1, 0);
    const movement = new THREE.Vector3();

    if (this.keys.has("w")) movement.add(forward);
    if (this.keys.has("s")) movement.sub(forward);
    if (this.keys.has("d")) movement.add(right);
    if (this.keys.has("a")) movement.sub(right);
    if (this.keys.has("e")) movement.add(up);
    if (this.keys.has("q")) movement.sub(up);

    if (movement.lengthSq() === 0) {
      return;
    }

    const distance = this.camera.position.distanceTo(this.controls.target);
    const speed = Math.max(this.baseMoveSpeed, distance * 0.65) * delta;
    movement.normalize().multiplyScalar(speed);
    this.camera.position.add(movement);
    this.controls.target.add(movement);
  }

  resetView() {
    this.camera.position.copy(this.defaultView.position);
    this.controls.target.copy(this.defaultView.target);
    this.camera.updateProjectionMatrix();
    this.controls.update();
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

    const separator = url.includes("?") ? "&" : "?";
    const cacheBustedUrl = `${url}${separator}t=${Date.now()}`;
    const splat = new SplatMesh({ url: cacheBustedUrl, lod: true });
    splat.quaternion.set(1, 0, 0, 0);
    splat.position.set(0, 0, -1.5);
    this.scene.add(splat);
    this.splat = splat;
    await splat.initialized;
    this.frameMesh(splat);
  }

  clear() {
    if (this.splat) {
      this.scene.remove(this.splat);
      this.splat.dispose();
      this.splat = null;
    }
    this.camera.position.copy(this.defaultView.position);
    this.controls.target.copy(this.defaultView.target);
    this.controls.update();
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
    this.defaultView.position.copy(this.camera.position);
    this.defaultView.target.copy(this.controls.target);
    this.baseMoveSpeed = Math.max(0.35, radius * 0.45);
    this.controls.update();
  }

  dispose() {
    this.renderer.setAnimationLoop(null);
    this.resizeObserver.disconnect();
    this.renderer.domElement.removeEventListener("keydown", this.onKeyDown);
    this.renderer.domElement.removeEventListener("keyup", this.onKeyUp);
    this.renderer.domElement.removeEventListener("pointerdown", this.onPointerDown);
    this.renderer.domElement.removeEventListener("blur", this.onBlur);
    if (this.splat) {
      this.splat.dispose();
    }
    this.renderer.dispose();
  }
}
