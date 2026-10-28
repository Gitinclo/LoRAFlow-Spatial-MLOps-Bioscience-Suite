import * as THREE from 'three';
import * as xb from 'xrblocks';

export interface SpatialTelemetryState {
  runName: string;
  epoch: number;
  loss: number;
  valLoss: number;
  rank: number;
  alpha: number;
  learningRate: number;
  status: string;
  gameActive: boolean;
  gameScore: number;
  gameWave: number;
  gameIntegrity: number;
}

const _vec3 = new THREE.Vector3();
const _dummy = new THREE.Object3D();
const _color = new THREE.Color();

export class SpatialTensorLabScript extends xb.Script {
  public state: SpatialTelemetryState = {
    runName: 'ESM2-LoRA-TP53-Binders-r16',
    epoch: 14,
    loss: 0.0842,
    valLoss: 0.0915,
    rank: 16,
    alpha: 32,
    learningRate: 0.0002,
    status: 'running',
    gameActive: false,
    gameScore: 0,
    gameWave: 1,
    gameIntegrity: 100,
  };

  private reactorGroup = new THREE.Group();
  private matrixAWing = new THREE.Group();
  private matrixBWing = new THREE.Group();
  private coreRankRing?: THREE.Mesh;
  private lossRibbonMesh?: THREE.Mesh;
  private tokenInstancedMesh?: THREE.InstancedMesh;
  private hazardGroup = new THREE.Group();
  private hazards: Array<{
    mesh: THREE.Group;
    angle: number;
    distance: number;
    speed: number;
    hp: number;
  }> = [];

  private playerDrone = new THREE.Group();
  private playerAngle = 0;
  private playerRadius = 0.65;
  private heldKeys = new Set<string>();
  private hitstopTimer = 0;
  private shakeIntensity = 0;
  private audioMuted = false;
  private audioCtx: AudioContext | null = null;

  // XR Blocks Spatial UI Cards
  private controlCard?: xb.UICard;
  private statusText?: xb.UIText;
  private metricsText?: xb.UIText;
  private rankSlider?: xb.UISlider;
  private hudOverlay?: xb.UIOverlay;
  private hudText?: xb.UIText;

  public onRankChangedFromXR?: (newRank: number) => void;
  public onExportCheckpointFromXR?: () => void;
  public onGameStateUpdate?: (score: number, wave: number, integrity: number) => void;

  private clock = new THREE.Clock();

  init() {
    xb.ui.theme = 'grayGlass';

    // 1. Studio 3-Point PBR Lighting + Warm/Cool Contrast
    const hemiLight = new THREE.HemisphereLight(0xdbeafe, 0x0f172a, 1.6);
    this.add(hemiLight);

    const keyLight = new THREE.DirectionalLight(0xfff7ed, 2.6);
    keyLight.position.set(2.5, 4.0, 2.0);
    this.add(keyLight);

    const rimLight = new THREE.DirectionalLight(0x38bdf8, 1.4);
    rimLight.position.set(-2.5, 2.0, -2.5);
    this.add(rimLight);

    // 2. Build Multi-Part Procedural LoRA Tensor Reactor (10 distinct parts)
    this.reactorGroup.position.set(0, 1.15, -1.65);
    this.add(this.reactorGroup);
    this.buildProceduralTensorReactor();

    // 3. Build Instanced Embedding Token Lattice (64 instances)
    this.buildTokenLattice();

    // 4. Build Player Stabilizer Drone & Hazard Container for "Rank Stabilizer" 3D Game
    this.buildPlayerDrone();
    this.reactorGroup.add(this.hazardGroup);

    // 5. Build World-Space Spatial UI Card (`xb.UICard`) + View-Space Overlay (`xb.UIOverlay`)
    this.buildSpatialUI();

    // 6. Keyboard Input Listeners for 3D Game Mode
    window.addEventListener('keydown', (e) => {
      this.heldKeys.add(e.code);
      if (e.code === 'Space' && this.state.gameActive) {
        this.fireStabilizerPulse();
      }
    });
    window.addEventListener('keyup', (e) => {
      this.heldKeys.delete(e.code);
    });
    window.addEventListener('blur', () => {
      this.heldKeys.clear();
    });
  }

  private createProceduralGridTexture(): THREE.CanvasTexture {
    const canvas = document.createElement('canvas');
    canvas.width = 256;
    canvas.height = 256;
    const ctx = canvas.getContext('2d')!;
    ctx.fillStyle = '#0f172a';
    ctx.fillRect(0, 0, 256, 256);
    ctx.strokeStyle = '#1e293b';
    ctx.lineWidth = 2;
    for (let i = 0; i <= 256; i += 32) {
      ctx.beginPath();
      ctx.moveTo(i, 0);
      ctx.lineTo(i, 256);
      ctx.stroke();
      ctx.beginPath();
      ctx.moveTo(0, i);
      ctx.lineTo(256, i);
      ctx.stroke();
    }
    const tex = new THREE.CanvasTexture(canvas);
    tex.colorSpace = THREE.SRGBColorSpace;
    tex.wrapS = THREE.RepeatWrapping;
    tex.wrapT = THREE.RepeatWrapping;
    tex.repeat.set(4, 4);
    return tex;
  }

  private buildProceduralTensorReactor() {
    // Part 1: Grounded Pedestal Base (flush bottom-center pivot)
    const baseMat = new THREE.MeshStandardMaterial({
      color: 0x1e293b,
      metalness: 0.85,
      roughness: 0.22,
      map: this.createProceduralGridTexture(),
    });
    const pedestal = new THREE.Mesh(new THREE.CylinderGeometry(0.58, 0.68, 0.08, 36), baseMat);
    pedestal.position.y = -0.45;
    this.reactorGroup.add(pedestal);

    // Part 2: Soft Radial Contact Shadow Plane
    const shadowCanvas = document.createElement('canvas');
    shadowCanvas.width = 128;
    shadowCanvas.height = 128;
    const sctx = shadowCanvas.getContext('2d')!;
    const grad = sctx.createRadialGradient(64, 64, 8, 64, 64, 60);
    grad.addColorStop(0, 'rgba(0,0,0,0.65)');
    grad.addColorStop(1, 'rgba(0,0,0,0)');
    sctx.fillStyle = grad;
    sctx.fillRect(0, 0, 128, 128);
    const shadowTex = new THREE.CanvasTexture(shadowCanvas);
    const shadowPlane = new THREE.Mesh(
      new THREE.PlaneGeometry(1.6, 1.6),
      new THREE.MeshBasicMaterial({
        map: shadowTex,
        transparent: true,
        depthWrite: false,
        opacity: 0.55,
      })
    );
    shadowPlane.rotation.x = -Math.PI / 2;
    shadowPlane.position.y = -0.485;
    this.reactorGroup.add(shadowPlane);

    // Part 3: Brushed Gold Rank Bottleneck Core Ring (Lathe/Torus)
    const goldMat = new THREE.MeshPhysicalMaterial({
      color: 0xf59e0b,
      metalness: 0.92,
      roughness: 0.18,
      clearcoat: 0.8,
    });
    this.coreRankRing = new THREE.Mesh(new THREE.TorusGeometry(0.24, 0.032, 24, 64), goldMat);
    this.reactorGroup.add(this.coreRankRing);

    // Part 4: Optical Glass Containment Vessel
    const glassMat = new THREE.MeshPhysicalMaterial({
      color: 0x38bdf8,
      metalness: 0.1,
      roughness: 0.1,
      transparent: true,
      opacity: 0.24,
      depthWrite: false,
    });
    const vessel = new THREE.Mesh(new THREE.CylinderGeometry(0.32, 0.32, 0.56, 32, 1, true), glassMat);
    vessel.renderOrder = 2;
    this.reactorGroup.add(vessel);

    // Part 5 & 6: Low-Rank Matrix A Down-Projection Blades (Left Wing Group)
    const matA = new THREE.MeshStandardMaterial({
      color: 0x0284c7,
      metalness: 0.75,
      roughness: 0.3,
    });
    for (let i = 0; i < 4; i++) {
      const fin = new THREE.Mesh(new THREE.BoxGeometry(0.28, 0.04, 0.14), matA);
      fin.position.set(-0.42, -0.15 + i * 0.1, 0);
      fin.rotation.z = 0.18;
      this.matrixAWing.add(fin);
    }
    this.reactorGroup.add(this.matrixAWing);

    // Part 7 & 8: Low-Rank Matrix B Up-Projection Blades (Right Wing Group)
    const matB = new THREE.MeshStandardMaterial({
      color: 0x0d9488,
      metalness: 0.7,
      roughness: 0.35,
    });
    for (let i = 0; i < 4; i++) {
      const fin = new THREE.Mesh(new THREE.BoxGeometry(0.28, 0.04, 0.14), matB);
      fin.position.set(0.42, -0.15 + i * 0.1, 0);
      fin.rotation.z = -0.18;
      this.matrixBWing.add(fin);
    }
    this.reactorGroup.add(this.matrixBWing);

    // Part 9: 3D Loss Landscape Parametric Tube Ribbon
    const curvePoints: THREE.Vector3[] = [];
    for (let i = 0; i <= 24; i++) {
      const t = i / 24;
      const angle = t * Math.PI * 2;
      const r = 0.46;
      const y = -0.25 + Math.exp(-t * 3.2) * 0.42 + Math.sin(t * 12) * 0.02;
      curvePoints.push(new THREE.Vector3(Math.cos(angle) * r, y, Math.sin(angle) * r));
    }
    const curve = new THREE.CatmullRomCurve3(curvePoints, true);
    const tubeGeo = new THREE.TubeGeometry(curve, 64, 0.014, 12, true);
    const tubeMat = new THREE.MeshStandardMaterial({
      color: 0x38bdf8,
      emissive: 0x0284c7,
      emissiveIntensity: 0.6,
      roughness: 0.25,
      metalness: 0.5,
    });
    this.lossRibbonMesh = new THREE.Mesh(tubeGeo, tubeMat);
    this.reactorGroup.add(this.lossRibbonMesh);

    // Part 10: Upper & Lower Cryo-Cooling Manifold Caps
    const capMat = new THREE.MeshStandardMaterial({
      color: 0x334155,
      metalness: 0.88,
      roughness: 0.48,
    });
    const topCap = new THREE.Mesh(new THREE.CylinderGeometry(0.22, 0.34, 0.06, 24), capMat);
    topCap.position.y = 0.31;
    this.reactorGroup.add(topCap);
  }

  private buildTokenLattice() {
    const count = 48;
    const geo = new THREE.OctahedronGeometry(0.018, 0);
    const mat = new THREE.MeshStandardMaterial({
      color: 0xffffff,
      roughness: 0.25,
      metalness: 0.8,
    });
    this.tokenInstancedMesh = new THREE.InstancedMesh(geo, mat, count);
    for (let i = 0; i < count; i++) {
      const angle = (i / count) * Math.PI * 2;
      const radius = 0.52 + (i % 3) * 0.06;
      const y = -0.28 + ((i * 7) % 15) * 0.04;
      _dummy.position.set(Math.cos(angle) * radius, y, Math.sin(angle) * radius);
      const s = 0.85 + (i % 5) * 0.07;
      _dummy.scale.set(s, s, s);
      _dummy.updateMatrix();
      this.tokenInstancedMesh.setMatrixAt(i, _dummy.matrix);
      _color.setHSL(0.54 + (i % 6) * 0.02, 0.85, 0.6);
      this.tokenInstancedMesh.setColorAt(i, _color);
    }
    this.tokenInstancedMesh.instanceMatrix.needsUpdate = true;
    if (this.tokenInstancedMesh.instanceColor) {
      this.tokenInstancedMesh.instanceColor.needsUpdate = true;
    }
    this.tokenInstancedMesh.computeBoundingSphere();
    this.reactorGroup.add(this.tokenInstancedMesh);
  }

  private buildPlayerDrone() {
    // 3-part composite player stabilizer drone
    const body = new THREE.Mesh(
      new THREE.ConeGeometry(0.04, 0.11, 16),
      new THREE.MeshStandardMaterial({ color: 0x10b981, metalness: 0.8, roughness: 0.2 })
    );
    body.rotation.x = Math.PI / 2;
    const ring = new THREE.Mesh(
      new THREE.TorusGeometry(0.05, 0.008, 12, 24),
      new THREE.MeshStandardMaterial({ color: 0x38bdf8, metalness: 0.9, roughness: 0.15 })
    );
    const emitter = new THREE.Mesh(
      new THREE.SphereGeometry(0.02, 12, 12),
      new THREE.MeshBasicMaterial({ color: 0xa7f3d0 })
    );
    emitter.position.z = -0.05;
    this.playerDrone.add(body, ring, emitter);
    this.reactorGroup.add(this.playerDrone);
  }

  private buildSpatialUI() {
    // World-Space UICard positioned to the left of the 3D Tensor Reactor
    this.controlCard = new xb.UICard({
      size: { width: 0.68, height: 'auto' },
      manipulation: true,
      edge: true,
      style: {
        flexDirection: 'column',
        gap: 12,
        padding: 20,
        backgroundColor: '#0f172a',
        borderRadius: 20,
      },
    });
    this.controlCard.position.set(-0.92, 1.32, -1.45);
    this.controlCard.rotation.y = 0.28;

    const headerTitle = new xb.UIText({
      text: 'LoRAFlow Spatial Telemetry',
      style: { fontSize: 24, color: '#f8fafc' },
    });

    this.statusText = new xb.UIText({
      text: `Run: ${this.state.runName}\nStatus: RUNNING · AES-256 Encrypted`,
      style: { fontSize: 16, lineHeight: 1.35, color: '#38bdf8', whiteSpace: 'pre-line' },
    });

    this.metricsText = new xb.UIText({
      text: `Epoch ${this.state.epoch} | Train Loss: ${this.state.loss.toFixed(4)} | Val: ${this.state.valLoss.toFixed(4)}\nRank r=${this.state.rank} | Alpha α=${this.state.alpha}`,
      style: { fontSize: 16, lineHeight: 1.35, color: '#e2e8f0', whiteSpace: 'pre-line' },
    });

    this.rankSlider = new xb.UISlider({
      ariaLabel: 'LoRA Rank Dimension',
      min: 4,
      max: 128,
      step: 4,
      value: this.state.rank,
      onInput: (val: number) => {
        this.state.rank = Math.round(val);
        this.refreshSpatialText();
      },
      onChange: (val: number) => {
        const r = Math.round(val);
        this.state.rank = r;
        this.refreshSpatialText();
        if (this.onRankChangedFromXR) this.onRankChangedFromXR(r);
      },
    });

    const btnExport = new xb.UIButton({
      label: 'Export SafeTensors Checkpoint',
      onClick: () => {
        this.playSynthTone(660, 0.12, 'triangle');
        if (this.onExportCheckpointFromXR) this.onExportCheckpointFromXR();
      },
    });

    const btnToggleGame = new xb.UIButton({
      label: 'Toggle Rank Stabilizer Sim',
      onClick: () => {
        this.toggleGameMode(!this.state.gameActive);
      },
    });

    this.controlCard.add(
      headerTitle,
      this.statusText,
      this.metricsText,
      this.rankSlider,
      btnExport,
      btnToggleGame
    );
    this.add(this.controlCard);

    // View-Space UIOverlay at Top Center
    this.hudOverlay = new xb.UIOverlay({
      appearance: 'none',
      style: {
        width: 520,
        position: 'absolute',
        left: '50%',
        top: 14,
        transform: { translateX: '-50%' },
      },
      children: [],
    });
    this.hudText = new xb.UIText({
      text: 'XR Spatial Control Room Ready · Pinch or Click Nodes to Inspect',
      style: { fontSize: 15, color: '#94a3b8' },
    });
    this.hudOverlay.add(this.hudText);
    this.add(this.hudOverlay);
  }

  public refreshSpatialText() {
    if (this.statusText) {
      this.statusText.text = `Run: ${this.state.runName}\nStatus: ${this.state.status.toUpperCase()} · AES-256 Encrypted`;
    }
    if (this.metricsText) {
      this.metricsText.text = `Epoch ${this.state.epoch} | Train Loss: ${this.state.loss.toFixed(4)} | Val: ${this.state.valLoss.toFixed(4)}\nRank r=${this.state.rank} | Alpha α=${this.state.alpha}`;
    }
    if (this.hudText) {
      this.hudText.text = this.state.gameActive
        ? `RANK STABILIZER ACTIVE · Score: ${this.state.gameScore} · Wave: ${this.state.gameWave} · Integrity: ${this.state.gameIntegrity}%`
        : `LoRA Rank r=${this.state.rank} · Loss ${this.state.loss.toFixed(4)} · Pinch/Click or Use Bottom Dock`;
    }
  }

  public toggleGameMode(active: boolean) {
    this.state.gameActive = active;
    if (active) {
      this.state.gameScore = 0;
      this.state.gameWave = 1;
      this.state.gameIntegrity = 100;
      this.spawnHazardWave(4);
      this.playSynthTone(523.25, 0.15, 'sine');
    } else {
      for (const h of this.hazards) {
        this.hazardGroup.remove(h.mesh);
      }
      this.hazards = [];
    }
    this.refreshSpatialText();
    if (this.onGameStateUpdate) {
      this.onGameStateUpdate(this.state.gameScore, this.state.gameWave, this.state.gameIntegrity);
    }
  }

  private spawnHazardWave(count: number) {
    for (let i = 0; i < count; i++) {
      const hMesh = new THREE.Group();
      const core = new THREE.Mesh(
        new THREE.OctahedronGeometry(0.045, 0),
        new THREE.MeshStandardMaterial({
          color: 0xef4444,
          emissive: 0x991b1b,
          emissiveIntensity: 0.7,
          roughness: 0.2,
          metalness: 0.7,
        })
      );
      const spikeRing = new THREE.Mesh(
        new THREE.TorusGeometry(0.06, 0.008, 8, 16),
        new THREE.MeshStandardMaterial({ color: 0xf97316, roughness: 0.4, metalness: 0.8 })
      );
      hMesh.add(core, spikeRing);
      const angle = (i / count) * Math.PI * 2 + Math.random() * 0.5;
      const distance = 0.85 + Math.random() * 0.25;
      hMesh.position.set(Math.cos(angle) * distance, 0, Math.sin(angle) * distance);
      this.hazardGroup.add(hMesh);
      this.hazards.push({
        mesh: hMesh,
        angle,
        distance,
        speed: 0.12 + this.state.gameWave * 0.025,
        hp: 1,
      });
    }
  }

  public fireStabilizerPulse() {
    if (!this.state.gameActive) return;
    this.playSynthTone(880, 0.08, 'sawtooth');
    // Hit nearest hazard within angular arc
    let hitIndex = -1;
    let bestAngDiff = 0.55;
    for (let i = 0; i < this.hazards.length; i++) {
      const diff = Math.abs(
        Math.atan2(
          Math.sin(this.hazards[i].angle - this.playerAngle),
          Math.cos(this.hazards[i].angle - this.playerAngle)
        )
      );
      if (diff < bestAngDiff) {
        bestAngDiff = diff;
        hitIndex = i;
      }
    }
    if (hitIndex >= 0) {
      const target = this.hazards[hitIndex];
      this.hazardGroup.remove(target.mesh);
      this.hazards.splice(hitIndex, 1);
      this.state.gameScore += 150;
      this.hitstopTimer = 0.045; // 45ms hitstop on heavy impact
      this.shakeIntensity = 0.025;
      this.playSynthTone(1174.66, 0.11, 'triangle');
      if (this.hazards.length === 0) {
        this.state.gameWave += 1;
        this.spawnHazardWave(3 + this.state.gameWave);
      }
      this.refreshSpatialText();
      if (this.onGameStateUpdate) {
        this.onGameStateUpdate(this.state.gameScore, this.state.gameWave, this.state.gameIntegrity);
      }
    }
  }

  public setAudioMuted(muted: boolean) {
    this.audioMuted = muted;
  }

  private playSynthTone(freq: number, duration: number, type: OscillatorType) {
    if (this.audioMuted) return;
    try {
      if (!this.audioCtx) {
        this.audioCtx = new AudioContext();
      }
      const osc = this.audioCtx.createOscillator();
      const gain = this.audioCtx.createGain();
      osc.type = type;
      osc.frequency.setValueAtTime(freq, this.audioCtx.currentTime);
      gain.gain.setValueAtTime(0.08, this.audioCtx.currentTime);
      gain.gain.exponentialRampToValueAtTime(0.001, this.audioCtx.currentTime + duration);
      osc.connect(gain);
      gain.connect(this.audioCtx.destination);
      osc.start();
      osc.stop(this.audioCtx.currentTime + duration);
    } catch {
      // ignore if user hasn't interacted with DOM yet
    }
  }

  onSelectEnd(_event: xb.SelectEvent) {
    if (this.state.gameActive) {
      this.fireStabilizerPulse();
    } else {
      this.playSynthTone(587.33, 0.09, 'sine');
    }
  }

  update() {
    const rawDt = this.clock.getDelta();
    const dt = Math.min(rawDt, 0.05);

    if (this.hitstopTimer > 0) {
      this.hitstopTimer -= dt;
      return;
    }

    // Exponentially damped scale of core rank ring based on LoRA rank
    const targetScale = 0.75 + (this.state.rank / 128) * 0.75;
    if (this.coreRankRing) {
      const s = THREE.MathUtils.damp(this.coreRankRing.scale.x, targetScale, 6, dt);
      this.coreRankRing.scale.set(s, s, s);
      this.coreRankRing.rotation.z += dt * 0.65;
      this.coreRankRing.rotation.x = Math.sin(this.clock.elapsedTime * 1.4) * 0.25;
    }

    // Damped Matrix A & B Wing separation
    const targetWingX = 0.04 + (this.state.alpha / 128) * 0.14;
    this.matrixAWing.position.x = THREE.MathUtils.damp(
      this.matrixAWing.position.x,
      -targetWingX,
      5,
      dt
    );
    this.matrixBWing.position.x = THREE.MathUtils.damp(
      this.matrixBWing.position.x,
      targetWingX,
      5,
      dt
    );

    if (this.lossRibbonMesh) {
      this.lossRibbonMesh.rotation.y += dt * 0.35;
    }

    // Player Drone Orbital Movement (A/D or ArrowLeft/ArrowRight)
    if (this.heldKeys.has('KeyA') || this.heldKeys.has('ArrowLeft')) {
      this.playerAngle -= dt * 2.8;
    }
    if (this.heldKeys.has('KeyD') || this.heldKeys.has('ArrowRight')) {
      this.playerAngle += dt * 2.8;
    }
    _vec3.set(
      Math.cos(this.playerAngle) * this.playerRadius,
      0.04,
      Math.sin(this.playerAngle) * this.playerRadius
    );
    this.playerDrone.position.copy(_vec3);
    this.playerDrone.lookAt(this.reactorGroup.position);

    // Update Game Hazards if active
    if (this.state.gameActive) {
      for (let i = this.hazards.length - 1; i >= 0; i--) {
        const h = this.hazards[i];
        h.distance -= h.speed * dt;
        h.angle += dt * 0.4;
        h.mesh.position.set(Math.cos(h.angle) * h.distance, 0.04, Math.sin(h.angle) * h.distance);
        h.mesh.rotation.x += dt * 3;
        h.mesh.rotation.y += dt * 2;
        if (h.distance <= 0.22) {
          this.hazardGroup.remove(h.mesh);
          this.hazards.splice(i, 1);
          this.state.gameIntegrity = Math.max(0, this.state.gameIntegrity - 15);
          this.shakeIntensity = 0.04;
          this.playSynthTone(164.81, 0.18, 'sawtooth');
          if (this.state.gameIntegrity <= 0) {
            this.toggleGameMode(false);
          } else if (this.hazards.length === 0) {
            this.state.gameWave += 1;
            this.spawnHazardWave(3 + this.state.gameWave);
          }
          this.refreshSpatialText();
          if (this.onGameStateUpdate) {
            this.onGameStateUpdate(
              this.state.gameScore,
              this.state.gameWave,
              this.state.gameIntegrity
            );
          }
        }
      }
    }

    // Decaying Camera / Reactor Shake
    if (this.shakeIntensity > 0.001) {
      this.reactorGroup.position.x = (Math.random() - 0.5) * this.shakeIntensity;
      this.shakeIntensity = THREE.MathUtils.damp(this.shakeIntensity, 0, 8, dt);
    } else {
      this.reactorGroup.position.x = 0;
    }
  }
}
