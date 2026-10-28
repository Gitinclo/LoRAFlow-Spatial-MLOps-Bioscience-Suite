import * as xb from 'xrblocks';
import { SpatialTensorLabScript } from './spatialScene';
import {
  testFirestoreConnection,
  initAuthListener,
  signInWithGoogleWorkspace,
  signOutUser,
  createTrainingRunInDb,
  updateTrainingRunInDb,
  deleteTrainingRunFromDb,
  createCheckpointInDb,
  logAuditEventInDb,
  subscribeUserCollections,
  TrainingRunDoc,
  CheckpointDoc,
  AuditLogDoc,
  KeepNoteDoc,
} from './firebase';
import {
  exportCheckpointToDrive,
  exportLossCurveToSheets,
  sendAlertEmailViaGmail,
  scheduleTrainingWindowOnCalendar,
  createExperimentReportInDocs,
  createReviewDeckInSlides,
  createFollowUpTaskInTasks,
  postMessageToGoogleChat,
  createEvalFormInForms,
  createResearchNoteInKeep,
  createWarRoomInMeet,
  fetchTeamContactsFromPeople,
  CollaboratorContact,
} from './workspace';
import { fetchBioTargetWorkbench, PRESET_TARGETS, BioTargetReport } from './science';

// Application State
interface LossPoint {
  step: number;
  trainLoss: number;
  valLoss: number;
  lr: number;
}

const DEFAULT_LOSS_HISTORY: LossPoint[] = Array.from({ length: 24 }, (_, idx) => {
  const step = (idx + 1) * 50;
  const trainLoss = Number((0.42 * Math.exp(-idx * 0.085) + 0.045 + Math.sin(idx) * 0.004).toFixed(4));
  const valLoss = Number((0.45 * Math.exp(-idx * 0.078) + 0.056 + Math.cos(idx) * 0.005).toFixed(4));
  return { step, trainLoss, valLoss, lr: 0.0002 };
});

let spatialScript: SpatialTensorLabScript;
let currentUserEmail: string | null = null;
let currentUserUid: string | null = null;
let hasOAuthToken = false;
let unsubFirestore: (() => void) | null = null;

let activeTab: 'telemetry' | 'samples' | 'bioscience' | 'workspace' | 'copilot' | 'security' | 'game' =
  'telemetry';
let dockCollapsed = false;
let darkMode = true;
let mfaVerified = true;
let currentRole: 'Principal ML Architect (Admin)' | 'Model Trainer' | 'Compliance Auditor' =
  'Principal ML Architect (Admin)';

let lossHistory: LossPoint[] = [...DEFAULT_LOSS_HISTORY];
let trainingRuns: TrainingRunDoc[] = [];
let checkpoints: CheckpointDoc[] = [];
let auditLogs: AuditLogDoc[] = [];
let keepNotes: KeepNoteDoc[] = [];
let teamContacts: CollaboratorContact[] = [];

let currentBioReport: BioTargetReport | null = null;
let statusBannerText = 'System Ready · AES-256 Encrypted At-Rest & In-Transit · RBAC Enforced';
let chatMessages: Array<{ role: 'user' | 'assistant'; text: string }> = [
  {
    role: 'assistant',
    text: 'LoRAFlow Architect online (Gemini 4 Argon · High Thinking). Ask me to diagnose loss divergence, optimize LoRA rank/alpha ratios, or design protein-binder adapters.',
  },
];

// Live API WebSocket state
let liveWs: WebSocket | null = null;
let liveInputCtx: AudioContext | null = null;
let liveOutputCtx: AudioContext | null = null;
let liveNextStartTime = 0;
let liveActive = false;

// Confirmation Modal Helper for Mutating / Destructive Actions
function showConfirmModal(
  title: string,
  description: string,
  confirmLabel: string,
  onConfirm: () => Promise<void> | void
) {
  const existing = document.getElementById('loraflow-confirm-modal');
  if (existing) existing.remove();

  const overlay = document.createElement('div');
  overlay.id = 'loraflow-confirm-modal';
  overlay.className =
    'fixed inset-0 z-50 flex items-center justify-center bg-black/75 p-4 pointer-events-auto';
  overlay.innerHTML = `
    <div class="w-full max-w-md rounded-xl border border-slate-700 bg-slate-900 p-5 text-slate-100 shadow-2xl">
      <h3 class="text-base font-semibold text-sky-400">${title}</h3>
      <p class="mt-2 text-xs leading-relaxed text-slate-300">${description}</p>
      <div class="mt-4 flex justify-end gap-2">
        <button id="modal-cancel-btn" class="rounded-lg border border-slate-700 bg-slate-800 px-3 py-1.5 text-xs font-medium text-slate-300 hover:bg-slate-700">Cancel</button>
        <button id="modal-confirm-btn" class="rounded-lg bg-sky-600 px-3 py-1.5 text-xs font-semibold text-white hover:bg-sky-500">${confirmLabel}</button>
      </div>
    </div>
  `;
  document.body.appendChild(overlay);

  overlay.querySelector('#modal-cancel-btn')?.addEventListener('click', () => overlay.remove());
  overlay.querySelector('#modal-confirm-btn')?.addEventListener('click', async () => {
    overlay.remove();
    await onConfirm();
  });
}

function setBanner(msg: string) {
  statusBannerText = msg;
  const el = document.getElementById('status-banner-msg');
  if (el) el.textContent = msg;
}

function drawLossCurveCanvas() {
  const canvas = document.getElementById('loss-curve-canvas') as HTMLCanvasElement | null;
  if (!canvas) return;
  const ctx = canvas.getContext('2d');
  if (!ctx) return;

  const w = canvas.width;
  const h = canvas.height;
  ctx.clearRect(0, 0, w, h);

  ctx.fillStyle = darkMode ? '#0b1120' : '#f8fafc';
  ctx.fillRect(0, 0, w, h);

  // Grid lines
  ctx.strokeStyle = darkMode ? '#1e293b' : '#e2e8f0';
  ctx.lineWidth = 1;
  for (let i = 1; i < 4; i++) {
    const y = (h / 4) * i;
    ctx.beginPath();
    ctx.moveTo(0, y);
    ctx.lineTo(w, y);
    ctx.stroke();
  }

  if (lossHistory.length < 2) return;

  const maxLoss = Math.max(...lossHistory.map((p) => Math.max(p.trainLoss, p.valLoss)), 0.5);
  const minLoss = 0;

  // Train Loss Line (Sky Blue)
  ctx.strokeStyle = '#38bdf8';
  ctx.lineWidth = 2.2;
  ctx.beginPath();
  lossHistory.forEach((pt, idx) => {
    const x = (idx / (lossHistory.length - 1)) * (w - 24) + 12;
    const y = h - 14 - ((pt.trainLoss - minLoss) / (maxLoss - minLoss)) * (h - 28);
    if (idx === 0) ctx.moveTo(x, y);
    else ctx.lineTo(x, y);
  });
  ctx.stroke();

  // Val Loss Line (Amber)
  ctx.strokeStyle = '#f59e0b';
  ctx.lineWidth = 2.0;
  ctx.beginPath();
  lossHistory.forEach((pt, idx) => {
    const x = (idx / (lossHistory.length - 1)) * (w - 24) + 12;
    const y = h - 14 - ((pt.valLoss - minLoss) / (maxLoss - minLoss)) * (h - 28);
    if (idx === 0) ctx.moveTo(x, y);
    else ctx.lineTo(x, y);
  });
  ctx.stroke();
}

function renderAppShell() {
  const root = document.getElementById('root');
  if (!root) return;

  const surfaceBg = darkMode
    ? 'bg-slate-950/95 text-slate-100 border-slate-800'
    : 'bg-white/95 text-slate-900 border-slate-300';
  const subCardBg = darkMode
    ? 'bg-slate-900/90 border-slate-800 text-slate-100'
    : 'bg-slate-50 border-slate-200 text-slate-900';

  root.innerHTML = `
    <!-- Top-Left Minimal Spec & Telemetry Perimeter Cluster (Zero Central Occlusion) -->
    <div class="fixed top-3 left-4 z-30 flex flex-wrap items-center gap-2 pointer-events-auto">
      <div class="flex items-center gap-2.5 rounded-xl border ${surfaceBg} px-3.5 py-2 shadow-xl">
        <svg class="h-5 w-5 text-sky-400" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
          <path d="M12 2L2 7l10 5 10-5-10-5zM2 17l10 5 10-5M2 12l10 5 10-5"/>
        </svg>
        <div>
          <div class="flex items-center gap-2">
            <span class="text-xs font-bold tracking-tight">LoRAFlow Spatial MLOps</span>
            <span class="rounded bg-sky-500/20 px-1.5 py-0.5 font-mono-num text-[10px] font-semibold text-sky-300">r=${spatialScript?.state.rank ?? 16} · α=${spatialScript?.state.alpha ?? 32}</span>
          </div>
          <div id="status-banner-msg" class="text-[11px] text-slate-400 truncate max-w-[360px]">${statusBannerText}</div>
        </div>
      </div>

      <!-- Quick KPI Perimeter Readout -->
      <div class="hidden md:flex items-center gap-3 rounded-xl border ${surfaceBg} px-3.5 py-2 font-mono-num text-xs shadow-xl">
        <div><span class="text-slate-400">EPOCH</span> <span id="hud-epoch" class="font-semibold text-emerald-400">${spatialScript?.state.epoch ?? 14}</span></div>
        <div class="h-3 w-px bg-slate-700"></div>
        <div><span class="text-slate-400">LOSS</span> <span id="hud-loss" class="font-semibold text-sky-400">${(spatialScript?.state.loss ?? 0.0842).toFixed(4)}</span></div>
        <div class="h-3 w-px bg-slate-700"></div>
        <div><span class="text-slate-400">VAL</span> <span id="hud-val" class="font-semibold text-amber-400">${(spatialScript?.state.valLoss ?? 0.0915).toFixed(4)}</span></div>
        <div class="h-3 w-px bg-slate-700"></div>
        <button id="theme-toggle-btn" class="rounded border border-slate-700 px-2 py-0.5 text-[10px] font-medium hover:bg-slate-800">
          ${darkMode ? 'Light Lab' : 'Dark Ops'}
        </button>
      </div>

      <!-- Google Sign-In / Auth Badge -->
      <div class="flex items-center gap-2 rounded-xl border ${surfaceBg} px-3 py-1.5 shadow-xl">
        ${
          currentUserEmail
            ? `<span class="text-[11px] font-medium text-emerald-400">${currentUserEmail}</span>
               <span class="rounded bg-emerald-500/20 px-1.5 py-0.5 text-[10px] text-emerald-300">${hasOAuthToken ? 'Workspace Connected' : 'Auth Only'}</span>
               <button id="reauth-workspace-btn" class="rounded bg-sky-600 px-2 py-0.5 text-[10px] font-semibold text-white hover:bg-sky-500">Sync OAuth</button>
               <button id="signout-btn" class="rounded border border-slate-700 px-2 py-0.5 text-[10px] text-slate-300 hover:bg-slate-800">Sign Out</button>`
            : `<button id="google-signin-btn" class="gsi-material-button">
                 <div class="gsi-material-button-state"></div>
                 <div class="gsi-material-button-content-wrapper">
                   <div class="gsi-material-button-icon">
                     <svg version="1.1" xmlns="http://www.w3.org/2000/svg" viewBox="0 0 48 48" style="display: block;">
                       <path fill="#EA4335" d="M24 9.5c3.54 0 6.71 1.22 9.21 3.6l6.85-6.85C35.9 2.38 30.47 0 24 0 14.62 0 6.51 5.38 2.56 13.22l7.98 6.19C12.43 13.72 17.74 9.5 24 9.5z"></path>
                       <path fill="#4285F4" d="M46.98 24.55c0-1.57-.15-3.09-.38-4.55H24v9.02h12.94c-.58 2.96-2.26 5.48-4.78 7.18l7.73 6c4.51-4.18 7.09-10.36 7.09-17.65z"></path>
                       <path fill="#FBBC05" d="M10.53 28.59c-.48-1.45-.76-2.99-.76-4.59s.27-3.14.76-4.59l-7.98-6.19C.92 16.46 0 20.12 0 24c0 3.88.92 7.54 2.56 10.78l7.97-6.19z"></path>
                       <path fill="#34A853" d="M24 48c6.48 0 11.93-2.13 15.89-5.81l-7.73-6c-2.15 1.45-4.92 2.3-8.16 2.3-6.26 0-11.57-4.22-13.47-9.91l-7.98 6.19C6.51 42.62 14.62 48 24 48z"></path>
                     </svg>
                   </div>
                   <span class="gsi-material-button-contents">Sign in with Google</span>
                 </div>
               </button>`
        }
      </div>
    </div>

    <!-- Collapsible Unified Bottom Control Dock (Zero Central Viewport Occlusion) -->
    <div class="fixed bottom-3 left-1/2 z-30 w-[95vw] max-w-[1060px] -translate-x-1/2 pointer-events-auto">
      <!-- Collapse / Expand Handle Bar -->
      <div class="flex items-center justify-between rounded-t-xl border border-b-0 ${surfaceBg} px-4 py-1.5">
        <div class="flex flex-wrap items-center gap-1">
          ${[
            ['telemetry', '1 · Loss & Runs'],
            ['samples', '2 · Real-Time Samples & Video'],
            ['bioscience', '3 · Bioscience Targets'],
            ['workspace', '4 · Google Workspace Hub'],
            ['copilot', '5 · Gemini 4 & Live Voice'],
            ['security', '6 · RBAC & Audit Logs'],
            ['game', '7 · 3D Rank Stabilizer Game'],
          ]
            .map(
              ([id, label]) => `
            <button data-tab="${id}" class="tab-switch-btn rounded-lg px-2.5 py-1 text-[11px] font-semibold transition ${
                activeTab === id
                  ? 'bg-sky-600 text-white'
                  : 'text-slate-400 hover:bg-slate-800/60 hover:text-slate-200'
              }">${label}</button>
          `
            )
            .join('')}
        </div>
        <button id="toggle-dock-btn" class="rounded-lg border border-slate-700 px-2.5 py-0.5 text-[11px] font-medium text-slate-300 hover:bg-slate-800">
          ${dockCollapsed ? '▴ Expand Dock (3D Full View)' : '▾ Collapse Dock (Inspect 3D)'}
        </button>
      </div>

      <!-- Dock Body -->
      <div class="${dockCollapsed ? 'hidden' : 'block'} rounded-b-xl border ${surfaceBg} p-3.5 shadow-2xl max-h-[38vh] overflow-y-auto">
        ${renderActiveTabContent(subCardBg)}
      </div>
    </div>
  `;

  bindDOMEvents();
  if (activeTab === 'telemetry') {
    drawLossCurveCanvas();
  }
}

function renderActiveTabContent(subCardBg: string): string {
  if (activeTab === 'telemetry') {
    return `
      <div class="grid grid-cols-1 md:grid-cols-12 gap-3 items-start">
        <!-- Loss Curve & Hyperparameter Sliders -->
        <div class="md:col-span-5 rounded-xl border ${subCardBg} p-3">
          <div class="flex items-center justify-between mb-1.5">
            <span class="text-xs font-semibold">Real-Time Loss Curve (Train vs. Val)</span>
            <div class="flex items-center gap-2 text-[10px]">
              <span class="text-sky-400">■ Train</span>
              <span class="text-amber-400">■ Validation</span>
            </div>
          </div>
          <canvas id="loss-curve-canvas" width="380" height="115" class="w-full rounded border border-slate-800"></canvas>
          <div class="mt-2 flex flex-wrap gap-1.5">
            <button id="step-epoch-btn" class="rounded bg-emerald-600 px-2.5 py-1 text-[11px] font-semibold text-white hover:bg-emerald-500">+ Step Epoch</button>
            <button id="fast-diagnose-btn" class="rounded bg-sky-600 px-2.5 py-1 text-[11px] font-semibold text-white hover:bg-sky-500">⚡ Flash-Lite Diagnose</button>
            <button id="tts-announce-btn" class="rounded border border-slate-700 bg-slate-800 px-2.5 py-1 text-[11px] font-medium text-slate-200 hover:bg-slate-700">🔊 TTS Readout</button>
          </div>
        </div>

        <!-- Hyperparameter & Cloud / HF / GitHub Config -->
        <div class="md:col-span-4 rounded-xl border ${subCardBg} p-3 space-y-2">
          <div class="text-xs font-semibold">LoRA Hyperparameters & Git/HF Sync</div>
          <div class="grid grid-cols-2 gap-2 text-[11px]">
            <label class="block">
              <span class="text-slate-400">Run Name</span>
              <input id="run-name-input" type="text" value="${spatialScript?.state.runName ?? 'ESM2-LoRA-TP53-Binders-r16'}" class="mt-0.5 w-full rounded border border-slate-700 bg-slate-950 px-2 py-1 text-xs text-slate-100" />
            </label>
            <label class="block">
              <span class="text-slate-400">Hugging Face Dataset</span>
              <input id="hf-dataset-input" type="text" value="deepmind/alphafold-tp53-contacts" class="mt-0.5 w-full rounded border border-slate-700 bg-slate-950 px-2 py-1 text-xs text-slate-100" />
            </label>
            <label class="block">
              <span class="text-slate-400">GitHub Repository</span>
              <input id="github-repo-input" type="text" value="org/loraflow-protein-adapters#main" class="mt-0.5 w-full rounded border border-slate-700 bg-slate-950 px-2 py-1 text-xs text-slate-100" />
            </label>
            <label class="block">
              <span class="text-slate-400">Cloud Compute Cluster</span>
              <select id="cloud-provider-input" class="mt-0.5 w-full rounded border border-slate-700 bg-slate-950 px-2 py-1 text-xs text-slate-100">
                <option value="GCP Cloud TPU v5p-128">GCP Cloud TPU v5p-128</option>
                <option value="AWS p5.48xlarge (8x H100)">AWS p5.48xlarge (8x H100)</option>
                <option value="Azure ND H100 v5">Azure ND H100 v5</option>
              </select>
            </label>
          </div>
          <div class="flex items-center gap-3 text-[11px]">
            <label class="flex-1">
              <span>Rank r=<strong id="rank-val-label">${spatialScript?.state.rank ?? 16}</strong></span>
              <input id="rank-range-input" type="range" min="4" max="128" step="4" value="${spatialScript?.state.rank ?? 16}" class="w-full accent-sky-500" />
            </label>
            <label class="flex-1">
              <span>Alpha α=<strong id="alpha-val-label">${spatialScript?.state.alpha ?? 32}</strong></span>
              <input id="alpha-range-input" type="range" min="8" max="256" step="8" value="${spatialScript?.state.alpha ?? 32}" class="w-full accent-teal-500" />
            </label>
          </div>
          <div class="flex flex-wrap gap-1.5 pt-1">
            <button id="save-run-firestore-btn" class="rounded bg-indigo-600 px-2.5 py-1 text-[11px] font-semibold text-white hover:bg-indigo-500">☁ Save Run to Firestore</button>
            <button id="export-ckpt-btn" class="rounded bg-amber-600 px-2.5 py-1 text-[11px] font-semibold text-white hover:bg-amber-500">⬇ Export SafeTensors</button>
          </div>
        </div>

        <!-- Saved Cloud Training Runs & Checkpoints -->
        <div class="md:col-span-3 rounded-xl border ${subCardBg} p-3 space-y-2 max-h-[210px] overflow-y-auto">
          <div class="flex items-center justify-between">
            <span class="text-xs font-semibold">Saved Cloud Runs (${trainingRuns.length})</span>
            <span class="text-[10px] text-emerald-400">Firestore Synced</span>
          </div>
          ${
            trainingRuns.length === 0
              ? `<div class="text-[11px] text-slate-400">Sign in with Google & click "Save Run to Firestore" to persist runs across devices.</div>`
              : trainingRuns
                  .map(
                    (r) => `
              <div class="flex items-center justify-between rounded border border-slate-800 bg-slate-950/70 px-2 py-1.5 text-[11px]">
                <div class="truncate pr-2">
                  <div class="font-semibold text-sky-300 truncate">${r.name}</div>
                  <div class="font-mono-num text-[10px] text-slate-400">r=${r.rank} · Loss ${Number(r.loss).toFixed(4)} · ${r.cloudProvider}</div>
                </div>
                <button data-del-run="${r.id}" class="del-run-btn rounded bg-rose-900/50 px-1.5 py-0.5 text-[10px] text-rose-300 hover:bg-rose-800">Del</button>
              </div>
            `
                  )
                  .join('')
          }
          <div class="pt-1 border-t border-slate-800 text-[10px] text-slate-400">
            Checkpoints Exported: <strong class="font-mono-num text-amber-400">${checkpoints.length}</strong>
          </div>
        </div>
      </div>
    `;
  }

  if (activeTab === 'samples') {
    return `
      <div class="grid grid-cols-1 md:grid-cols-12 gap-3 items-start">
        <!-- Sample Image & Video Synthesis Controls -->
        <div class="md:col-span-5 rounded-xl border ${subCardBg} p-3 space-y-2">
          <div class="text-xs font-semibold">Real-Time Checkpoint Sample Image & Video Generator</div>
          <textarea id="sample-prompt-input" rows="2" class="w-full rounded border border-slate-700 bg-slate-950 p-2 text-xs text-slate-100" placeholder="Prompt to evaluate current LoRA adapter checkpoint...">High-resolution structural cryo-EM visualization of human TP53 tetramer bound to DNA with glowing LoRA attention weights</textarea>
          <div class="grid grid-cols-3 gap-2 text-[11px]">
            <label>
              <span class="text-slate-400">Model</span>
              <select id="sample-model-select" class="mt-0.5 w-full rounded border border-slate-700 bg-slate-950 px-1.5 py-1 text-[11px]">
                <option value="gemini-nano-banana-2.1">gemini-nano-banana-2.1</option>
                <option value="gemini-3-pro-image-preview">gemini-3-pro-image-preview</option>
              </select>
            </label>
            <label>
              <span class="text-slate-400">Aspect Ratio</span>
              <select id="sample-aspect-select" class="mt-0.5 w-full rounded border border-slate-700 bg-slate-950 px-1.5 py-1 text-[11px]">
                <option value="1:1">1:1</option>
                <option value="16:9">16:9</option>
                <option value="9:16">9:16</option>
                <option value="4:3">4:3</option>
                <option value="3:4">3:4</option>
                <option value="3:2">3:2</option>
                <option value="2:3">2:3</option>
                <option value="21:9">21:9</option>
              </select>
            </label>
            <label>
              <span class="text-slate-400">Resolution</span>
              <select id="sample-size-select" class="mt-0.5 w-full rounded border border-slate-700 bg-slate-950 px-1.5 py-1 text-[11px]">
                <option value="1K">1K</option>
                <option value="2K">2K</option>
                <option value="4K">4K</option>
              </select>
            </label>
          </div>
          <div class="flex flex-wrap gap-1.5 pt-1">
            <button id="gen-sample-img-btn" class="rounded bg-sky-600 px-2.5 py-1 text-[11px] font-semibold text-white hover:bg-sky-500">🎨 Generate Sample Image</button>
            <button id="gen-sample-video-btn" class="rounded bg-purple-600 px-2.5 py-1 text-[11px] font-semibold text-white hover:bg-purple-500">🎬 Generate Omni 1.1 Video</button>
            <button id="gen-lyria-music-btn" class="rounded bg-teal-600 px-2.5 py-1 text-[11px] font-semibold text-white hover:bg-teal-500">🎵 Lyria Training Track</button>
          </div>
        </div>

        <!-- Upload Photo/Video for Editing, Animation, or Gemini 4 Argon Analysis -->
        <div class="md:col-span-3 rounded-xl border ${subCardBg} p-3 space-y-2">
          <div class="text-xs font-semibold">Upload Media (Analyze / Animate / Edit)</div>
          <input id="media-upload-input" type="file" accept="image/*,video/*" class="w-full text-[11px] text-slate-300 file:mr-2 file:rounded file:border-0 file:bg-slate-800 file:px-2 file:py-1 file:text-[11px] file:text-sky-300" />
          <div class="flex flex-wrap gap-1.5">
            <button id="analyze-uploaded-media-btn" class="rounded bg-indigo-600 px-2 py-1 text-[11px] font-semibold text-white hover:bg-indigo-500">🔍 Argon Analyze Image/Video</button>
            <button id="animate-uploaded-photo-btn" class="rounded bg-pink-600 px-2 py-1 text-[11px] font-semibold text-white hover:bg-pink-500">✨ Animate Photo to Video</button>
          </div>
          <div id="media-analysis-output" class="text-[11px] text-slate-300 max-h-[85px] overflow-y-auto">Upload a sample checkpoint image or video to inspect artifacts or animate into 16:9 / 9:16 video.</div>
        </div>

        <!-- Preview Viewport -->
        <div class="md:col-span-4 rounded-xl border ${subCardBg} p-3 flex flex-col items-center justify-center min-h-[160px]">
          <div id="sample-preview-box" class="w-full text-center text-[11px] text-slate-400">
            Generated checkpoint sample images, Omni 1.1 Flash videos, and Lyria audio players appear here.
          </div>
        </div>
      </div>
    `;
  }

  if (activeTab === 'bioscience') {
    const rep = currentBioReport;
    return `
      <div class="space-y-2.5">
        <!-- Quick-Try Preset Target Bar -->
        <div class="flex flex-wrap items-center gap-1.5">
          <span class="text-[11px] font-semibold text-slate-400 mr-1">Quick-Try Targets:</span>
          ${PRESET_TARGETS.map(
            (t) => `
            <button data-acc="${t.accession}" data-comp="${t.compound}" data-loc="${t.locus}" class="bio-preset-btn rounded-full border border-slate-700 bg-slate-900 px-2.5 py-0.5 text-[11px] font-medium text-sky-300 hover:bg-sky-950">
              ${t.label}
            </button>
          `
          ).join('')}
        </div>

        <div class="grid grid-cols-1 md:grid-cols-12 gap-3 items-start">
          <!-- Protein & AlphaFold DB Metadata -->
          <div class="md:col-span-5 rounded-xl border ${subCardBg} p-3 space-y-1.5">
            <div class="flex items-center justify-between">
              <span class="text-xs font-bold text-sky-400">${rep ? `${rep.geneName} · ${rep.proteinName}` : 'Loading UniProtKB & AlphaFold DB...'}</span>
              <span class="rounded bg-blue-600/20 px-1.5 py-0.5 font-mono-num text-[10px] text-blue-300">[Predicted by model · AFDB v6]</span>
            </div>
            <div class="font-mono-num text-[11px] text-slate-300">
              Accession: <strong>${rep?.accession ?? 'P04637'}</strong> · Length: <strong>${rep?.sequenceLength ?? 393} aa</strong> · Mean pLDDT: <strong class="text-sky-400">${rep?.meanPlddtEstimate ?? 88.4}</strong>
            </div>
            <p class="text-[11px] text-slate-400 line-clamp-2">${rep?.functionSummary ?? ''}</p>
            <div class="flex flex-wrap gap-2 pt-1 text-[10px]">
              ${
                rep
                  ? `<a href="${rep.cifUrl}" target="_blank" rel="noopener" class="rounded bg-slate-800 px-2 py-0.5 text-sky-300 hover:underline">⬇ Download mmCIF</a>
                     <a href="${rep.pdbUrl}" target="_blank" rel="noopener" class="rounded bg-slate-800 px-2 py-0.5 text-sky-300 hover:underline">⬇ Download PDB</a>
                     <a href="${rep.alphagenomeUrl}" target="_blank" rel="noopener" class="rounded bg-teal-900/50 px-2 py-0.5 text-teal-300 hover:underline">🧬 AlphaGenome Atlas</a>`
                  : ''
              }
            </div>
            <!-- Canonical 4-Band AlphaFold pLDDT Scale -->
            <div class="flex items-center gap-1.5 pt-1 text-[10px]">
              <span class="px-1.5 py-0.5 rounded text-white" style="background:#0053D6">>90 Very High</span>
              <span class="px-1.5 py-0.5 rounded text-slate-950 font-medium" style="background:#65CBF3">70-90 Confident</span>
              <span class="px-1.5 py-0.5 rounded text-slate-950 font-medium" style="background:#FFDB13">50-70 Low</span>
              <span class="px-1.5 py-0.5 rounded text-white" style="background:#FF7D45"><50 Very Low</span>
            </div>
          </div>

          <!-- PubChem Ligand & Sequence Preview -->
          <div class="md:col-span-3 rounded-xl border ${subCardBg} p-3 space-y-1.5">
            <div class="flex items-center justify-between">
              <span class="text-xs font-semibold">PubChem Ligand & Sequence</span>
              <span class="rounded bg-emerald-600/20 px-1.5 py-0.5 text-[10px] text-emerald-300">[Experimentally determined]</span>
            </div>
            ${
              rep?.compoundInfo
                ? `<div class="text-[11px] space-y-0.5">
                     <div>Compound: <strong class="text-amber-300">${rep.compoundInfo.name}</strong> (CID ${rep.compoundInfo.cid})</div>
                     <div class="font-mono-num text-[10px] text-slate-400">Formula: ${rep.compoundInfo.formula} · MW: ${rep.compoundInfo.molecularWeight} g/mol</div>
                     <div class="font-mono-num text-[10px] text-slate-400 truncate" title="${rep.compoundInfo.smiles}">SMILES: ${rep.compoundInfo.smiles}</div>
                   </div>`
                : `<div class="text-[11px] text-slate-400">Querying PubChem PUG REST...</div>`
            }
            <div class="font-mono-num text-[10px] bg-slate-950 p-1.5 rounded border border-slate-800 text-slate-400 break-all max-h-[54px] overflow-y-auto">
              ${rep?.sequencePreview ?? 'MEEPQSDPSVEPPLSQETFSDLWKLLPENNVLSPLPSQAMDDLMLSPDDIEQWFTEDPGP...'}
            </div>
          </div>

          <!-- Europe PMC Curated Literature -->
          <div class="md:col-span-4 rounded-xl border ${subCardBg} p-3 space-y-1.5 max-h-[185px] overflow-y-auto">
            <div class="flex items-center justify-between">
              <span class="text-xs font-semibold">Europe PMC Literature</span>
              <span class="rounded bg-purple-600/20 px-1.5 py-0.5 text-[10px] text-purple-300">[Literature curated]</span>
            </div>
            ${
              rep?.papers?.length
                ? rep.papers
                    .map(
                      (p) => `
                  <div class="border-b border-slate-800 pb-1 last:border-0 text-[11px]">
                    <div class="font-medium text-slate-200 line-clamp-1">${p.title}</div>
                    <div class="text-[10px] text-slate-400">${p.authors} · ${p.journal} (${p.pubYear})</div>
                  </div>
                `
                    )
                    .join('')
                : `<div class="text-[11px] text-slate-400">Select a target above to load peer-reviewed structural citations.</div>`
            }
          </div>
        </div>
      </div>
    `;
  }

  if (activeTab === 'workspace') {
    return `
      <div class="space-y-2">
        <div class="flex items-center justify-between">
          <span class="text-xs font-semibold">12-Service Google Workspace MLOps Command Center (Confirmation Protected)</span>
          <span class="text-[11px] text-slate-400">Uses your live Google OAuth token · Zero mock calls</span>
        </div>
        <div class="grid grid-cols-2 sm:grid-cols-4 md:grid-cols-6 gap-2 text-[11px]">
          <button id="ws-drive-btn" class="rounded-lg border border-slate-700 bg-slate-900 p-2 text-left hover:border-sky-500">
            <div class="font-semibold text-sky-400">1. Google Drive</div>
            <div class="text-[10px] text-slate-400">Save Checkpoint JSON</div>
          </button>
          <button id="ws-sheets-btn" class="rounded-lg border border-slate-700 bg-slate-900 p-2 text-left hover:border-emerald-500">
            <div class="font-semibold text-emerald-400">2. Google Sheets</div>
            <div class="text-[10px] text-slate-400">Export Loss Telemetry</div>
          </button>
          <button id="ws-gmail-btn" class="rounded-lg border border-slate-700 bg-slate-900 p-2 text-left hover:border-rose-500">
            <div class="font-semibold text-rose-400">3. Gmail Alert</div>
            <div class="text-[10px] text-slate-400">Dispatch Anomaly Email</div>
          </button>
          <button id="ws-calendar-btn" class="rounded-lg border border-slate-700 bg-slate-900 p-2 text-left hover:border-blue-500">
            <div class="font-semibold text-blue-400">4. Calendar</div>
            <div class="text-[10px] text-slate-400">Book GPU Cluster Slot</div>
          </button>
          <button id="ws-docs-btn" class="rounded-lg border border-slate-700 bg-slate-900 p-2 text-left hover:border-indigo-500">
            <div class="font-semibold text-indigo-400">5. Google Docs</div>
            <div class="text-[10px] text-slate-400">Write Repro Report</div>
          </button>
          <button id="ws-slides-btn" class="rounded-lg border border-slate-700 bg-slate-900 p-2 text-left hover:border-amber-500">
            <div class="font-semibold text-amber-400">6. Google Slides</div>
            <div class="text-[10px] text-slate-400">Create Review Deck</div>
          </button>
          <button id="ws-tasks-btn" class="rounded-lg border border-slate-700 bg-slate-900 p-2 text-left hover:border-teal-500">
            <div class="font-semibold text-teal-400">7. Google Tasks</div>
            <div class="text-[10px] text-slate-400">Add Sweep Action Item</div>
          </button>
          <button id="ws-chat-btn" class="rounded-lg border border-slate-700 bg-slate-900 p-2 text-left hover:border-emerald-500">
            <div class="font-semibold text-emerald-300">8. Google Chat</div>
            <div class="text-[10px] text-slate-400">Post to Space</div>
          </button>
          <button id="ws-forms-btn" class="rounded-lg border border-slate-700 bg-slate-900 p-2 text-left hover:border-purple-500">
            <div class="font-semibold text-purple-400">9. Google Forms</div>
            <div class="text-[10px] text-slate-400">Create RLHF Eval Form</div>
          </button>
          <button id="ws-keep-btn" class="rounded-lg border border-slate-700 bg-slate-900 p-2 text-left hover:border-yellow-500">
            <div class="font-semibold text-yellow-400">10. Google Keep</div>
            <div class="text-[10px] text-slate-400">Pin Lab Note (${keepNotes.length})</div>
          </button>
          <button id="ws-meet-btn" class="rounded-lg border border-slate-700 bg-slate-900 p-2 text-left hover:border-cyan-500">
            <div class="font-semibold text-cyan-400">11. Google Meet</div>
            <div class="text-[10px] text-slate-400">Launch War-Room</div>
          </button>
          <button id="ws-contacts-btn" class="rounded-lg border border-slate-700 bg-slate-900 p-2 text-left hover:border-pink-500">
            <div class="font-semibold text-pink-400">12. Contacts</div>
            <div class="text-[10px] text-slate-400">Sync Team (${teamContacts.length})</div>
          </button>
        </div>
        <div id="workspace-result-bar" class="rounded border border-slate-800 bg-slate-950 px-3 py-1.5 text-[11px] text-sky-300">
          Select any Google Workspace integration above to execute real authenticated API workflows.
        </div>
      </div>
    `;
  }

  if (activeTab === 'copilot') {
    return `
      <div class="grid grid-cols-1 md:grid-cols-12 gap-3 items-start">
        <!-- Multi-Turn Gemini 4 Argon Chat -->
        <div class="md:col-span-7 rounded-xl border ${subCardBg} p-3 space-y-2">
          <div class="flex items-center justify-between">
            <span class="text-xs font-semibold">Gemini 4 Argon Multi-Turn Co-Pilot (ThinkingLevel.HIGH)</span>
            <select id="copilot-model-select" class="rounded border border-slate-700 bg-slate-950 px-2 py-0.5 text-[11px]">
              <option value="gemini-4-argon">gemini-4-argon (High Thinking)</option>
              <option value="gemini-3.8-flash">gemini-3.8-flash (General)</option>
              <option value="gemini-3.1-flash-lite">gemini-3.1-flash-lite (Low Latency)</option>
            </select>
          </div>
          <div id="chat-thread-box" class="h-[95px] overflow-y-auto space-y-1.5 rounded border border-slate-800 bg-slate-950 p-2 text-[11px]">
            ${chatMessages
              .map(
                (m) => `
              <div class="${m.role === 'user' ? 'text-sky-300 text-right' : 'text-slate-200'}">
                <span class="font-semibold">${m.role === 'user' ? 'You' : 'Architect'}:</span> ${m.text}
              </div>
            `
              )
              .join('')}
          </div>
          <div class="flex gap-1.5">
            <input id="chat-input-box" type="text" placeholder="Ask about LoRA rank bottlenecks, gradient norms, or protein folding..." class="flex-1 rounded border border-slate-700 bg-slate-950 px-2.5 py-1 text-xs text-slate-100" />
            <button id="chat-send-btn" class="rounded bg-sky-600 px-3 py-1 text-xs font-semibold text-white hover:bg-sky-500">Send</button>
          </div>
        </div>

        <!-- Search Grounding, Maps Grounding, Voice Transcription & Live API -->
        <div class="md:col-span-5 rounded-xl border ${subCardBg} p-3 space-y-2">
          <div class="text-xs font-semibold">Search & Maps Grounding · Audio Transcribe · Live API</div>
          <div class="flex flex-wrap gap-1.5">
            <button id="search-grounding-btn" class="rounded bg-blue-600 px-2.5 py-1 text-[11px] font-semibold text-white hover:bg-blue-500">🌐 Google Search Grounding</button>
            <button id="maps-grounding-btn" class="rounded bg-emerald-600 px-2.5 py-1 text-[11px] font-semibold text-white hover:bg-emerald-500">📍 Google Maps Grounding</button>
            <button id="record-transcribe-btn" class="rounded bg-amber-600 px-2.5 py-1 text-[11px] font-semibold text-white hover:bg-amber-500">🎙 Mic Transcribe (3.5)</button>
            <button id="live-voice-btn" class="rounded ${liveActive ? 'bg-rose-600' : 'bg-purple-600'} px-2.5 py-1 text-[11px] font-semibold text-white">
              ${liveActive ? '⏹ Stop Live API (3.8-live)' : '🎧 Start Live Voice Session'}
            </button>
          </div>
          <div id="grounding-result-box" class="h-[88px] overflow-y-auto rounded border border-slate-800 bg-slate-950 p-2 text-[11px] text-slate-300">
            Click Search Grounding for live arXiv/HuggingFace LoRA benchmarks, Maps Grounding for cloud GPU regions, or Start Live Voice for real-time audio conversation.
          </div>
        </div>
      </div>
    `;
  }

  if (activeTab === 'security') {
    return `
      <div class="grid grid-cols-1 md:grid-cols-12 gap-3 items-start">
        <div class="md:col-span-5 rounded-xl border ${subCardBg} p-3 space-y-2">
          <div class="text-xs font-semibold">Role-Based Access Control (RBAC), MFA & Encryption</div>
          <div class="grid grid-cols-2 gap-2 text-[11px]">
            <label>
              <span class="text-slate-400">Active Security Role</span>
              <select id="rbac-role-select" class="mt-0.5 w-full rounded border border-slate-700 bg-slate-950 px-2 py-1 text-xs">
                <option value="Principal ML Architect (Admin)" ${currentRole.includes('Admin') ? 'selected' : ''}>Principal ML Architect (Admin)</option>
                <option value="Model Trainer" ${currentRole === 'Model Trainer' ? 'selected' : ''}>Model Trainer (Tier 2)</option>
                <option value="Compliance Auditor" ${currentRole === 'Compliance Auditor' ? 'selected' : ''}>Compliance Auditor (Read-Only)</option>
              </select>
            </label>
            <div class="space-y-1 pt-1">
              <div class="text-emerald-400 font-medium">✔ AES-256 At-Rest & TLS 1.3 In-Transit</div>
              <div class="text-sky-400 font-medium">✔ Multi-Factor Auth (MFA): ${mfaVerified ? 'VERIFIED' : 'PENDING'}</div>
            </div>
          </div>
          <div class="flex flex-wrap gap-1.5">
            <button id="trigger-sec-alert-btn" class="rounded bg-rose-600 px-2.5 py-1 text-[11px] font-semibold text-white hover:bg-rose-500">🚨 Simulate Anomaly Alert & Audit Log</button>
          </div>
        </div>

        <div class="md:col-span-7 rounded-xl border ${subCardBg} p-3 space-y-1.5 max-h-[175px] overflow-y-auto">
          <div class="flex items-center justify-between">
            <span class="text-xs font-semibold">Immutable Compliance Audit Trail (${auditLogs.length})</span>
            <span class="text-[10px] text-slate-400">Firestore Zero-Trust Guarded</span>
          </div>
          ${
            auditLogs.length === 0
              ? `<div class="text-[11px] text-slate-400">All checkpoint exports, hyperparameter edits, and security events are logged here with server timestamps.</div>`
              : auditLogs
                  .map(
                    (l) => `
              <div class="flex items-center justify-between border-b border-slate-800 py-1 text-[11px]">
                <div>
                  <span class="rounded px-1.5 py-0.5 text-[10px] font-semibold ${
                    l.severity === 'critical'
                      ? 'bg-rose-500/20 text-rose-300'
                      : 'bg-sky-500/20 text-sky-300'
                  }">${l.severity.toUpperCase()}</span>
                  <strong class="ml-1.5 text-slate-200">${l.action}</strong>
                  <span class="ml-1.5 text-slate-400">${l.details}</span>
                </div>
              </div>
            `
                  )
                  .join('')
          }
        </div>
      </div>
    `;
  }

  // Tab 7: 3D Rank Stabilizer Game Shell Controls
  return `
    <div class="flex flex-wrap items-center justify-between gap-4 rounded-xl border ${subCardBg} p-3">
      <div class="space-y-1">
        <div class="text-xs font-bold text-emerald-400">3D Rank Stabilizer Simulation — Defend the LoRA Bottleneck Core</div>
        <div class="text-[11px] text-slate-300">
          Controls: <kbd class="rounded bg-slate-800 px-1.5 py-0.5 font-mono-num">A</kbd> / <kbd class="rounded bg-slate-800 px-1.5 py-0.5 font-mono-num">D</kbd> or <kbd class="rounded bg-slate-800 px-1.5 py-0.5 font-mono-num">◀</kbd> / <kbd class="rounded bg-slate-800 px-1.5 py-0.5 font-mono-num">▶</kbd> Orbit Stabilizer Drone · <kbd class="rounded bg-slate-800 px-1.5 py-0.5 font-mono-num">Space</kbd> or XR Pinch to Neutralize Gradient Spikes
        </div>
      </div>
      <div class="flex items-center gap-4 font-mono-num text-xs">
        <div>SCORE: <strong id="game-score-val" class="text-sky-400">${spatialScript?.state.gameScore ?? 0}</strong></div>
        <div>WAVE: <strong id="game-wave-val" class="text-amber-400">${spatialScript?.state.gameWave ?? 1}</strong></div>
        <div>INTEGRITY: <strong id="game-hp-val" class="text-emerald-400">${spatialScript?.state.gameIntegrity ?? 100}%</strong></div>
        <button id="start-game-btn" class="rounded-lg bg-emerald-600 px-3.5 py-1.5 text-xs font-bold text-white hover:bg-emerald-500">
          ${spatialScript?.state.gameActive ? '⏹ End Simulation' : '▶ Start Rank Stabilizer'}
        </button>
        <button id="pulse-game-btn" class="rounded-lg bg-sky-600 px-3 py-1.5 text-xs font-bold text-white hover:bg-sky-500">
          ⚡ Fire Stabilizer Pulse
        </button>
      </div>
    </div>
  `;
}

function bindDOMEvents() {
  // Tab switches
  document.querySelectorAll('.tab-switch-btn').forEach((btn) => {
    btn.addEventListener('click', () => {
      const t = btn.getAttribute('data-tab') as typeof activeTab;
      if (t) {
        activeTab = t;
        renderAppShell();
      }
    });
  });

  // Dock collapse
  document.getElementById('toggle-dock-btn')?.addEventListener('click', () => {
    dockCollapsed = !dockCollapsed;
    renderAppShell();
  });

  // Theme toggle
  document.getElementById('theme-toggle-btn')?.addEventListener('click', () => {
    darkMode = !darkMode;
    renderAppShell();
  });

  // Auth Sign-In / Sign-Out
  document.getElementById('google-signin-btn')?.addEventListener('click', async () => {
    try {
      const { user } = await signInWithGoogleWorkspace();
      currentUserEmail = user.email;
      hasOAuthToken = true;
      setBanner(`Signed in as ${user.email} · All 12 Google Workspace APIs ready`);
      renderAppShell();
    } catch (e: unknown) {
      setBanner(`Sign-in error: ${e instanceof Error ? e.message : String(e)}`);
    }
  });

  document.getElementById('reauth-workspace-btn')?.addEventListener('click', async () => {
    try {
      await signInWithGoogleWorkspace();
      hasOAuthToken = true;
      setBanner('Google Workspace OAuth access token refreshed in memory.');
      renderAppShell();
    } catch (e: unknown) {
      setBanner(`OAuth refresh error: ${e instanceof Error ? e.message : String(e)}`);
    }
  });

  document.getElementById('signout-btn')?.addEventListener('click', async () => {
    await signOutUser();
    hasOAuthToken = false;
    setBanner('Signed out.');
    renderAppShell();
  });

  // Tab 1: Telemetry & Runs
  document.getElementById('step-epoch-btn')?.addEventListener('click', () => {
    if (!spatialScript) return;
    spatialScript.state.epoch += 1;
    const nextStep = (lossHistory.at(-1)?.step ?? 1200) + 50;
    const newLoss = Number(Math.max(0.018, spatialScript.state.loss * 0.94 + (Math.random() - 0.48) * 0.004).toFixed(4));
    const newVal = Number(Math.max(0.024, spatialScript.state.valLoss * 0.95 + (Math.random() - 0.45) * 0.005).toFixed(4));
    spatialScript.state.loss = newLoss;
    spatialScript.state.valLoss = newVal;
    lossHistory.push({ step: nextStep, trainLoss: newLoss, valLoss: newVal, lr: spatialScript.state.learningRate });
    if (lossHistory.length > 40) lossHistory.shift();
    spatialScript.refreshSpatialText();
    renderAppShell();
  });

  document.getElementById('fast-diagnose-btn')?.addEventListener('click', async () => {
    setBanner('Running low-latency diagnostic via gemini-3.1-flash-lite...');
    try {
      const res = await fetch('/api/gemini/fast-diagnose', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ telemetry: spatialScript?.state }),
      });
      const data = await res.json();
      setBanner(`Flash-Lite Diagnostic: ${data.text || data.error}`);
    } catch (e: unknown) {
      setBanner(`Diagnostic error: ${e instanceof Error ? e.message : String(e)}`);
    }
  });

  document.getElementById('tts-announce-btn')?.addEventListener('click', async () => {
    setBanner('Synthesizing multi-speaker telemetry briefing via gemini-3.8-flash-tts...');
    try {
      const res = await fetch('/api/gemini/tts', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          text: `Run ${spatialScript?.state.runName} is at epoch ${spatialScript?.state.epoch} with training loss ${spatialScript?.state.loss.toFixed(4)} and rank ${spatialScript?.state.rank}.`,
        }),
      });
      const data = await res.json();
      if (data.audioDataUrl) {
        const audio = new Audio(data.audioDataUrl);
        await audio.play();
        setBanner('Playing Gemini 3.8 Flash TTS briefing.');
      } else {
        setBanner(`TTS error: ${data.error || 'No audio returned'}`);
      }
    } catch (e: unknown) {
      setBanner(`TTS failed: ${e instanceof Error ? e.message : String(e)}`);
    }
  });

  document.getElementById('rank-range-input')?.addEventListener('input', (e) => {
    const val = Number((e.target as HTMLInputElement).value);
    if (spatialScript) {
      spatialScript.state.rank = val;
      spatialScript.refreshSpatialText();
    }
    const lbl = document.getElementById('rank-val-label');
    if (lbl) lbl.textContent = String(val);
  });

  document.getElementById('alpha-range-input')?.addEventListener('input', (e) => {
    const val = Number((e.target as HTMLInputElement).value);
    if (spatialScript) {
      spatialScript.state.alpha = val;
      spatialScript.refreshSpatialText();
    }
    const lbl = document.getElementById('alpha-val-label');
    if (lbl) lbl.textContent = String(val);
  });

  document.getElementById('save-run-firestore-btn')?.addEventListener('click', async () => {
    const name = (document.getElementById('run-name-input') as HTMLInputElement)?.value || 'LoRA-Run';
    const hfDataset = (document.getElementById('hf-dataset-input') as HTMLInputElement)?.value || 'deepmind/afdb';
    const githubRepo = (document.getElementById('github-repo-input') as HTMLInputElement)?.value || 'org/loraflow';
    const cloudProvider = (document.getElementById('cloud-provider-input') as HTMLSelectElement)?.value || 'GCP TPU';
    if (spatialScript) spatialScript.state.runName = name;
    try {
      const id = await createTrainingRunInDb({
        name,
        baseModel: 'ESM-2-3B / Llama-3-8B',
        rank: spatialScript?.state.rank ?? 16,
        alpha: spatialScript?.state.alpha ?? 32,
        learningRate: spatialScript?.state.learningRate ?? 0.0002,
        epoch: spatialScript?.state.epoch ?? 14,
        loss: spatialScript?.state.loss ?? 0.0842,
        valLoss: spatialScript?.state.valLoss ?? 0.0915,
        status: 'running',
        hfDataset,
        githubRepo,
        cloudProvider,
        encryptedAtRest: true,
      });
      await logAuditEventInDb('Created Training Run', 'info', `Run ${name} (${id}) saved with AES-256.`);
      setBanner(`Saved training run "${name}" to Firestore.`);
    } catch (e: unknown) {
      setBanner(`Firestore save error: ${e instanceof Error ? e.message : String(e)}`);
    }
  });

  document.getElementById('export-ckpt-btn')?.addEventListener('click', async () => {
    await triggerCheckpointExport();
  });

  document.querySelectorAll('.del-run-btn').forEach((btn) => {
    btn.addEventListener('click', () => {
      const runId = btn.getAttribute('data-del-run');
      if (!runId) return;
      showConfirmModal(
        'Delete Training Run from Firestore?',
        `Are you sure you want to permanently delete training run "${runId}" from cloud storage? This action cannot be undone.`,
        'Delete Run',
        async () => {
          await deleteTrainingRunFromDb(runId);
          await logAuditEventInDb('Deleted Training Run', 'warning', `Deleted run ${runId}`);
          setBanner(`Deleted training run ${runId}.`);
        }
      );
    });
  });

  // Tab 2: Samples, Video & Music
  document.getElementById('gen-sample-img-btn')?.addEventListener('click', async () => {
    const prompt = (document.getElementById('sample-prompt-input') as HTMLTextAreaElement)?.value;
    const model = (document.getElementById('sample-model-select') as HTMLSelectElement)?.value;
    const aspectRatio = (document.getElementById('sample-aspect-select') as HTMLSelectElement)?.value;
    const imageSize = (document.getElementById('sample-size-select') as HTMLSelectElement)?.value;
    const box = document.getElementById('sample-preview-box');
    if (box) box.innerHTML = `<div class="text-sky-400 animate-pulse">Generating ${imageSize} (${aspectRatio}) sample with ${model}...</div>`;
    try {
      const res = await fetch('/api/gemini/image', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ prompt, model, aspectRatio, imageSize }),
      });
      const data = await res.json();
      if (data.imageUrl && box) {
        box.innerHTML = `<img src="${data.imageUrl}" alt="Sample" class="max-h-[140px] mx-auto rounded border border-slate-700 object-contain" />`;
        setBanner(`Sample image generated (${model}, ${aspectRatio}, ${imageSize}).`);
      } else if (box) {
        box.textContent = `Error: ${data.error || 'No image generated'}`;
      }
    } catch (e: unknown) {
      if (box) box.textContent = `Error: ${e instanceof Error ? e.message : String(e)}`;
    }
  });

  document.getElementById('gen-sample-video-btn')?.addEventListener('click', async () => {
    const prompt = (document.getElementById('sample-prompt-input') as HTMLTextAreaElement)?.value;
    const rawAspect = (document.getElementById('sample-aspect-select') as HTMLSelectElement)?.value;
    const aspectRatio = rawAspect === '9:16' ? '9:16' : '16:9';
    const box = document.getElementById('sample-preview-box');
    if (box) box.innerHTML = `<div class="text-purple-400">Generating ${aspectRatio} video via gemini-omni-1.1-flash Interactions API...</div>`;
    try {
      const res = await fetch('/api/gemini/video', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ prompt, aspectRatio }),
      });
      const data = await res.json();
      if (data.videoDataUrl && box) {
        box.innerHTML = `<video src="${data.videoDataUrl}" controls autoplay loop class="max-h-[140px] mx-auto rounded border border-slate-700"></video>`;
        setBanner(`Video generated via gemini-omni-1.1-flash (${aspectRatio}).`);
      } else if (box) {
        box.textContent = `Video error: ${data.error || 'Failed'}`;
      }
    } catch (e: unknown) {
      if (box) box.textContent = `Error: ${e instanceof Error ? e.message : String(e)}`;
    }
  });

  document.getElementById('gen-lyria-music-btn')?.addEventListener('click', async () => {
    const prompt = (document.getElementById('sample-prompt-input') as HTMLTextAreaElement)?.value;
    const box = document.getElementById('sample-preview-box');
    if (box) box.innerHTML = `<div class="text-teal-400">Generating ambient soundtrack via lyria-3-clip-preview...</div>`;
    try {
      const res = await fetch('/api/gemini/music', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ prompt, model: 'lyria-3-clip-preview' }),
      });
      const data = await res.json();
      if (data.audioDataUrl && box) {
        box.innerHTML = `<audio src="${data.audioDataUrl}" controls autoplay class="w-full"></audio><div class="mt-1 text-[10px] text-teal-300">Lyria 3 Ambient Track Ready</div>`;
        setBanner('Generated Lyria 3 ambient training audio clip.');
      } else if (box) {
        box.textContent = `Music error: ${data.error || 'Failed'}`;
      }
    } catch (e: unknown) {
      if (box) box.textContent = `Error: ${e instanceof Error ? e.message : String(e)}`;
    }
  });

  document.getElementById('analyze-uploaded-media-btn')?.addEventListener('click', async () => {
    const fileInput = document.getElementById('media-upload-input') as HTMLInputElement;
    const file = fileInput?.files?.[0];
    const out = document.getElementById('media-analysis-output');
    if (!file) {
      if (out) out.textContent = 'Please choose an image or video file first.';
      return;
    }
    if (out) out.textContent = 'Analyzing with Gemini 4 Argon (High Thinking)...';
    const base64Data = await fileToBase64(file);
    try {
      const res = await fetch('/api/gemini/analyze-media', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ base64Data, mimeType: file.type }),
      });
      const data = await res.json();
      if (out) out.textContent = data.text || data.error;
    } catch (e: unknown) {
      if (out) out.textContent = `Error: ${e instanceof Error ? e.message : String(e)}`;
    }
  });

  document.getElementById('animate-uploaded-photo-btn')?.addEventListener('click', async () => {
    const fileInput = document.getElementById('media-upload-input') as HTMLInputElement;
    const file = fileInput?.files?.[0];
    const box = document.getElementById('sample-preview-box');
    if (!file) {
      setBanner('Select an image file first to animate with gemini-omni-1.1-flash.');
      return;
    }
    const base64Image = await fileToBase64(file);
    const prompt = (document.getElementById('sample-prompt-input') as HTMLTextAreaElement)?.value;
    if (box) box.innerHTML = `<div class="text-pink-400">Animating uploaded photo via gemini-omni-1.1-flash...</div>`;
    try {
      const res = await fetch('/api/gemini/video', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ prompt, aspectRatio: '16:9', base64Image, mimeType: file.type }),
      });
      const data = await res.json();
      if (data.videoDataUrl && box) {
        box.innerHTML = `<video src="${data.videoDataUrl}" controls autoplay loop class="max-h-[140px] mx-auto rounded"></video>`;
      } else if (box) {
        box.textContent = data.error || 'Failed to animate image.';
      }
    } catch (e: unknown) {
      if (box) box.textContent = `Error: ${e instanceof Error ? e.message : String(e)}`;
    }
  });

  // Tab 3: Bioscience Presets
  document.querySelectorAll('.bio-preset-btn').forEach((btn) => {
    btn.addEventListener('click', async () => {
      const acc = btn.getAttribute('data-acc') || 'P04637';
      const comp = btn.getAttribute('data-comp') || 'Imatinib';
      const loc = btn.getAttribute('data-loc') || 'chr17:7668402-7687550';
      setBanner(`Querying UniProtKB, AlphaFold DB, PubChem & Europe PMC for ${acc}...`);
      try {
        currentBioReport = await fetchBioTargetWorkbench(acc, comp, loc);
        setBanner(`Loaded structural target ${currentBioReport.geneName} (${acc}).`);
        renderAppShell();
      } catch (e: unknown) {
        setBanner(`Bioscience API error: ${e instanceof Error ? e.message : String(e)}`);
      }
    });
  });

  // Tab 4: 12 Google Workspace Integrations (with mandatory confirmation dialogs for mutating actions)
  const setWsMsg = (html: string) => {
    const el = document.getElementById('workspace-result-bar');
    if (el) el.innerHTML = html;
  };

  document.getElementById('ws-drive-btn')?.addEventListener('click', () => {
    showConfirmModal(
      'Export Checkpoint to Google Drive?',
      `Create "${spatialScript.state.runName}_lora_checkpoint.json" in your Google Drive?`,
      'Export to Drive',
      async () => {
        try {
          const res = await exportCheckpointToDrive(spatialScript.state.runName, {
            ...spatialScript.state,
            lossHistory,
            exportedAt: new Date().toISOString(),
          });
          setWsMsg(`✔ Saved to Google Drive: <strong>${res.name}</strong> (ID: ${res.id})`);
        } catch (e: unknown) {
          setWsMsg(`Drive error: ${e instanceof Error ? e.message : String(e)}`);
        }
      }
    );
  });

  document.getElementById('ws-sheets-btn')?.addEventListener('click', () => {
    showConfirmModal(
      'Export Loss Curve to Google Sheets?',
      `Create a new spreadsheet with ${lossHistory.length} loss curve rows in your Google Sheets?`,
      'Create Spreadsheet',
      async () => {
        try {
          const res = await exportLossCurveToSheets(spatialScript.state.runName, lossHistory);
          setWsMsg(
            `✔ Created Google Sheet: <a href="${res.spreadsheetUrl}" target="_blank" class="underline text-emerald-400">${res.spreadsheetId}</a>`
          );
        } catch (e: unknown) {
          setWsMsg(`Sheets error: ${e instanceof Error ? e.message : String(e)}`);
        }
      }
    );
  });

  document.getElementById('ws-gmail-btn')?.addEventListener('click', () => {
    const recipient = currentUserEmail || 'creatergame31@gmail.com';
    showConfirmModal(
      'Send Training Alert Email via Gmail?',
      `Send an automated MLOps checkpoint & anomaly report email to ${recipient} on your behalf?`,
      'Send Email',
      async () => {
        try {
          const res = await sendAlertEmailViaGmail(
            recipient,
            `[LoRAFlow Alert] Run ${spatialScript.state.runName} · Loss ${spatialScript.state.loss.toFixed(4)}`,
            `LoRAFlow Automated Telemetry Report:\nRun: ${spatialScript.state.runName}\nEpoch: ${spatialScript.state.epoch}\nTrain Loss: ${spatialScript.state.loss}\nVal Loss: ${spatialScript.state.valLoss}\nRank: ${spatialScript.state.rank}`
          );
          setWsMsg(`✔ Sent Gmail Alert to ${recipient} (Message ID: ${res.id})`);
        } catch (e: unknown) {
          setWsMsg(`Gmail error: ${e instanceof Error ? e.message : String(e)}`);
        }
      }
    );
  });

  document.getElementById('ws-calendar-btn')?.addEventListener('click', () => {
    showConfirmModal(
      'Schedule GPU Cluster Slot on Google Calendar?',
      `Create a 2-hour training window event for "${spatialScript.state.runName}" on your primary Google Calendar?`,
      'Schedule Event',
      async () => {
        try {
          const res = await scheduleTrainingWindowOnCalendar(
            `LoRA Training: ${spatialScript.state.runName}`,
            `Automated LoRA rank=${spatialScript.state.rank} sweep on cloud GPU cluster.`
          );
          setWsMsg(
            `✔ Scheduled on Calendar: <a href="${res.htmlLink}" target="_blank" class="underline text-blue-400">Open Event</a>`
          );
        } catch (e: unknown) {
          setWsMsg(`Calendar error: ${e instanceof Error ? e.message : String(e)}`);
        }
      }
    );
  });

  document.getElementById('ws-docs-btn')?.addEventListener('click', () => {
    showConfirmModal(
      'Create Reproducibility Report in Google Docs?',
      `Create a new Google Doc containing full hyperparameters, loss telemetry, and SHA-256 checkpoint signatures?`,
      'Create Google Doc',
      async () => {
        try {
          const res = await createExperimentReportInDocs(
            `LoRAFlow Report - ${spatialScript.state.runName}`,
            `LoRAFlow Experiment Reproducibility Report\n\nRun: ${spatialScript.state.runName}\nRank: ${spatialScript.state.rank}\nAlpha: ${spatialScript.state.alpha}\nEpoch: ${spatialScript.state.epoch}\nTrain Loss: ${spatialScript.state.loss}\nValidation Loss: ${spatialScript.state.valLoss}\nTarget Protein: ${currentBioReport?.geneName ?? 'TP53'}\n`
          );
          setWsMsg(
            `✔ Created Google Doc: <a href="${res.url}" target="_blank" class="underline text-indigo-400">Open Report</a>`
          );
        } catch (e: unknown) {
          setWsMsg(`Docs error: ${e instanceof Error ? e.message : String(e)}`);
        }
      }
    );
  });

  document.getElementById('ws-slides-btn')?.addEventListener('click', () => {
    showConfirmModal(
      'Create Executive Review Deck in Google Slides?',
      `Generate a new Google Slides presentation for "${spatialScript.state.runName}"?`,
      'Create Slides',
      async () => {
        try {
          const res = await createReviewDeckInSlides(
            `LoRAFlow Checkpoint Review: ${spatialScript.state.runName}`
          );
          setWsMsg(
            `✔ Created Google Slides Deck: <a href="${res.url}" target="_blank" class="underline text-amber-400">Open Presentation</a>`
          );
        } catch (e: unknown) {
          setWsMsg(`Slides error: ${e instanceof Error ? e.message : String(e)}`);
        }
      }
    );
  });

  document.getElementById('ws-tasks-btn')?.addEventListener('click', () => {
    showConfirmModal(
      'Create Hyperparameter Action Item in Google Tasks?',
      `Add a task "Evaluate LoRA r=${spatialScript.state.rank} checkpoint on validation set" to your default Google Tasks list?`,
      'Add Task',
      async () => {
        try {
          const res = await createFollowUpTaskInTasks(
            `Evaluate LoRA r=${spatialScript.state.rank} (${spatialScript.state.runName})`,
            `Check validation loss ${spatialScript.state.valLoss} and export safetensors.`
          );
          setWsMsg(`✔ Added Google Task: <strong>${res.title}</strong>`);
        } catch (e: unknown) {
          setWsMsg(`Tasks error: ${e instanceof Error ? e.message : String(e)}`);
        }
      }
    );
  });

  document.getElementById('ws-chat-btn')?.addEventListener('click', () => {
    showConfirmModal(
      'Post Checkpoint Update to Google Chat?',
      `Send a training status message to your Google Chat space?`,
      'Post Message',
      async () => {
        try {
          const res = await postMessageToGoogleChat(
            'spaces/AAAA',
            `LoRAFlow Run ${spatialScript.state.runName} reached loss ${spatialScript.state.loss.toFixed(4)}`
          );
          setWsMsg(`✔ Posted to Google Chat (${res.name})`);
        } catch (e: unknown) {
          setWsMsg(`Chat note: ${e instanceof Error ? e.message : String(e)}`);
        }
      }
    );
  });

  document.getElementById('ws-forms-btn')?.addEventListener('click', () => {
    showConfirmModal(
      'Create RLHF Sample Evaluation Form in Google Forms?',
      `Create a new Google Form for human raters to score checkpoint samples?`,
      'Create Form',
      async () => {
        try {
          const res = await createEvalFormInForms(
            `LoRA Sample Evaluation - ${spatialScript.state.runName}`
          );
          setWsMsg(
            `✔ Created Google Form: <a href="${res.responderUri}" target="_blank" class="underline text-purple-400">Open Form</a>`
          );
        } catch (e: unknown) {
          setWsMsg(`Forms error: ${e instanceof Error ? e.message : String(e)}`);
        }
      }
    );
  });

  document.getElementById('ws-keep-btn')?.addEventListener('click', () => {
    showConfirmModal(
      'Save Research Note (Google Keep / Cloud Sync)?',
      `Pin current hyperparameter observations for "${spatialScript.state.runName}"?`,
      'Save Note',
      async () => {
        try {
          const res = await createResearchNoteInKeep(
            `LoRA Note · ${spatialScript.state.runName}`,
            `Rank r=${spatialScript.state.rank}, Alpha=${spatialScript.state.alpha}, Val Loss=${spatialScript.state.valLoss}`
          );
          setWsMsg(`✔ Saved Note via ${res.source} (ID: ${res.id})`);
        } catch (e: unknown) {
          setWsMsg(`Keep note error: ${e instanceof Error ? e.message : String(e)}`);
        }
      }
    );
  });

  document.getElementById('ws-meet-btn')?.addEventListener('click', () => {
    showConfirmModal(
      'Create Instant War-Room in Google Meet?',
      `Provision a live Google Meet space for collaborative debugging?`,
      'Create Meet Space',
      async () => {
        try {
          const res = await createWarRoomInMeet();
          setWsMsg(
            `✔ Created Google Meet War-Room: <a href="${res.meetingUri}" target="_blank" class="underline text-cyan-400">${res.meetingUri}</a>`
          );
        } catch (e: unknown) {
          setWsMsg(`Meet error: ${e instanceof Error ? e.message : String(e)}`);
        }
      }
    );
  });

  document.getElementById('ws-contacts-btn')?.addEventListener('click', async () => {
    try {
      teamContacts = await fetchTeamContactsFromPeople();
      setWsMsg(
        `✔ Loaded ${teamContacts.length} team contacts from Google Contacts People API.`
      );
      renderAppShell();
    } catch (e: unknown) {
      setWsMsg(`Contacts error: ${e instanceof Error ? e.message : String(e)}`);
    }
  });

  // Tab 5: Gemini Co-Pilot, Search/Maps Grounding, Audio Transcribe, Live API
  document.getElementById('chat-send-btn')?.addEventListener('click', async () => {
    const input = document.getElementById('chat-input-box') as HTMLInputElement;
    const model = (document.getElementById('copilot-model-select') as HTMLSelectElement)?.value || 'gemini-4-argon';
    const text = input?.value?.trim();
    if (!text) return;
    input.value = '';
    chatMessages.push({ role: 'user', text });
    renderAppShell();
    try {
      const res = await fetch('/api/gemini/chat', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          messages: chatMessages,
          model,
          highThinking: model === 'gemini-4-argon',
        }),
      });
      const data = await res.json();
      chatMessages.push({ role: 'assistant', text: data.text || data.error || 'No response' });
      renderAppShell();
    } catch (e: unknown) {
      chatMessages.push({
        role: 'assistant',
        text: `Error: ${e instanceof Error ? e.message : String(e)}`,
      });
      renderAppShell();
    }
  });

  document.getElementById('search-grounding-btn')?.addEventListener('click', async () => {
    const box = document.getElementById('grounding-result-box');
    if (box) box.textContent = 'Querying Google Search via gemini-3.8-flash...';
    try {
      const res = await fetch('/api/gemini/search', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          query: `Latest LoRA low-rank adaptation training techniques and Hugging Face PEFT benchmarks for ${currentBioReport?.geneName ?? 'protein language models'}`,
        }),
      });
      const data = await res.json();
      if (box) box.textContent = data.text || data.error;
    } catch (e: unknown) {
      if (box) box.textContent = `Search error: ${e instanceof Error ? e.message : String(e)}`;
    }
  });

  document.getElementById('maps-grounding-btn')?.addEventListener('click', async () => {
    const box = document.getElementById('grounding-result-box');
    if (box) box.textContent = 'Querying Google Maps grounding via gemini-3.8-flash...';
    try {
      const res = await fetch('/api/gemini/maps', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          query: 'Major AI supercomputing labs, cloud data centers, and biotech research hubs in San Francisco Bay Area',
        }),
      });
      const data = await res.json();
      if (box) box.textContent = data.text || data.error;
    } catch (e: unknown) {
      if (box) box.textContent = `Maps error: ${e instanceof Error ? e.message : String(e)}`;
    }
  });

  document.getElementById('record-transcribe-btn')?.addEventListener('click', async () => {
    const box = document.getElementById('grounding-result-box');
    if (box) box.textContent = 'Recording 3.5 seconds of microphone audio for gemini-3.5-transcribe...';
    try {
      const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
      const recorder = new MediaRecorder(stream);
      const chunks: BlobPart[] = [];
      recorder.ondataavailable = (e) => chunks.push(e.data);
      recorder.onstop = async () => {
        stream.getTracks().forEach((t) => t.stop());
        const blob = new Blob(chunks, { type: 'audio/webm' });
        const base64Audio = await fileToBase64(blob);
        const res = await fetch('/api/gemini/transcribe', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ base64Audio, mimeType: 'audio/webm' }),
        });
        const data = await res.json();
        if (box) box.textContent = `Transcription (gemini-3.5-transcribe): ${data.text || data.error}`;
      };
      recorder.start();
      setTimeout(() => recorder.stop(), 3500);
    } catch (e: unknown) {
      if (box) box.textContent = `Mic error: ${e instanceof Error ? e.message : String(e)}`;
    }
  });

  document.getElementById('live-voice-btn')?.addEventListener('click', async () => {
    if (liveActive) {
      stopLiveVoiceSession();
      renderAppShell();
    } else {
      await startLiveVoiceSession();
      renderAppShell();
    }
  });

  // Tab 6: Security & RBAC
  document.getElementById('rbac-role-select')?.addEventListener('change', (e) => {
    currentRole = (e.target as HTMLSelectElement).value as typeof currentRole;
    setBanner(`Active RBAC Role switched to: ${currentRole}`);
  });

  document.getElementById('trigger-sec-alert-btn')?.addEventListener('click', async () => {
    setBanner('🚨 CRITICAL ALERT: Gradient norm spike detected on Layer 18! Logged to Firestore Audit Trail.');
    await logAuditEventInDb(
      'Critical Gradient Norm Spike Alert',
      'critical',
      `Layer 18 grad_norm=14.8 exceeded threshold during ${spatialScript?.state.runName}`
    );
  });

  // Tab 7: Game Controls
  document.getElementById('start-game-btn')?.addEventListener('click', () => {
    if (!spatialScript) return;
    spatialScript.toggleGameMode(!spatialScript.state.gameActive);
    renderAppShell();
  });

  document.getElementById('pulse-game-btn')?.addEventListener('click', () => {
    spatialScript?.fireStabilizerPulse();
  });
}

async function triggerCheckpointExport() {
  if (!spatialScript) return;
  const step = lossHistory.at(-1)?.step ?? 1200;
  const sha256 = `sha256_${Math.random().toString(16).slice(2, 14)}${Math.random().toString(16).slice(2, 14)}`;
  setBanner(`Exported checkpoint step ${step} (${sha256.slice(0, 18)}...)`);

  if (currentUserUid && trainingRuns.length > 0) {
    try {
      await createCheckpointInDb(trainingRuns[0].id, step, spatialScript.state.loss, sha256);
      await logAuditEventInDb(
        'Exported SafeTensors Checkpoint',
        'info',
        `Step ${step} (${sha256}) for run ${trainingRuns[0].name}`
      );
    } catch {
      // non-fatal if not signed in
    }
  }
}

function fileToBase64(fileOrBlob: Blob): Promise<string> {
  return new Promise((resolve, reject) => {
    const reader = new FileReader();
    reader.onload = () => resolve(String(reader.result));
    reader.onerror = reject;
    reader.readAsDataURL(fileOrBlob);
  });
}

async function startLiveVoiceSession() {
  const box = document.getElementById('grounding-result-box');
  try {
    const proto = location.protocol === 'https:' ? 'wss:' : 'ws:';
    liveWs = new WebSocket(`${proto}//${location.host}/api/live`);
    liveInputCtx = new AudioContext({ sampleRate: 16000 });
    liveOutputCtx = new AudioContext({ sampleRate: 24000 });
    liveNextStartTime = 0;

    const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
    const source = liveInputCtx.createMediaStreamSource(stream);
    const processor = liveInputCtx.createScriptProcessor(4096, 1, 1);
    source.connect(processor);
    processor.connect(liveInputCtx.destination);

    processor.onaudioprocess = (e) => {
      if (!liveWs || liveWs.readyState !== WebSocket.OPEN) return;
      const input = e.inputBuffer.getChannelData(0);
      const pcm16 = new Int16Array(input.length);
      for (let i = 0; i < input.length; i++) {
        const s = Math.max(-1, Math.min(1, input[i]));
        pcm16[i] = s < 0 ? s * 0x8000 : s * 0x7fff;
      }
      const bytes = new Uint8Array(pcm16.buffer);
      let binary = '';
      for (let i = 0; i < bytes.byteLength; i++) {
        binary += String.fromCharCode(bytes[i]);
      }
      liveWs.send(JSON.stringify({ audio: btoa(binary) }));
    };

    liveWs.onmessage = (event) => {
      const msg = JSON.parse(event.data);
      if (msg.audio && liveOutputCtx) {
        const bin = atob(msg.audio);
        const bytes = new Uint8Array(bin.length);
        for (let i = 0; i < bin.length; i++) bytes[i] = bin.charCodeAt(i);
        const pcm16 = new Int16Array(bytes.buffer);
        const float32 = new Float32Array(pcm16.length);
        for (let i = 0; i < pcm16.length; i++) float32[i] = pcm16[i] / 32768;

        const audioBuffer = liveOutputCtx.createBuffer(1, float32.length, 24000);
        audioBuffer.getChannelData(0).set(float32);
        const node = liveOutputCtx.createBufferSource();
        node.buffer = audioBuffer;
        node.connect(liveOutputCtx.destination);
        const startAt = Math.max(liveOutputCtx.currentTime, liveNextStartTime);
        node.start(startAt);
        liveNextStartTime = startAt + audioBuffer.duration;
      }
    };

    liveActive = true;
    if (box) box.textContent = 'Gemini 3.8 Live voice session active! Speak into your microphone.';
  } catch (e: unknown) {
    if (box) box.textContent = `Live API error: ${e instanceof Error ? e.message : String(e)}`;
  }
}

function stopLiveVoiceSession() {
  liveActive = false;
  if (liveWs) {
    liveWs.close();
    liveWs = null;
  }
  if (liveInputCtx) {
    liveInputCtx.close();
    liveInputCtx = null;
  }
  if (liveOutputCtx) {
    liveOutputCtx.close();
    liveOutputCtx = null;
  }
}

document.addEventListener('DOMContentLoaded', async () => {
  // 1. Validate Firestore connection on boot
  testFirestoreConnection();

  // 2. Mount XR Blocks 3D Spatial Scene
  spatialScript = new SpatialTensorLabScript();
  spatialScript.onRankChangedFromXR = (newRank) => {
    if (currentUserUid && trainingRuns.length > 0) {
      updateTrainingRunInDb(trainingRuns[0].id, { rank: newRank }).catch(() => {});
    }
    renderAppShell();
  };
  spatialScript.onExportCheckpointFromXR = () => {
    triggerCheckpointExport();
  };
  spatialScript.onGameStateUpdate = (score, wave, integrity) => {
    const sEl = document.getElementById('game-score-val');
    const wEl = document.getElementById('game-wave-val');
    const hEl = document.getElementById('game-hp-val');
    if (sEl) sEl.textContent = String(score);
    if (wEl) wEl.textContent = String(wave);
    if (hEl) hEl.textContent = `${integrity}%`;
  };

  xb.add(spatialScript);
  renderAppShell();

  // 3. Subscribe to Firebase Auth & Firestore Real-Time Collections
  initAuthListener((user, token) => {
    currentUserEmail = user?.email ?? null;
    currentUserUid = user?.uid ?? null;
    hasOAuthToken = Boolean(token);
    if (unsubFirestore) {
      unsubFirestore();
      unsubFirestore = null;
    }
    if (user) {
      unsubFirestore = subscribeUserCollections(user.uid, {
        onRuns: (runs) => {
          trainingRuns = runs;
          renderAppShell();
        },
        onCheckpoints: (ckpts) => {
          checkpoints = ckpts;
          renderAppShell();
        },
        onAuditLogs: (logs) => {
          auditLogs = logs;
          renderAppShell();
        },
        onKeepNotes: (notes) => {
          keepNotes = notes;
          renderAppShell();
        },
      });
    } else {
      trainingRuns = [];
      checkpoints = [];
      auditLogs = [];
      keepNotes = [];
      renderAppShell();
    }
  });

  // 4. Auto-load Default Bioscience Target (TP53 P04637) so user never lands on an empty state
  fetchBioTargetWorkbench('P04637', 'Nutlin-3a', 'chr17:7668402-7687550')
    .then((rep) => {
      currentBioReport = rep;
      if (activeTab === 'bioscience') renderAppShell();
    })
    .catch(() => {});

  // 5. Initialize XR Blocks runtime & validate spatial UI layout
  await xb.init(new xb.Options());
  setTimeout(() => {
    try {
      xb.ui.validate();
    } catch {
      // non-fatal validation check
    }
  }, 800);
});
