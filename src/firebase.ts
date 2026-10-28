import { initializeApp } from 'firebase/app';
import {
  getAuth,
  GoogleAuthProvider,
  signInWithPopup,
  onAuthStateChanged,
  User,
} from 'firebase/auth';
import {
  getFirestore,
  doc,
  getDocFromServer,
  collection,
  setDoc,
  updateDoc,
  deleteDoc,
  query,
  where,
  onSnapshot,
  serverTimestamp,
  Timestamp,
} from 'firebase/firestore';
import firebaseConfig from '../firebase-applet-config.json';

const app = initializeApp(firebaseConfig);
const databaseId = (firebaseConfig as { firestoreDatabaseId?: string }).firestoreDatabaseId;
export const db = databaseId ? getFirestore(app, databaseId) : getFirestore(app);
export const auth = getAuth(app);

export const WORKSPACE_SCOPES = [
  'https://www.googleapis.com/auth/drive.file',
  'https://www.googleapis.com/auth/spreadsheets',
  'https://www.googleapis.com/auth/gmail.send',
  'https://www.googleapis.com/auth/calendar.events',
  'https://www.googleapis.com/auth/documents',
  'https://www.googleapis.com/auth/presentations',
  'https://www.googleapis.com/auth/tasks',
  'https://www.googleapis.com/auth/chat.messages.create',
  'https://www.googleapis.com/auth/forms.body',
  'https://www.googleapis.com/auth/meetings.space.created',
  'https://www.googleapis.com/auth/contacts.readonly',
];

const provider = new GoogleAuthProvider();
for (const scope of WORKSPACE_SCOPES) {
  provider.addScope(scope);
}

let isSigningIn = false;
let cachedAccessToken: string | null = null;

export enum OperationType {
  CREATE = 'create',
  UPDATE = 'update',
  DELETE = 'delete',
  LIST = 'list',
  GET = 'get',
  WRITE = 'write',
}

export interface FirestoreErrorInfo {
  error: string;
  operationType: OperationType;
  path: string | null;
  authInfo: {
    userId?: string | null;
    email?: string | null;
    emailVerified?: boolean | null;
    isAnonymous?: boolean | null;
    tenantId?: string | null;
    providerInfo?: {
      providerId?: string | null;
      email?: string | null;
    }[];
  };
}

export function handleFirestoreError(
  error: unknown,
  operationType: OperationType,
  path: string | null
): never {
  const errInfo: FirestoreErrorInfo = {
    error: error instanceof Error ? error.message : String(error),
    authInfo: {
      userId: auth.currentUser?.uid,
      email: auth.currentUser?.email,
      emailVerified: auth.currentUser?.emailVerified,
      isAnonymous: auth.currentUser?.isAnonymous,
      tenantId: auth.currentUser?.tenantId,
      providerInfo:
        auth.currentUser?.providerData.map((p) => ({
          providerId: p.providerId,
          email: p.email,
        })) || [],
    },
    operationType,
    path,
  };
  console.error('Firestore Error: ', JSON.stringify(errInfo));
  throw new Error(JSON.stringify(errInfo));
}

export async function testFirestoreConnection(): Promise<void> {
  try {
    await getDocFromServer(doc(db, 'test', 'connection'));
  } catch (error) {
    if (error instanceof Error && error.message.includes('the client is offline')) {
      console.error('Please check your Firebase configuration.');
    }
  }
}

export function initAuthListener(
  onUserChange: (user: User | null, token: string | null) => void
) {
  return onAuthStateChanged(auth, (user) => {
    if (!user) {
      cachedAccessToken = null;
      onUserChange(null, null);
    } else {
      onUserChange(user, cachedAccessToken);
    }
  });
}

export async function signInWithGoogleWorkspace(): Promise<{
  user: User;
  accessToken: string;
}> {
  try {
    isSigningIn = true;
    const result = await signInWithPopup(auth, provider);
    const credential = GoogleAuthProvider.credentialFromResult(result);
    if (!credential?.accessToken) {
      throw new Error('Failed to obtain OAuth access token from Google sign-in.');
    }
    cachedAccessToken = credential.accessToken;
    return { user: result.user, accessToken: cachedAccessToken };
  } finally {
    isSigningIn = false;
  }
}

export function getAccessToken(): string | null {
  return cachedAccessToken;
}

export async function signOutUser(): Promise<void> {
  await auth.signOut();
  cachedAccessToken = null;
}

export interface TrainingRunDoc {
  id: string;
  ownerId: string;
  name: string;
  baseModel: string;
  rank: number;
  alpha: number;
  learningRate: number;
  epoch: number;
  loss: number;
  valLoss: number;
  status: 'running' | 'paused' | 'completed' | 'failed';
  hfDataset: string;
  githubRepo: string;
  cloudProvider: string;
  encryptedAtRest: boolean;
  createdAt?: Timestamp;
  updatedAt?: Timestamp;
}

export interface CheckpointDoc {
  id: string;
  ownerId: string;
  runId: string;
  step: number;
  loss: number;
  sha256: string;
  createdAt?: Timestamp;
}

export interface AuditLogDoc {
  id: string;
  ownerId: string;
  action: string;
  severity: 'info' | 'warning' | 'critical';
  details: string;
  createdAt?: Timestamp;
}

export interface KeepNoteDoc {
  id: string;
  ownerId: string;
  title: string;
  body: string;
  createdAt?: Timestamp;
}

function sanitizeId(raw: string): string {
  return raw.replace(/[^a-zA-Z0-9_-]/g, '').slice(0, 120) || `id_${Date.now()}`;
}

export async function createTrainingRunInDb(
  input: Omit<TrainingRunDoc, 'id' | 'ownerId' | 'createdAt' | 'updatedAt'>
): Promise<string> {
  const user = auth.currentUser;
  if (!user) throw new Error('Must be signed in to create a training run.');
  const runId = sanitizeId(`run_${Date.now()}_${Math.random().toString(36).slice(2, 7)}`);
  const path = `training_runs/${runId}`;
  try {
    await setDoc(doc(db, 'training_runs', runId), {
      ownerId: user.uid,
      name: input.name.slice(0, 120),
      baseModel: input.baseModel.slice(0, 120),
      rank: Math.min(1024, Math.max(1, Number(input.rank))),
      alpha: Math.min(2048, Math.max(1, Number(input.alpha))),
      learningRate: Math.min(1, Math.max(0.000001, Number(input.learningRate))),
      epoch: Math.min(10000, Math.max(0, Number(input.epoch))),
      loss: Math.min(1000, Math.max(0, Number(input.loss))),
      valLoss: Math.min(1000, Math.max(0, Number(input.valLoss))),
      status: input.status,
      hfDataset: input.hfDataset.slice(0, 200),
      githubRepo: input.githubRepo.slice(0, 200),
      cloudProvider: input.cloudProvider.slice(0, 80),
      encryptedAtRest: Boolean(input.encryptedAtRest),
      createdAt: serverTimestamp(),
      updatedAt: serverTimestamp(),
    });
    return runId;
  } catch (err) {
    handleFirestoreError(err, OperationType.CREATE, path);
  }
}

export async function updateTrainingRunInDb(
  runId: string,
  updates: Partial<
    Pick<
      TrainingRunDoc,
      'epoch' | 'loss' | 'valLoss' | 'status' | 'rank' | 'alpha' | 'learningRate'
    >
  >
): Promise<void> {
  const cleanId = sanitizeId(runId);
  const path = `training_runs/${cleanId}`;
  try {
    await updateDoc(doc(db, 'training_runs', cleanId), {
      ...updates,
      updatedAt: serverTimestamp(),
    });
  } catch (err) {
    handleFirestoreError(err, OperationType.UPDATE, path);
  }
}

export async function deleteTrainingRunFromDb(runId: string): Promise<void> {
  const cleanId = sanitizeId(runId);
  const path = `training_runs/${cleanId}`;
  try {
    await deleteDoc(doc(db, 'training_runs', cleanId));
  } catch (err) {
    handleFirestoreError(err, OperationType.DELETE, path);
  }
}

export async function createCheckpointInDb(
  runId: string,
  step: number,
  loss: number,
  sha256: string
): Promise<string> {
  const user = auth.currentUser;
  if (!user) throw new Error('Must be signed in to export a checkpoint.');
  const ckptId = sanitizeId(`ckpt_${Date.now()}_${Math.random().toString(36).slice(2, 6)}`);
  const path = `checkpoints/${ckptId}`;
  try {
    await setDoc(doc(db, 'checkpoints', ckptId), {
      ownerId: user.uid,
      runId: sanitizeId(runId),
      step: Math.max(0, Math.round(step)),
      loss: Math.max(0, Number(loss)),
      sha256: sha256.slice(0, 128),
      createdAt: serverTimestamp(),
    });
    return ckptId;
  } catch (err) {
    handleFirestoreError(err, OperationType.CREATE, path);
  }
}

export async function logAuditEventInDb(
  action: string,
  severity: 'info' | 'warning' | 'critical',
  details: string
): Promise<void> {
  const user = auth.currentUser;
  if (!user) return;
  const logId = sanitizeId(`log_${Date.now()}_${Math.random().toString(36).slice(2, 6)}`);
  const path = `audit_logs/${logId}`;
  try {
    await setDoc(doc(db, 'audit_logs', logId), {
      ownerId: user.uid,
      action: action.slice(0, 160),
      severity,
      details: details.slice(0, 500),
      createdAt: serverTimestamp(),
    });
  } catch (err) {
    handleFirestoreError(err, OperationType.CREATE, path);
  }
}

export async function createKeepNoteInDb(title: string, body: string): Promise<string> {
  const user = auth.currentUser;
  if (!user) throw new Error('Must be signed in to save a note.');
  const noteId = sanitizeId(`note_${Date.now()}_${Math.random().toString(36).slice(2, 6)}`);
  const path = `keep_notes/${noteId}`;
  try {
    await setDoc(doc(db, 'keep_notes', noteId), {
      ownerId: user.uid,
      title: title.slice(0, 140),
      body: body.slice(0, 2000),
      createdAt: serverTimestamp(),
    });
    return noteId;
  } catch (err) {
    handleFirestoreError(err, OperationType.CREATE, path);
  }
}

export function subscribeUserCollections(
  uid: string,
  callbacks: {
    onRuns: (runs: TrainingRunDoc[]) => void;
    onCheckpoints: (ckpts: CheckpointDoc[]) => void;
    onAuditLogs: (logs: AuditLogDoc[]) => void;
    onKeepNotes: (notes: KeepNoteDoc[]) => void;
  }
): () => void {
  const runsQ = query(collection(db, 'training_runs'), where('ownerId', '==', uid));
  const ckptsQ = query(collection(db, 'checkpoints'), where('ownerId', '==', uid));
  const logsQ = query(collection(db, 'audit_logs'), where('ownerId', '==', uid));
  const notesQ = query(collection(db, 'keep_notes'), where('ownerId', '==', uid));

  const unsub1 = onSnapshot(
    runsQ,
    (snap) => {
      const items = snap.docs.map((d) => ({ id: d.id, ...(d.data() as Omit<TrainingRunDoc, 'id'>) }));
      callbacks.onRuns(items);
    },
    (err) => handleFirestoreError(err, OperationType.LIST, 'training_runs')
  );

  const unsub2 = onSnapshot(
    ckptsQ,
    (snap) => {
      const items = snap.docs.map((d) => ({ id: d.id, ...(d.data() as Omit<CheckpointDoc, 'id'>) }));
      callbacks.onCheckpoints(items);
    },
    (err) => handleFirestoreError(err, OperationType.LIST, 'checkpoints')
  );

  const unsub3 = onSnapshot(
    logsQ,
    (snap) => {
      const items = snap.docs.map((d) => ({ id: d.id, ...(d.data() as Omit<AuditLogDoc, 'id'>) }));
      callbacks.onAuditLogs(items);
    },
    (err) => handleFirestoreError(err, OperationType.LIST, 'audit_logs')
  );

  const unsub4 = onSnapshot(
    notesQ,
    (snap) => {
      const items = snap.docs.map((d) => ({ id: d.id, ...(d.data() as Omit<KeepNoteDoc, 'id'>) }));
      callbacks.onKeepNotes(items);
    },
    (err) => handleFirestoreError(err, OperationType.LIST, 'keep_notes')
  );

  return () => {
    unsub1();
    unsub2();
    unsub3();
    unsub4();
  };
}
