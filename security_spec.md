# Security Specification — LoRAFlow Spatial MLOps

## 1. Data Invariants
1. Every document in `training_runs`, `checkpoints`, `audit_logs`, and `keep_notes` MUST be owned by the authenticated user (`ownerId == request.auth.uid`) with `email_verified == true`.
2. `ownerId` and `createdAt` are immortal (immutable) after creation; `createdAt` and `updatedAt` must equal `request.time`.
3. Terminal state locking: Once a `TrainingRun` reaches `status == 'completed'` or `status == 'failed'`, no non-admin updates are permitted.
4. `audit_logs` and `keep_notes` are append-only/owner-scoped with strict key and length boundaries.

## 2. The "Dirty Dozen" Payloads
1. **Identity Spoofing**: Creating a `training_runs` document where `ownerId` is another user's UID.
2. **Unverified Email Write**: Creating a `training_runs` document when `request.auth.token.email_verified == false`.
3. **Shadow Field Injection**: Updating a `training_runs` document with an undeclared key `isAdmin: true`.
4. **Timestamp Forgery**: Creating a `checkpoints` document with a past or future client timestamp (`createdAt != request.time`).
5. **Terminal State Bypass**: Updating `loss` on a `training_runs` document whose `existing().status == 'completed'`.
6. **Owner Mutation**: Updating `ownerId` on an existing `training_runs` document.
7. **Oversized String Denial-of-Wallet**: Creating an `audit_logs` document with `details` > 500 chars.
8. **Invalid Enum State**: Creating a `training_runs` document with `status: 'hacked'`.
9. **Blanket List Scraping**: Listing `training_runs` without filtering `resource.data.ownerId == request.auth.uid`.
10. **Orphaned Checkpoint Creation**: Creating a `checkpoints` document referencing a non-existent `runId`.
11. **Audit Log Tampering**: Attempting to `update` or `delete` an immutable `audit_logs` entry.
12. **Poisoned Document ID**: Creating a document with a 500-char or special-char document ID failing `isValidId()`.
