import { getAccessToken, createKeepNoteInDb } from './firebase';

function requireToken(): string {
  const token = getAccessToken();
  if (!token) {
    throw new Error('Please sign in with Google first to authorize Google Workspace APIs.');
  }
  return token;
}

// 1. Google Drive: Export LoRA Checkpoint & Config JSON
export async function exportCheckpointToDrive(
  runName: string,
  checkpointPayload: Record<string, unknown>
): Promise<{ id: string; name: string; webViewLink?: string }> {
  const token = requireToken();
  const metadata = {
    name: `${runName.replace(/\s+/g, '_')}_lora_checkpoint.json`,
    mimeType: 'application/json',
  };
  const boundary = '-------314159265358979323846';
  const delimiter = `\r\n--${boundary}\r\n`;
  const closeDelim = `\r\n--${boundary}--`;

  const multipartBody =
    delimiter +
    'Content-Type: application/json; charset=UTF-8\r\n\r\n' +
    JSON.stringify(metadata) +
    delimiter +
    'Content-Type: application/json\r\n\r\n' +
    JSON.stringify(checkpointPayload, null, 2) +
    closeDelim;

  const res = await fetch(
    'https://www.googleapis.com/upload/drive/v3/files?uploadType=multipart&fields=id,name,webViewLink',
    {
      method: 'POST',
      headers: {
        Authorization: `Bearer ${token}`,
        'Content-Type': `multipart/related; boundary=${boundary}`,
      },
      body: multipartBody,
    }
  );
  if (!res.ok) {
    const errText = await res.text();
    throw new Error(`Drive export failed (${res.status}): ${errText}`);
  }
  return await res.json();
}

// 2. Google Sheets: Export Loss Curve & Epoch Telemetry
export async function exportLossCurveToSheets(
  runName: string,
  rows: Array<{ step: number; trainLoss: number; valLoss: number; lr: number }>
): Promise<{ spreadsheetId: string; spreadsheetUrl: string }> {
  const token = requireToken();
  const createRes = await fetch('https://sheets.googleapis.com/v4/spreadsheets', {
    method: 'POST',
    headers: {
      Authorization: `Bearer ${token}`,
      'Content-Type': 'application/json',
    },
    body: JSON.stringify({
      properties: { title: `LoRAFlow Telemetry - ${runName}` },
    }),
  });
  if (!createRes.ok) {
    throw new Error(`Sheets create failed (${createRes.status}): ${await createRes.text()}`);
  }
  const sheetData = await createRes.json();
  const spreadsheetId = sheetData.spreadsheetId;

  const values = [
    ['Step', 'Train Loss', 'Validation Loss', 'Learning Rate'],
    ...rows.map((r) => [r.step, r.trainLoss, r.valLoss, r.lr]),
  ];

  await fetch(
    `https://sheets.googleapis.com/v4/spreadsheets/${spreadsheetId}/values/A1:append?valueInputOption=USER_ENTERED`,
    {
      method: 'POST',
      headers: {
        Authorization: `Bearer ${token}`,
        'Content-Type': 'application/json',
      },
      body: JSON.stringify({ values }),
    }
  );

  return {
    spreadsheetId,
    spreadsheetUrl:
      sheetData.spreadsheetUrl || `https://docs.google.com/spreadsheets/d/${spreadsheetId}`,
  };
}

// 3. Gmail: Send Critical System Alert or Training Summary Email
export async function sendAlertEmailViaGmail(
  toEmail: string,
  subject: string,
  bodyText: string
): Promise<{ id: string }> {
  const token = requireToken();
  const rawMessage = [
    `To: ${toEmail}`,
    `Subject: ${subject}`,
    'Content-Type: text/plain; charset="UTF-8"',
    '',
    bodyText,
  ].join('\r\n');

  const encoded = btoa(unescape(encodeURIComponent(rawMessage)))
    .replace(/\+/g, '-')
    .replace(/\//g, '_')
    .replace(/=+$/, '');

  const res = await fetch('https://gmail.googleapis.com/gmail/v1/users/me/messages/send', {
    method: 'POST',
    headers: {
      Authorization: `Bearer ${token}`,
      'Content-Type': 'application/json',
    },
    body: JSON.stringify({ raw: encoded }),
  });
  if (!res.ok) {
    throw new Error(`Gmail send failed (${res.status}): ${await res.text()}`);
  }
  return await res.json();
}

// 4. Google Calendar: Schedule GPU Cluster Training Window
export async function scheduleTrainingWindowOnCalendar(
  summary: string,
  description: string
): Promise<{ id: string; htmlLink: string }> {
  const token = requireToken();
  const start = new Date(Date.now() + 15 * 60 * 1000);
  const end = new Date(start.getTime() + 2 * 60 * 60 * 1000);

  const res = await fetch('https://www.googleapis.com/calendar/v3/calendars/primary/events', {
    method: 'POST',
    headers: {
      Authorization: `Bearer ${token}`,
      'Content-Type': 'application/json',
    },
    body: JSON.stringify({
      summary,
      description,
      start: { dateTime: start.toISOString() },
      end: { dateTime: end.toISOString() },
    }),
  });
  if (!res.ok) {
    throw new Error(`Calendar event failed (${res.status}): ${await res.text()}`);
  }
  return await res.json();
}

// 5. Google Docs: Generate Reproducible Experiment Report
export async function createExperimentReportInDocs(
  title: string,
  reportMarkdown: string
): Promise<{ documentId: string; url: string }> {
  const token = requireToken();
  const createRes = await fetch('https://docs.googleapis.com/v1/documents', {
    method: 'POST',
    headers: {
      Authorization: `Bearer ${token}`,
      'Content-Type': 'application/json',
    },
    body: JSON.stringify({ title }),
  });
  if (!createRes.ok) {
    throw new Error(`Docs create failed (${createRes.status}): ${await createRes.text()}`);
  }
  const docData = await createRes.json();
  const documentId = docData.documentId;

  await fetch(`https://docs.googleapis.com/v1/documents/${documentId}:batchUpdate`, {
    method: 'POST',
    headers: {
      Authorization: `Bearer ${token}`,
      'Content-Type': 'application/json',
    },
    body: JSON.stringify({
      requests: [
        {
          insertText: {
            location: { index: 1 },
            text: reportMarkdown,
          },
        },
      ],
    }),
  });

  return {
    documentId,
    url: `https://docs.google.com/document/d/${documentId}/edit`,
  };
}

// 6. Google Slides: Generate Executive Checkpoint Review Deck
export async function createReviewDeckInSlides(
  title: string
): Promise<{ presentationId: string; url: string }> {
  const token = requireToken();
  const res = await fetch('https://slides.googleapis.com/v1/presentations', {
    method: 'POST',
    headers: {
      Authorization: `Bearer ${token}`,
      'Content-Type': 'application/json',
    },
    body: JSON.stringify({ title }),
  });
  if (!res.ok) {
    throw new Error(`Slides create failed (${res.status}): ${await res.text()}`);
  }
  const data = await res.json();
  return {
    presentationId: data.presentationId,
    url: `https://docs.google.com/presentation/d/${data.presentationId}/edit`,
  };
}

// 7. Google Tasks: Create Hyperparameter Follow-Up Task
export async function createFollowUpTaskInTasks(
  title: string,
  notes: string
): Promise<{ id: string; title: string }> {
  const token = requireToken();
  const res = await fetch('https://tasks.googleapis.com/tasks/v1/lists/@default/tasks', {
    method: 'POST',
    headers: {
      Authorization: `Bearer ${token}`,
      'Content-Type': 'application/json',
    },
    body: JSON.stringify({ title, notes }),
  });
  if (!res.ok) {
    throw new Error(`Tasks create failed (${res.status}): ${await res.text()}`);
  }
  return await res.json();
}

// 8. Google Chat: Post Training Alert to Space
export async function postMessageToGoogleChat(
  spaceName: string,
  text: string
): Promise<{ name: string }> {
  const token = requireToken();
  const formattedSpace = spaceName.startsWith('spaces/') ? spaceName : `spaces/${spaceName}`;
  const res = await fetch(`https://chat.googleapis.com/v1/${formattedSpace}/messages`, {
    method: 'POST',
    headers: {
      Authorization: `Bearer ${token}`,
      'Content-Type': 'application/json',
    },
    body: JSON.stringify({ text }),
  });
  if (!res.ok) {
    throw new Error(`Google Chat post failed (${res.status}): ${await res.text()}`);
  }
  return await res.json();
}

// 9. Google Forms: Create Human Preference Evaluation Form
export async function createEvalFormInForms(
  title: string
): Promise<{ formId: string; responderUri: string }> {
  const token = requireToken();
  const res = await fetch('https://forms.googleapis.com/v1/forms', {
    method: 'POST',
    headers: {
      Authorization: `Bearer ${token}`,
      'Content-Type': 'application/json',
    },
    body: JSON.stringify({
      info: {
        title,
        documentTitle: title,
      },
    }),
  });
  if (!res.ok) {
    throw new Error(`Forms create failed (${res.status}): ${await res.text()}`);
  }
  const data = await res.json();
  return {
    formId: data.formId,
    responderUri: data.responderUri || `https://docs.google.com/forms/d/${data.formId}/edit`,
  };
}

// 10. Google Keep: Attempt Enterprise Keep API with automatic Firestore Research Note Sync
export async function createResearchNoteInKeep(
  title: string,
  bodyText: string
): Promise<{ source: string; id: string }> {
  const token = getAccessToken();
  if (token) {
    try {
      const res = await fetch('https://keep.googleapis.com/v1/notes', {
        method: 'POST',
        headers: {
          Authorization: `Bearer ${token}`,
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          title,
          body: { text: { text: bodyText } },
        }),
      });
      if (res.ok) {
        const data = await res.json();
        return { source: 'Google Keep Enterprise API', id: data.name || 'keep_note' };
      }
    } catch {
      // Fall through to Firestore-backed Keep note sync for consumer @gmail.com accounts
    }
  }
  const noteId = await createKeepNoteInDb(title, bodyText);
  return { source: 'Cloud Keep Sync (Firestore Encrypted)', id: noteId };
}

// 11. Google Meet: Create Instant War-Room Space
export async function createWarRoomInMeet(): Promise<{ meetingUri: string; meetingCode: string }> {
  const token = requireToken();
  const res = await fetch('https://meet.googleapis.com/v2/spaces', {
    method: 'POST',
    headers: {
      Authorization: `Bearer ${token}`,
      'Content-Type': 'application/json',
    },
    body: JSON.stringify({}),
  });
  if (!res.ok) {
    throw new Error(`Google Meet create failed (${res.status}): ${await res.text()}`);
  }
  const data = await res.json();
  return {
    meetingUri: data.meetingUri || `https://meet.google.com/${data.meetingCode || ''}`,
    meetingCode: data.meetingCode || data.name || 'meet-space',
  };
}

// 12. Google Contacts (People API): Fetch Team Collaborators for RBAC
export interface CollaboratorContact {
  name: string;
  email: string;
}

export async function fetchTeamContactsFromPeople(): Promise<CollaboratorContact[]> {
  const token = requireToken();
  const res = await fetch(
    'https://people.googleapis.com/v1/people/me/connections?personFields=names,emailAddresses&pageSize=12',
    {
      headers: {
        Authorization: `Bearer ${token}`,
      },
    }
  );
  if (!res.ok) {
    throw new Error(`People API failed (${res.status}): ${await res.text()}`);
  }
  const data = await res.json();
  const connections = data.connections || [];
  return connections.map((c: Record<string, Array<Record<string, string>>>) => ({
    name: c.names?.[0]?.displayName || 'Research Collaborator',
    email: c.emailAddresses?.[0]?.value || 'collaborator@workspace.org',
  }));
}
