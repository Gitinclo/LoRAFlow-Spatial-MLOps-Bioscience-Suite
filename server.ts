import express from 'express';
import http from 'http';
import path from 'path';
import { fileURLToPath } from 'url';
import { WebSocketServer, WebSocket } from 'ws';
import { GoogleGenAI, ThinkingLevel, Modality, LiveServerMessage } from '@google/genai';

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);

const app = express();
app.use(express.json({ limit: '50mb' }));

function getAi() {
  return new GoogleGenAI({
    apiKey: process.env.GEMINI_API_KEY,
    httpOptions: {
      headers: {
        'User-Agent': 'aistudio-build',
      },
    },
  });
}

// 1. Multi-Turn Chat & Intelligence (gemini-4-argon, gemini-3.8-flash, gemini-3.1-flash-lite)
app.post('/api/gemini/chat', async (req, res) => {
  try {
    const { messages, model = 'gemini-4-argon', highThinking = true, systemInstruction } = req.body;
    const ai = getAi();
    const config: Record<string, unknown> = {
      systemInstruction:
        systemInstruction ||
        'You are LoRAFlow Architect, a principal deep-learning & structural bioinformatics research assistant specializing in Low-Rank Adaptation (LoRA), rank/alpha hyperparameter tuning, loss curve diagnostics, and protein binder fine-tuning.',
    };
    if (model === 'gemini-4-argon' && highThinking) {
      config.thinkingConfig = { thinkingLevel: ThinkingLevel.HIGH };
    } else if (model === 'gemini-3.1-flash-lite') {
      config.thinkingConfig = { thinkingLevel: ThinkingLevel.MINIMAL };
    }
    const contents = Array.isArray(messages)
      ? messages.map((m: { role: string; text: string }) => ({
          role: m.role === 'assistant' ? 'model' : 'user',
          parts: [{ text: m.text }],
        }))
      : String(messages || 'Hello');

    const response = await ai.models.generateContent({
      model,
      contents,
      config,
    });
    res.json({ text: response.text || '', model });
  } catch (err: unknown) {
    res.status(500).json({ error: err instanceof Error ? err.message : String(err) });
  }
});

// 2. Low-Latency Diagnostics (gemini-3.1-flash-lite)
app.post('/api/gemini/fast-diagnose', async (req, res) => {
  try {
    const { telemetry } = req.body;
    const ai = getAi();
    const response = await ai.models.generateContent({
      model: 'gemini-3.1-flash-lite',
      contents: `Analyze this LoRA training telemetry in 2 concise bullet points (check gradient norm, rank/alpha ratio, and validation loss divergence): ${JSON.stringify(telemetry)}`,
      config: {
        thinkingConfig: { thinkingLevel: ThinkingLevel.MINIMAL },
      },
    });
    res.json({ text: response.text || '' });
  } catch (err: unknown) {
    res.status(500).json({ error: err instanceof Error ? err.message : String(err) });
  }
});

// 3. Search Grounding (gemini-3.8-flash with googleSearch)
app.post('/api/gemini/search', async (req, res) => {
  try {
    const { query } = req.body;
    const ai = getAi();
    const response = await ai.models.generateContent({
      model: 'gemini-3.8-flash',
      contents: query || 'Latest LoRA & QLoRA rank scheduling papers and Hugging Face PEFT releases',
      config: {
        tools: [{ googleSearch: {} }],
      },
    });
    const chunks = response.candidates?.[0]?.groundingMetadata?.groundingChunks || [];
    res.json({ text: response.text || '', groundingChunks: chunks });
  } catch (err: unknown) {
    res.status(500).json({ error: err instanceof Error ? err.message : String(err) });
  }
});

// 4. Maps Grounding (gemini-3.8-flash with googleMaps)
app.post('/api/gemini/maps', async (req, res) => {
  try {
    const { query } = req.body;
    const ai = getAi();
    const response = await ai.models.generateContent({
      model: 'gemini-3.8-flash',
      contents: query || 'Major AI GPU cloud data center regions and research hubs in California',
      config: {
        tools: [{ googleMaps: {} }],
      },
    });
    const chunks = response.candidates?.[0]?.groundingMetadata?.groundingChunks || [];
    res.json({ text: response.text || '', groundingChunks: chunks });
  } catch (err: unknown) {
    res.status(500).json({ error: err instanceof Error ? err.message : String(err) });
  }
});

// 5. Generate & Edit Sample Images (gemini-nano-banana-2.1 and gemini-3-pro-image-preview)
app.post('/api/gemini/image', async (req, res) => {
  try {
    const {
      prompt,
      model = 'gemini-nano-banana-2.1',
      aspectRatio = '1:1',
      imageSize = '1K',
      base64Image,
      mimeType = 'image/png',
    } = req.body;
    const ai = getAi();

    const parts: Array<Record<string, unknown>> = [];
    if (base64Image) {
      parts.push({
        inlineData: {
          data: base64Image.replace(/^data:image\/\w+;base64,/, ''),
          mimeType,
        },
      });
    }
    parts.push({ text: prompt || 'A futuristic LoRA neural adapter matrix crystallizing in a dark laboratory' });

    const allowedAspect = ['1:1', '3:4', '4:3', '9:16', '16:9'].includes(aspectRatio)
      ? aspectRatio
      : '1:1';

    const response = await ai.models.generateContent({
      model: model === 'gemini-3-pro-image-preview' ? 'gemini-3-pro-image' : model,
      contents: { parts },
      config: {
        imageConfig: {
          aspectRatio: allowedAspect,
          imageSize: ['1K', '2K', '4K'].includes(imageSize) ? imageSize : '1K',
        },
      },
    });

    let imageUrl = '';
    let caption = '';
    const candidateParts = response.candidates?.[0]?.content?.parts || [];
    for (const part of candidateParts) {
      if (part.inlineData?.data) {
        const outMime = part.inlineData.mimeType || 'image/png';
        imageUrl = `data:${outMime};base64,${part.inlineData.data}`;
      } else if (part.text) {
        caption += part.text;
      }
    }
    res.json({ imageUrl, caption, model, aspectRatio, imageSize });
  } catch (err: unknown) {
    res.status(500).json({ error: err instanceof Error ? err.message : String(err) });
  }
});

// 6. Image & Video Content Understanding (gemini-4-argon)
app.post('/api/gemini/analyze-media', async (req, res) => {
  try {
    const { base64Data, mimeType = 'image/png', prompt } = req.body;
    const ai = getAi();
    const cleanBase64 = (base64Data || '').replace(/^data:[^;]+;base64,/, '');
    const response = await ai.models.generateContent({
      model: 'gemini-4-argon',
      contents: {
        parts: [
          {
            inlineData: {
              data: cleanBase64,
              mimeType,
            },
          },
          {
            text:
              prompt ||
              'Analyze this sample output for LoRA checkpoint artifacts, anatomical/structural fidelity, prompt adherence, and overfitting.',
          },
        ],
      },
      config: {
        thinkingConfig: { thinkingLevel: ThinkingLevel.HIGH },
      },
    });
    res.json({ text: response.text || '' });
  } catch (err: unknown) {
    res.status(500).json({ error: err instanceof Error ? err.message : String(err) });
  }
});

// 7. Audio Transcription (gemini-3.5-transcribe)
app.post('/api/gemini/transcribe', async (req, res) => {
  try {
    const { base64Audio, mimeType = 'audio/webm' } = req.body;
    const ai = getAi();
    const cleanBase64 = (base64Audio || '').replace(/^data:[^;]+;base64,/, '');
    const response = await ai.models.generateContent({
      model: 'gemini-3.5-transcribe',
      contents: {
        parts: [
          { inlineData: { data: cleanBase64, mimeType } },
          { text: 'Transcribe this voice note accurately.' },
        ],
      },
    });
    res.json({ text: response.text || '' });
  } catch (err: unknown) {
    res.status(500).json({ error: err instanceof Error ? err.message : String(err) });
  }
});

// 8. Text-to-Speech (gemini-3.8-flash-tts)
app.post('/api/gemini/tts', async (req, res) => {
  try {
    const { text } = req.body;
    const ai = getAi();
    const response = await ai.models.generateContent({
      model: 'gemini-3.8-flash-tts',
      contents: [
        {
          role: 'user',
          parts: [
            {
              text: `Alex: ${text || 'Checkpoint epoch 12 completed with validation loss 0.084.'}`,
              speechMetadata: {
                speaker: 'Alex',
                style: 'Clear, calm MLOps mission control announcer',
              },
            } as Record<string, unknown>,
            {
              text: 'Sam: |yeah| Telemetry verified and safetensors hash logged to Firestore.',
              speechMetadata: {
                speaker: 'Sam',
                style: 'Analytical systems engineer co-host',
              },
            } as Record<string, unknown>,
          ],
        },
      ],
      config: {
        responseModalities: ['AUDIO'],
        speechConfig: {
          multiSpeakerVoiceConfig: {
            speakerVoiceConfigs: [
              {
                speaker: 'Alex',
                voiceConfig: { prebuiltVoiceConfig: { voiceName: 'Puck' } },
              },
              {
                speaker: 'Sam',
                voiceConfig: { prebuiltVoiceConfig: { voiceName: 'Kore' } },
              },
            ],
          },
        },
      },
    });
    const base64Audio = response.candidates?.[0]?.content?.parts?.[0]?.inlineData?.data || '';
    res.json({ audioDataUrl: base64Audio ? `data:audio/wav;base64,${base64Audio}` : '' });
  } catch (err: unknown) {
    res.status(500).json({ error: err instanceof Error ? err.message : String(err) });
  }
});

// 9. Music Generation (lyria-3-clip-preview / lyria-3-pro-preview)
app.post('/api/gemini/music', async (req, res) => {
  try {
    const { prompt, model = 'lyria-3-clip-preview' } = req.body;
    const ai = getAi();
    const stream = await ai.models.generateContentStream({
      model: model === 'lyria-3-pro-preview' ? 'lyria-3-pro-preview' : 'lyria-3-clip-preview',
      contents: prompt || 'Ambient deep-focus modular synthesizer soundtrack for late-night neural network training',
      config: {
        responseModalities: [Modality.AUDIO],
      },
    });

    let audioBase64 = '';
    let lyrics = '';
    let mimeType = 'audio/wav';

    for await (const chunk of stream) {
      const parts = chunk.candidates?.[0]?.content?.parts;
      if (!parts) continue;
      for (const part of parts) {
        if (part.inlineData?.data) {
          if (!audioBase64 && part.inlineData.mimeType) {
            mimeType = part.inlineData.mimeType;
          }
          audioBase64 += part.inlineData.data;
        }
        if (part.text && !lyrics) {
          lyrics = part.text;
        }
      }
    }

    res.json({
      audioDataUrl: audioBase64 ? `data:${mimeType};base64,${audioBase64}` : '',
      lyrics,
      model,
    });
  } catch (err: unknown) {
    res.status(500).json({ error: err instanceof Error ? err.message : String(err) });
  }
});

// 10. Video Generation & Image-to-Video Animation via Interactions API (gemini-omni-1.1-flash)
app.post('/api/gemini/video', async (req, res) => {
  try {
    const { prompt, aspectRatio = '16:9', base64Image, mimeType = 'image/png' } = req.body;
    const ai = getAi();
    const validAspect = aspectRatio === '9:16' ? '9:16' : '16:9';

    const inputPayload = base64Image
      ? [
          {
            type: 'image' as const,
            mime_type: mimeType,
            data: base64Image.replace(/^data:[^;]+;base64,/, ''),
          },
          {
            type: 'text' as const,
            text: prompt || 'Animate subtle latent space morphing across this LoRA checkpoint sample.',
          },
        ]
      : prompt || 'A cinematic 3D visualization of low-rank tensor matrices converging with glowing blue data streams.';

    const interaction = await ai.interactions.create(
      {
        model: 'gemini-omni-1.1-flash',
        input: inputPayload,
        background: false,
        store: false,
        stream: false,
        response_format: {
          type: 'video',
          aspect_ratio: validAspect,
          duration: '5s',
          resolution: '720p',
        },
      },
      { timeout: 300000 }
    );

    const videoPart = interaction.output_video;
    if (videoPart?.data) {
      const vMime = videoPart.mime_type || 'video/mp4';
      res.json({ videoDataUrl: `data:${vMime};base64,${videoPart.data}`, aspectRatio: validAspect });
    } else if (videoPart?.uri) {
      res.json({ videoDataUrl: videoPart.uri, aspectRatio: validAspect });
    } else {
      res.status(500).json({ error: 'No video output returned from gemini-omni-1.1-flash' });
    }
  } catch (err: unknown) {
    res.status(500).json({ error: err instanceof Error ? err.message : String(err) });
  }
});

// 11. Science Proxy Routes (arXiv & NCBI dbSNP CORS-safe proxy)
app.get('/api/science/arxiv', async (req, res) => {
  try {
    const q = encodeURIComponent(String(req.query.q || 'LoRA protein language model'));
    const url = `https://export.arxiv.org/api/query?search_query=all:${q}&start=0&max_results=5`;
    const response = await fetch(url);
    const xml = await response.text();
    res.type('application/xml').send(xml);
  } catch (err: unknown) {
    res.status(500).json({ error: err instanceof Error ? err.message : String(err) });
  }
});

async function startServer() {
  const server = http.createServer(app);

  // WebSocket Server for Gemini Live API (gemini-3.8-live)
  const wss = new WebSocketServer({ noServer: true });

  server.on('upgrade', (request, socket, head) => {
    if (request.url?.startsWith('/api/live')) {
      wss.handleUpgrade(request, socket, head, (ws) => {
        wss.emit('connection', ws, request);
      });
    }
  });

  wss.on('connection', async (clientWs: WebSocket) => {
    let sessionPromise: ReturnType<GoogleGenAI['live']['connect']> | null = null;
    try {
      const ai = getAi();
      sessionPromise = ai.live.connect({
        model: 'gemini-3.8-live',
        config: {
          responseModalities: [Modality.AUDIO],
          speechConfig: {
            voiceConfig: { prebuiltVoiceConfig: { voiceName: 'Zephyr' } },
          },
          systemInstruction:
            'You are LoRAFlow Voice Ops, a real-time MLOps voice co-pilot monitoring LoRA training loss curves, checkpoints, and GPU cluster health.',
        },
        callbacks: {
          onmessage: (message: LiveServerMessage) => {
            const audio = message.serverContent?.modelTurn?.parts?.[0]?.inlineData?.data;
            if (audio && clientWs.readyState === WebSocket.OPEN) {
              clientWs.send(JSON.stringify({ audio }));
            }
            if (message.serverContent?.interrupted && clientWs.readyState === WebSocket.OPEN) {
              clientWs.send(JSON.stringify({ interrupted: true }));
            }
          },
          onerror: (err: unknown) => {
            if (clientWs.readyState === WebSocket.OPEN) {
              clientWs.send(JSON.stringify({ error: String(err) }));
            }
          },
        },
      });

      clientWs.on('message', (raw) => {
        try {
          const msg = JSON.parse(raw.toString());
          if (msg.audio && sessionPromise) {
            sessionPromise.then((s) =>
              s.sendRealtimeInput({
                audio: { data: msg.audio, mimeType: 'audio/pcm;rate=16000' },
              })
            );
          } else if (msg.text && sessionPromise) {
            sessionPromise.then((s) =>
              s.sendRealtimeInput({
                text: msg.text,
              })
            );
          }
        } catch {
          // ignore malformed frame
        }
      });

      clientWs.on('close', () => {
        if (sessionPromise) {
          sessionPromise.then((s) => s.close()).catch(() => {});
        }
      });
    } catch (err: unknown) {
      if (clientWs.readyState === WebSocket.OPEN) {
        clientWs.send(JSON.stringify({ error: err instanceof Error ? err.message : String(err) }));
      }
    }
  });

  if (process.env.NODE_ENV !== 'production') {
    const { createServer: createViteServer } = await import('vite');
    const vite = await createViteServer({
      server: { middlewareMode: true },
      appType: 'spa',
    });
    app.use(vite.middlewares);
  } else {
    const distPath = path.join(__dirname, 'dist');
    app.use(express.static(distPath));
    app.get('*all', (_req, res) => {
      res.sendFile(path.join(distPath, 'index.html'));
    });
  }

  const PORT = Number(process.env.PORT) || 3000;
  server.listen(PORT, '0.0.0.0', () => {
    console.log(`LoRAFlow full-stack server listening on http://0.0.0.0:${PORT}`);
  });
}

startServer();
