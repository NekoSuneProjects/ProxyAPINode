const fs = require('fs');
const path = require('path');
const vosk = require("vosk");
const { Readable } = require('stream');
const { config } = require("./config");

const MODEL_PATH = `./model/${config.vaskmodel}`

let model = null;
const GRADIO_FILE_META = { _type: 'gradio.FileData' };

async function voskLoader() {
  // Only load model if not already loaded
  if (model !== null) return model;

  if (!fs.existsSync(MODEL_PATH)) {
    console.error(
      "❌ Vosk model still not found after attempted download:",
      MODEL_PATH
    );
    process.exit(1);
  }

  vosk.setLogLevel(0);
  model = new vosk.Model(MODEL_PATH);
  console.log("✅ Vosk model loaded.");
  return model;
}

async function transcribeWithVosk(filePath) {
  return new Promise(async (resolve, reject) => {
    try {
      const wfReader = fs.createReadStream(filePath, { highWaterMark: 4096 });
      const model = await voskLoader();

      const rec = new vosk.Recognizer({ model, sampleRate: 16000 });
      rec.setMaxAlternatives(1);
      rec.setWords(true);

      wfReader.on("data", chunk => rec.acceptWaveform(chunk));

      wfReader.on("end", () => {
        try {
          const finalResult = rec.finalResult();
          rec.free();

          const text =
            finalResult?.alternatives?.[0]?.text?.trim() ?? "";

          if (text) {
            resolve(text);
          } else {
            resolve(false); // no text detected
          }
        } catch (err) {
          rec.free();
          reject(err);
        }
      });

      wfReader.on("error", err => {
        rec.free();
        reject(err);
      });
    } catch (err) {
      reject(err);
    }
  });
}

function normalizeWhisperModel(modelName) {
  if (!modelName) return 'base';
  const normalized = String(modelName).toLowerCase().trim();
  if (normalized === 'mid') return 'medium';
  if (normalized === 'large_v2' || normalized === 'largev2') return 'large-v2';
  return normalized;
}

function extractWhisperText(result) {
  if (!result) return '';
  if (Array.isArray(result)) {
    const first = result[0];
    if (first && typeof first === 'object' && typeof first.text === 'string') {
      return first.text.trim();
    }
    return String(first ?? '').trim();
  }
  if (result.data && Array.isArray(result.data)) {
    return String(result.data[0] ?? '').trim();
  }
  if (result.data && result.data.data && Array.isArray(result.data.data)) {
    return String(result.data.data[0] ?? '').trim();
  }
  if (typeof result === 'string') {
    if (fs.existsSync(result)) {
      return fs.readFileSync(result, 'utf8').trim();
    }
    return result.trim();
  }
  if (typeof result.text === 'string') return result.text.trim();
  if (typeof result.transcription === 'string') return result.transcription.trim();
  if (result.data && typeof result.data.text === 'string') return result.data.text.trim();
  if (typeof result.outputPath === 'string' && fs.existsSync(result.outputPath)) {
    return fs.readFileSync(result.outputPath, 'utf8').trim();
  }
  return '';
}

function stripSrt(text) {
  let cleaned = String(text || '').trim();
  cleaned = cleaned.replace(/^\d{10,}-\d{6,}\s*/, '');
  const lines = cleaned.split(/\r?\n/);
  const output = [];
  for (const line of lines) {
    const trimmed = line.trim();
    if (!trimmed) continue;
    if (/^\d+$/.test(trimmed)) continue;
    if (/^\d{2}:\d{2}:\d{2},\d{3}\s-->\s\d{2}:\d{2}:\d{2},\d{3}$/.test(trimmed)) {
      continue;
    }
    if (trimmed.startsWith('Done in ') || trimmed === '------------------------------------') continue;
    output.push(trimmed);
  }
  return output.join(' ').trim();
}

async function extractWhisperTextFromResult(result) {
  if (Array.isArray(result) && Array.isArray(result[1]) && result[1][0] && result[1][0].url) {
    const response = await fetch(result[1][0].url);
    if (!response.ok) {
      throw new Error(`Whisper output fetch failed: ${response.status} ${response.statusText}`);
    }
    const text = await response.text();
    return stripSrt(text);
  }
  const rawText = extractWhisperText(result);
  return stripSrt(rawText);
}

function normalizeBaseUrl(url) {
  return String(url).replace(/\/+$/, '');
}

function getMimeType(fileName) {
  const ext = path.extname(fileName).toLowerCase();
  if (ext === '.wav') return 'audio/wav';
  if (ext === '.mp3') return 'audio/mpeg';
  if (ext === '.ogg') return 'audio/ogg';
  if (ext === '.flac') return 'audio/flac';
  if (ext === '.m4a') return 'audio/mp4';
  if (ext === '.aac') return 'audio/aac';
  return 'application/octet-stream';
}

async function uploadToGradio(apiUrl, filePath) {
  const resolvedPath = path.resolve(filePath);
  const data = await fs.promises.readFile(resolvedPath);
  const baseName = path.basename(resolvedPath);
  const ext = path.extname(baseName).toLowerCase();
  const name = ext ? baseName : `${baseName}.wav`;
  const mimeType = getMimeType(name);

  const form = new FormData();
  form.append('files', new Blob([data], { type: mimeType }), name);

  const uploadUrl = `${normalizeBaseUrl(apiUrl)}/gradio_api/upload`;
  const response = await fetch(uploadUrl, { method: 'POST', body: form });
  if (!response.ok) {
    throw new Error(`Gradio upload failed: ${response.status} ${response.statusText}`);
  }
  const result = await response.json();
  if (!Array.isArray(result) || !result[0]) {
    throw new Error('Gradio upload returned no file path.');
  }
  return result[0];
}

async function callGradio(apiUrl, apiName, dataArray) {
  const callUrl = `${normalizeBaseUrl(apiUrl)}/gradio_api/call${apiName}`;
  const response = await fetch(callUrl, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ data: dataArray }),
  });
  if (!response.ok) {
    throw new Error(`Gradio call failed: ${response.status} ${response.statusText}`);
  }
  const result = await response.json();
  const eventId = result.event_id || result.eventId || result.id;
  if (!eventId) {
    throw new Error('Gradio call returned no event id.');
  }

  const eventUrl = `${normalizeBaseUrl(apiUrl)}/gradio_api/call${apiName}/${eventId}`;
  const eventResponse = await fetch(eventUrl);
  if (!eventResponse.ok) {
    throw new Error(`Gradio event failed: ${eventResponse.status} ${eventResponse.statusText}`);
  }
  const text = await eventResponse.text();
  const lines = text.split('\n').map((line) => line.trim()).filter(Boolean);
  const dataLines = lines.filter((line) => line.startsWith('data:'));
  if (!dataLines.length) {
    throw new Error('Gradio event returned no data.');
  }
  const lastData = dataLines[dataLines.length - 1].slice('data:'.length).trim();
  try {
    return JSON.parse(lastData);
  } catch {
    return lastData;
  }
}

function normalizeGradioDevice(device) {
  const raw = String(device || '').toLowerCase();
  if (raw === 'gpu' || raw === 'cuda') return 'cuda';
  return 'cpu';
}

async function transcribeWithWhisperPrimary(filePath, options = {}) {
  const url = options.primaryUrl || config.whisperPrimaryUrl || 'http://localhost:8080/v1/audio/transcriptions';
  const modelName = options.primaryModel || options.model || config.whisperPrimaryModel || config.whisperModel || 'base';
  const resolvedPath = path.resolve(filePath);

  const form = new FormData();
  form.append('file', fs.createReadStream(resolvedPath));
  form.append('model', modelName);

  const response = await fetch(url, { method: 'POST', body: form });

  if (!response.ok) {
    throw new Error(`Primary whisper failed: ${response.status} ${response.statusText}`);
  }

  const result = await response.json();
  const text = result?.text ?? result?.transcript ?? result?.transcription;

  if (!text) {
    throw new Error('Primary whisper returned no text.');
  }

  return String(text).trim();
}

async function transcribeWithWhisper(filePath, options = {}) {
  const apiUrl = config.whisperApiUrl;
  if (!apiUrl) {
    throw new Error('Missing whisperApiUrl in config.');
  }
  const modelName = normalizeWhisperModel(options.model || config.whisperModel || 'base');
  const device = normalizeGradioDevice(options.device || config.whisperDevice || 'cpu');
  const apiName = options.apiName || config.whisperApiName || '/transcribe_file';
  const uploadPath = await uploadToGradio(apiUrl, filePath);

  const data = [
    [{ path: uploadPath, meta: GRADIO_FILE_META }],
    '',
    false,
    true,
    options.fileFormat || 'txt',
    false,
    modelName,
    options.language || 'Automatic Detection',
    false,
    5,
    -1,
    0.6,
    options.computeType || 'float16',
    5,
    1,
    true,
    0.5,
    options.initialPrompt || '',
    0,
    2.4,
    1,
    1,
    0,
    options.prefix || '',
    true,
    '[-1]',
    1,
    false,
    `"'([{-`,
    `"'.!?:)]}`,
    options.maxNewTokens ?? 3,
    30,
    options.hallucinationSilenceThreshold ?? 3,
    options.hotwords || '',
    0.5,
    1,
    24,
    true,
    false,
    0.5,
    250,
    9999,
    1000,
    2000,
    false,
    device,
    '',
    true,
    false,
    'UVR-MDX-NET-Inst_HQ_4',
    device,
    256,
    false,
    true,
  ];

  const result = await callGradio(apiUrl, apiName, data);
  return await extractWhisperTextFromResult(result);
}

async function transcribeWithWhisperWithFallback(filePath, options = {}) {
  try {
    return await transcribeWithWhisperPrimary(filePath, options);
  } catch (err) {
    try {
      return await transcribeWithWhisper(filePath, options);
    } catch (innerErr) {
      return await transcribeWithVosk(filePath);
    }
  }
}

module.exports = {
  transcribeWithVosk,
  transcribeWithWhisper,
  transcribeWithWhisperWithFallback,
  voskLoader,
  normalizeWhisperModel,
};
