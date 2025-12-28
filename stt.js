const fs = require('fs');
const path = require('path');
const vosk = require("vosk");
const { Readable } = require('stream');
const { config } = require("./config");

const MODEL_PATH = `./model/${config.vaskmodel}`

let model = null;
let gradioModulePromise = null;

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
    return String(result[0] ?? '').trim();
  }
  if (result.data && Array.isArray(result.data)) {
    return String(result.data[0] ?? '').trim();
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

let gradioClientPromise = null;

async function loadGradioModule() {
  if (!gradioModulePromise) {
    gradioModulePromise = import('@gradio/client');
  }
  return gradioModulePromise;
}

async function getGradioClient() {
  if (gradioClientPromise) return gradioClientPromise;
  const apiUrl = config.whisperApiUrl;
  if (!apiUrl) {
    throw new Error('Missing whisperApiUrl in config.');
  }
  const gradio = await loadGradioModule();
  const GradioClient = gradio.Client || gradio.client || gradio.default;
  if (GradioClient && typeof GradioClient.connect === 'function') {
    gradioClientPromise = GradioClient.connect(apiUrl);
  } else if (typeof GradioClient === 'function') {
    gradioClientPromise = GradioClient(apiUrl);
  } else {
    throw new Error('Unsupported @gradio/client API.');
  }
  return gradioClientPromise;
}

async function buildGradioFiles(filePath) {
  const gradio = await loadGradioModule();
  const handleFile =
    gradio.handle_file ||
    gradio.handleFile ||
    (gradio.default && (gradio.default.handle_file || gradio.default.handleFile));
  const resolvedPath = path.resolve(filePath);
  if (typeof handleFile === 'function') {
    return [await handleFile(resolvedPath)];
  }
  if (typeof File !== 'undefined') {
    const data = await fs.promises.readFile(resolvedPath);
    const baseName = path.basename(resolvedPath);
    const ext = path.extname(baseName).toLowerCase();
    const name = ext ? baseName : `${baseName}.wav`;
    return [new File([data], name, { type: 'audio/wav' })];
  }
  throw new Error('Cannot build Gradio file payload. handle_file and File are unavailable.');
}

function normalizeGradioDevice(device) {
  const raw = String(device || '').toLowerCase();
  if (raw === 'gpu' || raw === 'cuda') return 'cuda';
  return 'cpu';
}

async function transcribeWithWhisper(filePath, options = {}) {
  const client = await getGradioClient();
  const modelName = normalizeWhisperModel(options.model || config.whisperModel || 'base');
  const device = normalizeGradioDevice(options.device || config.whisperDevice || 'cpu');
  const apiName = options.apiName || config.whisperApiName || '/transcribe_file';
  const files = await buildGradioFiles(path.resolve(filePath));

  const payload = {
    files,
    input_folder_path: '',
    include_subdirectory: false,
    save_same_dir: true,
    file_format: options.fileFormat || 'SRT',
    add_timestamp: false,
    progress: modelName,
    param_7: options.language || 'Automatic Detection',
    param_8: false,
    param_9: 5,
    param_10: -1,
    param_11: 0.6,
    param_12: options.computeType || 'float16',
    param_13: 5,
    param_14: 1,
    param_15: true,
    param_16: 0.5,
    param_17: options.initialPrompt || '',
    param_18: 0,
    param_19: 2.4,
    param_20: 1,
    param_21: 1,
    param_22: 0,
    param_23: options.prefix || '',
    param_24: true,
    param_25: '[-1]',
    param_26: 1,
    param_27: false,
    param_28: `"'“¿([{-`,
    param_29: `"'.。,，!！?？:：”)]}、`,
    param_30: options.maxNewTokens ?? 3,
    param_31: 30,
    param_32: options.hallucinationSilenceThreshold ?? 3,
    param_33: options.hotwords || '',
    param_34: 0.5,
    param_35: 1,
    param_36: 24,
    param_37: true,
    param_38: false,
    param_39: 0.5,
    param_40: 250,
    param_41: 9999,
    param_42: 1000,
    param_43: 2000,
    param_44: false,
    param_45: device,
    param_46: '',
    param_47: true,
    param_48: false,
    param_49: 'UVR-MDX-NET-Inst_HQ_4',
    param_50: device,
    param_51: 256,
    param_52: false,
    param_53: true,
  };

  const result = await client.predict(apiName, payload);
  return extractWhisperText(result);
}

module.exports = {
  transcribeWithVosk,
  transcribeWithWhisper,
  voskLoader,
  normalizeWhisperModel,
};
