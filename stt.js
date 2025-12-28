const fs = require('fs');
const path = require('path');
const vosk = require("vosk");
const { Readable } = require('stream');
const { config } = require("./config");

const MODEL_PATH = `./model/${config.vaskmodel}`

const WHISPER_DIR = 'whisper';
const WHISPER_MODELS_DIR = path.join(WHISPER_DIR, 'models');
const WHISPER_MODEL_FILES = {
  tiny: 'ggml-tiny.bin',
  base: 'ggml-base.bin',
  small: 'ggml-small.bin',
  medium: 'ggml-medium.bin',
  large: 'ggml-large-v1.bin',
  largev2: 'ggml-large-v2.bin',
};

let model = null;

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
  if (normalized === 'large-v2' || normalized === 'large_v2') return 'largev2';
  return normalized;
}

function getWhisperModelPath(modelName) {
  const normalized = normalizeWhisperModel(modelName);
  const fileName = WHISPER_MODEL_FILES[normalized];
  if (!fileName) return null;
  return path.join(WHISPER_MODELS_DIR, fileName);
}

function getWhisperCppModelName(modelName) {
  const normalized = normalizeWhisperModel(modelName);
  if (normalized === 'largev2') return 'large-v2';
  return normalized;
}

function loadWhisperModule() {
  try {
    return require('nodejs-whisper');
  } catch (err) {
    throw new Error('Whisper module not installed. Add nodejs-whisper to dependencies.');
  }
}

function resolveWhisperFunction(whisperModule) {
  if (!whisperModule) return null;
  if (typeof whisperModule === 'function') return whisperModule;
  if (typeof whisperModule.nodewhisper === 'function') return whisperModule.nodewhisper;
  if (typeof whisperModule.whisper === 'function') return whisperModule.whisper;
  if (typeof whisperModule.transcribe === 'function') return whisperModule.transcribe;
  if (typeof whisperModule.default === 'function') return whisperModule.default;
  return null;
}

function extractWhisperText(result) {
  if (!result) return '';
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

async function transcribeWithWhisper(filePath, options = {}) {
  const whisperModule = loadWhisperModule();
  const whisperFn = resolveWhisperFunction(whisperModule);
  if (typeof whisperFn !== 'function') {
    const exportKeys = Object.keys(whisperModule || {}).join(', ') || 'none';
    throw new Error(`Unsupported whisper module API. Exports: ${exportKeys}`);
  }

  const modelName = normalizeWhisperModel(options.model || config.whisperModel || 'base');
  const modelFilePath = getWhisperModelPath(modelName);
  if (!modelFilePath || !fs.existsSync(modelFilePath)) {
    throw new Error(`Whisper model not found: ${modelFilePath || modelName}`);
  }

  const device = String(options.device || config.whisperDevice || 'cpu').toLowerCase();
  const whisperOptions = {
    modelName: getWhisperCppModelName(modelName),
    modelPath: WHISPER_MODELS_DIR,
    whisperOptions: {
      gpu: device === 'gpu',
      language: options.language,
    },
  };

  const result = await whisperFn(filePath, whisperOptions);
  return extractWhisperText(result);
}

module.exports = {
  transcribeWithVosk,
  transcribeWithWhisper,
  voskLoader,
  WHISPER_MODELS_DIR,
  WHISPER_MODEL_FILES,
  normalizeWhisperModel,
  getWhisperModelPath,
};
