const { spawn } = require("child_process");
const fs = require('fs-extra');
const path = require('path');
const { randomUUID } = require('crypto');
const logger = console;

// Store TTS configurations and user preferences
const ttsConfigs = {};

// Load TTS configurations
async function loadTtsConfigs() {
  try {
    await fs.mkdir("tts_configs", { recursive: true });
    const files = await fs.readdir("tts_configs");
    let loadedCount = 0;
    let errorCount = 0;
    for (const file of files) {
      if (file.endsWith(".json")) {
        const providerName = file.slice(0, -5);
        const configPath = path.join("tts_configs", file);
        const configData = await fs.readFile(configPath, "utf-8");
        try {
          ttsConfigs[providerName] = JSON.parse(configData);
          loadedCount += 1;
        } catch (error) {
          errorCount += 1;
        }
      }
    }
    if (loadedCount > 0 || errorCount > 0) {
      console.log(`TTS configs: loaded=${loadedCount}, errors=${errorCount}`);
    }
  } catch (error) {
    console.error(`Error loading TTS configs: ${error.message}`);
  }
}

// Generate TTS audio using piper-tts
async function generateTts(
  text,
  provider = `en_US-amy-medium`,
  outputFile = `uploads/${randomUUID()}.wav`
) {;
  const configstt = require(`./tts_configs/${provider}.json`);
  if (!configstt) {
    logger.error(`No TTS config found for provider: ${provider}`);
    return null;
  }

  const voice = provider || "en_US-lessac-medium";
  const modelPath = path.resolve(
    configstt.modelPath || `./piper/models/${voice}.onnx`
  );

  return new Promise((resolve, reject) => {
    const args = [
      "-m",
      "piper",
      "--model",
      modelPath,
      "--output_file",
      outputFile
    ];
    const child = spawn(configstt.pythonPath, args, {
      stdio: ["pipe", "inherit", "inherit"]
    });

    child.stdin.write(text + "\n");
    child.stdin.end();

    child.on("close", code => {
      if (code === 0) resolve(outputFile);
      else reject(new Error(`Piper exited with code ${code}`));
    });

    child.on("error", reject);
  });
}

function stripEmojis(text) {
  return text.replace(/[\p{Emoji_Presentation}\p{Emoji}\u200d]+/gu, '').trim();
}

async function synthesizeWithPiper(text, voice) {
  const outputWav = `uploads/${randomUUID()}.wav`;

  return new Promise(async (resolve, reject) => {
    const cleanSentence = stripEmojis(text);
    const audioFileAi = await generateTts(
      cleanSentence,
      voice || "en_US-lessac-medium",
      outputWav
    );

    console.log(audioFileAi)

    resolve(audioFileAi);
  });
}

module.exports = { synthesizeWithPiper, loadTtsConfigs };
