import express from 'express';
import cors from 'cors';
import path from 'path';
import { fileURLToPath } from 'url';
import { GoogleGenAI } from "@google/genai";
import dotenv from 'dotenv';

dotenv.config();

const app = express();
const PORT = 3010;

// Resolve __dirname for ES module
const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);

// Middleware
app.use(cors());
app.use(express.json());

// Serve static HTML files
app.get('/', (req, res) => {
  res.sendFile(path.join(__dirname, 'templates', 'home(main).html'));
});

app.get('/grammar', (req, res) => {
  res.sendFile(path.join(__dirname, 'templates', 'public.html'));
});

// Gemini AI setup
const genAI = new GoogleGenAI({ apiKey: process.env.GEMINI_API_KEY });

// Retry helper with exponential backoff for rate-limit (429) errors
async function callGeminiWithRetry(prompt, maxRetries = 3) {
  for (let attempt = 0; attempt <= maxRetries; attempt++) {
    try {
      const response = await genAI.models.generateContent({
        model: "gemini-1.5-flash",
        contents: prompt,
      });
      return response.text;
    } catch (err) {
      const isRateLimit = err.status === 429 ||
        (err.message && err.message.includes('429'));

      if (isRateLimit && attempt < maxRetries) {
        const waitMs = Math.pow(2, attempt + 1) * 1000; // 2s, 4s, 8s
        console.log(`Rate limited – retrying in ${waitMs / 1000}s (attempt ${attempt + 1}/${maxRetries})...`);
        await new Promise(resolve => setTimeout(resolve, waitMs));
      } else {
        throw err;
      }
    }
  }
}

app.post('/api/correct', async (req, res) => {
  try {
    const { text1 } = req.body;

    if (!text1) {
      return res.status(400).json({ error: "No text provided" });
    }

    const prompt = `You will be given a sentence with potential errors in capitalization, grammar, and punctuation. Your task is to correct these errors and provide all corrected possible sentences with the mistakes I made along with the explanation in detail. The output should not have any introduction sentences like here is the answer for your question.: ${text1}`;

    const correctedText = await callGeminiWithRetry(prompt);
    res.json({ correctedText });
  } catch (error) {
    console.error("Gemini API error:", error);

    const isRateLimit = error.status === 429 ||
      (error.message && error.message.includes('429'));

    if (isRateLimit) {
      return res.status(429).json({
        error: "Rate limit exceeded. Please wait a minute and try again.",
      });
    }

    res.status(500).json({
      error: "Failed to process text",
      details: error.message
    });
  }
});

app.listen(PORT, () => {
  console.log(`Server running on http://localhost:${PORT}`);
});

