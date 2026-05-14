import "dotenv/config";
import express from "express";
import cors from "cors";

const PORT = Number(process.env.PORT || 8787);
const PROVIDER = "ollama";
const OLLAMA_MODEL = "llama3";
const OLLAMA_ENDPOINT = process.env.OLLAMA_ENDPOINT || "http://localhost:11434/api/generate";
const REQUEST_TIMEOUT_MS = Number(process.env.ASSISTANT_TIMEOUT_MS || 45000);
const ALLOWED_MODES = new Set(["breakdown", "dailyBriefing", "review", "cleanup", "nextTask"]);
const SUGGESTION_TYPES = new Set(["subtask", "insight", "cleanup", "recommendation", "briefing"]);

// To use a smaller local model, install it with `ollama pull <model>` and change
// OLLAMA_MODEL above, for example "llama3.2:3b" or another model you have locally.
// To switch back to OpenAI later, keep the /api/assistant contract and replace
// callOllama() with an OpenAI provider function that returns the same JSON shape.

const app = express();
app.use(cors({ origin: true }));
app.use(express.json({ limit: "512kb" }));

const systemInstructions = `
You are the optional local AI layer for a Personal Command Centre task app.
You are running through Ollama on the user's machine.
Be practical, concise, calm, and action-oriented.
Return ONLY valid JSON. Do not include markdown, code fences, or commentary.
Use this exact shape:
{
  "summary": "short human-readable summary",
  "suggestions": [
    {
      "type": "subtask|insight|cleanup|recommendation|briefing",
      "title": "...",
      "reason": "...",
      "confidence": 0.0,
      "actions": []
    }
  ]
}
Do not invent deadlines, commitments, categories, or completed work.
Do not mark tasks done or imply changes were applied.
Prefer small next actions and preserve user agency.
For breakdown mode, suggestions should be actionable subtasks.
For cleanup mode, suggest review actions without deleting or modifying anything.
`;

function sanitizeBody(body) {
  return {
    mode: String(body?.mode || ""),
    focusMode: String(body?.focusMode || "normal"),
    task: body?.task && typeof body.task === "object" ? body.task : null,
    tasks: Array.isArray(body?.tasks) ? body.tasks.slice(0, 60) : [],
    localInsights: Array.isArray(body?.localInsights) ? body.localInsights.slice(0, 10) : [],
    userFeedback: body?.userFeedback && typeof body.userFeedback === "object" ? body.userFeedback : {}
  };
}

function promptForPayload(payload) {
  return `${systemInstructions}

Assistant mode: ${payload.mode}
Focus mode: ${payload.focusMode}

Input context as JSON:
${JSON.stringify(payload, null, 2)}

Return only the JSON object.`;
}

function normalizeAssistantResponse(value) {
  const result = value && typeof value === "object" ? value : {};
  const suggestions = Array.isArray(result.suggestions) ? result.suggestions : [];
  return {
    summary: String(result.summary || "Local AI suggestions are ready.").slice(0, 500),
    suggestions: suggestions.slice(0, 8).map(item => ({
      type: SUGGESTION_TYPES.has(item?.type) ? item.type : "insight",
      title: String(item?.title || "").slice(0, 180),
      reason: String(item?.reason || "").slice(0, 300),
      confidence: Math.max(0, Math.min(1, Number(item?.confidence ?? 0.5))),
      actions: Array.isArray(item?.actions) ? item.actions.slice(0, 5).map(action => String(action).slice(0, 160)) : []
    })).filter(item => item.title)
  };
}

function fallbackAssistantResponse(mode, detail = "The local model did not return valid JSON.") {
  return {
    summary: "Local AI could not produce structured suggestions.",
    suggestions: [
      {
        type: mode === "breakdown" ? "subtask" : "insight",
        title: "Use the local rule-based assistant fallback",
        reason: detail,
        confidence: 0.2,
        actions: ["Try again", "Check Ollama is running", "Use the existing local suggestions"]
      }
    ]
  };
}

function extractJsonObject(text) {
  const raw = String(text || "").trim();
  if (!raw) throw new Error("Empty Ollama response.");
  try { return JSON.parse(raw); } catch {}

  const fenced = raw.match(/```(?:json)?\s*([\s\S]*?)\s*```/i);
  if (fenced) {
    try { return JSON.parse(fenced[1]); } catch {}
  }

  const start = raw.indexOf("{");
  const end = raw.lastIndexOf("}");
  if (start >= 0 && end > start) return JSON.parse(raw.slice(start, end + 1));
  throw new Error("No valid JSON object found in Ollama response.");
}

async function fetchWithTimeout(url, options = {}, timeoutMs = REQUEST_TIMEOUT_MS) {
  const controller = new AbortController();
  const timer = setTimeout(() => controller.abort(), timeoutMs);
  try {
    return await fetch(url, { ...options, signal: controller.signal });
  } finally {
    clearTimeout(timer);
  }
}

async function isOllamaReachable() {
  try {
    const response = await fetchWithTimeout(OLLAMA_ENDPOINT, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ model: OLLAMA_MODEL, prompt: "Return {\"ok\":true}", stream: false, format: "json" })
    }, 5000);
    return response.ok;
  } catch {
    return false;
  }
}

async function callOllama(payload) {
  const response = await fetchWithTimeout(OLLAMA_ENDPOINT, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({
      model: OLLAMA_MODEL,
      prompt: promptForPayload(payload),
      stream: false,
      format: "json",
      options: { temperature: 0.2, num_predict: 900 }
    })
  });

  if (!response.ok) {
    const body = await response.text().catch(() => "");
    throw new Error(`Ollama returned ${response.status}${body ? `: ${body.slice(0, 180)}` : ""}`);
  }

  const data = await response.json();
  const parsed = extractJsonObject(data.response);
  return normalizeAssistantResponse(parsed);
}

app.get("/api/health", (_req, res) => {
  res.json({ ok: true, provider: PROVIDER, model: OLLAMA_MODEL });
});

app.get("/health", (_req, res) => {
  res.json({ ok: true, provider: PROVIDER, model: OLLAMA_MODEL });
});

app.post("/api/assistant", async (req, res) => {
  const payload = sanitizeBody(req.body);
  if (!ALLOWED_MODES.has(payload.mode)) {
    return res.status(400).json({ error: "Invalid assistant mode." });
  }

  try {
    res.json(await callOllama(payload));
  } catch (error) {
    const message = error.name === "AbortError" ? "Ollama request timed out." : error.message;
    console.error("Ollama assistant error:", message);
    res.status(503).json({ error: message, fallback: fallbackAssistantResponse(payload.mode, message) });
  }
});

app.listen(PORT, async () => {
  const reachable = await isOllamaReachable();
  console.log(`Personal Command Centre assistant backend listening on http://localhost:${PORT}`);
  console.log(`Provider: ${PROVIDER}`);
  console.log(`Model: ${OLLAMA_MODEL}`);
  console.log(`Ollama endpoint: ${OLLAMA_ENDPOINT}`);
  console.log(`Ollama reachable: ${reachable ? "yes" : "no"}`);
});
