import "dotenv/config";
import express from "express";
import cors from "cors";
import OpenAI from "openai";

const PORT = Number(process.env.PORT || 8787);
const MODEL = "gpt-5.2";
const ALLOWED_MODES = new Set(["breakdown", "dailyBriefing", "review", "cleanup", "nextTask"]);

const app = express();
app.use(cors({ origin: true }));
app.use(express.json({ limit: "512kb" }));

const client = new OpenAI({ apiKey: process.env.OPENAI_API_KEY });

const assistantSchema = {
  type: "object",
  additionalProperties: false,
  required: ["summary", "suggestions"],
  properties: {
    summary: { type: "string" },
    suggestions: {
      type: "array",
      maxItems: 8,
      items: {
        type: "object",
        additionalProperties: false,
        required: ["type", "title", "reason", "confidence", "actions"],
        properties: {
          type: { type: "string", enum: ["subtask", "insight", "cleanup", "recommendation", "briefing"] },
          title: { type: "string" },
          reason: { type: "string" },
          confidence: { type: "number", minimum: 0, maximum: 1 },
          actions: { type: "array", maxItems: 5, items: { type: "string" } }
        }
      }
    }
  }
};

const systemInstructions = `
You are the optional AI layer for a local Personal Command Centre task app.
Be practical, concise, calm, and action-oriented.
Return JSON only, matching the provided schema.
Do not invent deadlines, commitments, categories, or completed work.
Do not mark tasks done or imply changes were applied.
Prefer small next actions and preserve user agency.
Use the supplied local rule-based insights as context, but improve them when useful.
For breakdown mode, suggestions should be actionable subtasks, not commentary.
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

function normalizeAssistantResponse(value) {
  const result = value && typeof value === "object" ? value : {};
  const suggestions = Array.isArray(result.suggestions) ? result.suggestions : [];
  return {
    summary: String(result.summary || "AI suggestions are ready.").slice(0, 500),
    suggestions: suggestions.slice(0, 8).map(item => ({
      type: ["subtask", "insight", "cleanup", "recommendation", "briefing"].includes(item?.type) ? item.type : "insight",
      title: String(item?.title || "").slice(0, 180),
      reason: String(item?.reason || "").slice(0, 300),
      confidence: Math.max(0, Math.min(1, Number(item?.confidence ?? 0.5))),
      actions: Array.isArray(item?.actions) ? item.actions.slice(0, 5).map(action => String(action).slice(0, 160)) : []
    })).filter(item => item.title)
  };
}

app.get("/health", (_req, res) => {
  res.json({ ok: true, model: MODEL, hasApiKey: Boolean(process.env.OPENAI_API_KEY) });
});

app.post("/api/assistant", async (req, res) => {
  if (!process.env.OPENAI_API_KEY) {
    return res.status(503).json({ error: "OPENAI_API_KEY is not configured." });
  }

  const payload = sanitizeBody(req.body);
  if (!ALLOWED_MODES.has(payload.mode)) {
    return res.status(400).json({ error: "Invalid assistant mode." });
  }

  try {
    const response = await client.responses.create({
      model: MODEL,
      instructions: systemInstructions,
      input: [
        {
          role: "user",
          content: [
            {
              type: "input_text",
              text: JSON.stringify({
                request: "Return assistant suggestions for the Personal Command Centre.",
                payload
              })
            }
          ]
        }
      ],
      text: {
        format: {
          type: "json_schema",
          name: "command_centre_assistant_response",
          strict: true,
          schema: assistantSchema
        }
      }
    });

    const parsed = JSON.parse(response.output_text || "{}");
    res.json(normalizeAssistantResponse(parsed));
  } catch (error) {
    console.error("Assistant API error:", error);
    res.status(500).json({ error: "Assistant request failed." });
  }
});

app.listen(PORT, () => {
  console.log(`Personal Command Centre assistant backend listening on http://localhost:${PORT}`);
});
