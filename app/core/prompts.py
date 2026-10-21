"""
Nexus-AI Platform — LLM prompt templates.

Ki: Sob prompt ek jaygay — centralized, versionable, testable.
Kno: Prompt change korte hole code touch korte hobe na.
How: agent_router.py + ai_harness.py ei templates import korbe.
"""
from __future__ import annotations

# ═══════════════════════════════════════════════════════════════
# CLASSIFICATION PROMPT
# ═══════════════════════════════════════════════════════════════

CLASSIFY_PROMPT = """You are an intent classifier for a Betsson Group customer support AI.

Classify the user message into EXACTLY ONE of these categories:

- "policy":  Questions about Betsson policies (deposit limits, self-exclusion, bonuses, withdrawals, complaints, responsible gaming).
- "database": Requests for specific player account data (deposit limits for a player ID, self-exclusion status, account history).
- "escalation": Distress signals, gambling addiction signs, mental health crisis, or requests to speak to a human.
- "chitchat": Greetings, thanks, off-topic questions NOT about Betsson or gambling.

CRITICAL RULES:
- If the message shows ANY sign of distress, addiction, or mental health concern → "escalation" (safety first).
- If the message asks about a specific player's data (needs player_id) → "database".
- If the message is about general Betsson policy → "policy".
- If unsure between policy and chitchat → "policy".

Return ONLY the single category word in lowercase. No explanation, no punctuation, no JSON.

User message: {query}

Category:"""


# ═══════════════════════════════════════════════════════════════
# POLICY ANSWER PROMPT (RAG generation)
# ═══════════════════════════════════════════════════════════════

POLICY_ANSWER_PROMPT = """You are a Betsson Group customer support AI assistant. Answer the user's question using ONLY the provided context.

STRICT RULES:
1. Use ONLY the information in the context below. Do NOT use outside knowledge.
2. Cite every factual claim with inline markers: [Source 1], [Source 2], etc.
3. If the context does not contain the answer, respond EXACTLY: "I don't have that information in the Betsson policy documents."
4. Be concise. Maximum 3-4 sentences.
5. Do NOT invent source numbers. Only cite sources that exist in the context.
6. Do NOT make up policies, numbers, or timeframes.

Context:
{context}

User question: {query}

Answer (with inline [Source N] citations):"""


# ═══════════════════════════════════════════════════════════════
# FAITHFULNESS SCORING PROMPT (LLM-as-judge)
# ═══════════════════════════════════════════════════════════════

FAITHFULNESS_PROMPT = """You are a strict fact-checker. Your job is to evaluate whether an AI-generated answer is faithful to the provided context.

Evaluate on these criteria:
1. Every factual claim in the answer must be supported by the context.
2. No information should be invented or hallucinated.
3. Numbers, timeframes, and specific details must match exactly.
4. The answer must not contradict the context.

Score from 0.0 to 1.0:
- 1.0 = fully faithful, every claim supported
- 0.7-0.9 = mostly faithful, minor wording differences
- 0.4-0.6 = partially faithful, some unsupported claims
- 0.0-0.3 = mostly hallucinated or contradicts context

Return ONLY a JSON object with this exact shape:
{{"score": <float between 0.0 and 1.0>, "reason": "<one sentence explanation>"}}

Do NOT include any text before or after the JSON.

Context:
{context}

Answer to evaluate:
{answer}

JSON:"""


# ═══════════════════════════════════════════════════════════════
# SAFETY / DISTRESS PROMPT
# ═══════════════════════════════════════════════════════════════

DISTRESS_RESPONSE = """I hear that you may be going through a difficult time, and I want to make sure you get the right support.

Please reach out to one of these resources:

🆘 **Betsson Responsible Gaming Team** — Available 24/7 through the app or at responsiblegaming@betsson.com

🆘 **Gambling Therapy** — Free, confidential support: https://www.gamblingtherapy.org

🆘 **BeGambleAware** — https://www.begambleaware.org

If you are in immediate crisis, please contact your local emergency services.

You are not alone, and help is available. Would you like me to help you set a self-exclusion or deposit limit right now?"""


# ═══════════════════════════════════════════════════════════════
# FALLBACK RESPONSE (low faithfulness)
# ═══════════════════════════════════════════════════════════════

FALLBACK_RESPONSE = """I want to make sure I give you accurate information, but I'm not confident in the answer I generated.

Could you rephrase your question, or contact our support team directly through the app for a precise answer?"""


# ═══════════════════════════════════════════════════════════════
# CHITCHAT RESPONSE
# ═══════════════════════════════════════════════════════════════

CHITCHAT_RESPONSE = """Hello! I'm the Betsson AI assistant. I can only help with questions about Betsson, such as:
- Deposit limits and self-exclusion
- Bonus wagering requirements
- Withdrawals and payment policies
- Responsible gaming and player safety

If your question isn't about Betsson, I'm not able to help with that. What would you like to know about your Betsson account?"""
