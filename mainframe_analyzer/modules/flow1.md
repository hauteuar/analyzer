🏗️ Architecture Diagram Components

The architecture is split into three layers: User, Agent, and Data.

1️⃣ User Layer

👤 Operations Team / FAs

The end users of the system.

They submit queries related to pricing or business logic.

Examples:

“What’s today’s price of Asset X?”

“Why does the system show a discrepancy in historical pricing?”

2️⃣ Agent Layer (Opulence AI Agent)

This is the intelligent core of the system. It has multiple modules:

🔍 Context-Aware Quest Module

First point of entry for any query.

Understands the intent and context of the user question.

Decides whether to check cache, use memory, or query external data.

📦 Cache Store

Fast-access memory for previously answered questions.

Avoids re-computation when the same or similar queries appear.

Improves speed and efficiency.

🧠 Chat Session Memory

Maintains conversation context for the current session.

Supports follow-up queries where users rely on past conversation history.

Example: “What about yesterday?” → Links back to the previous pricing query.

⚖️ LLM / SLM Selector

Smart decision-maker for model choice:

SLM (Small Language Model): Handles simple, structured queries.

LLM (Large Language Model): Handles complex, unstructured, or ambiguous queries.

Balances speed vs. accuracy vs. cost.

📚 Agentic RAG Retriever

Retrieves relevant context from external data sources.

Uses retrieval-augmented generation so the model answers based on facts, not just memory.

Chooses source depending on query type: Oracle DB or Document KB.

📝 Summarizer

Converts raw DB results or document extracts into concise business-friendly answers.

Ensures readability and avoids overwhelming users with raw technical data.

💬 Response Generator

Final layer that formats and delivers the response back to the user.

Also stores the response into cache for reuse in future queries.

3️⃣ Data Layer (External Sources)
🗄️ Oracle DB (Pricing Data)

Authoritative source of real pricing data.

Queried whenever users request prices, comparisons, or discrepancies.

Ensures accuracy and consistency with financial records.

📑 Document Knowledge Base (Business Logic Docs)

Repository of business rules, operational manuals, policies, and reference docs.

Queried whenever users ask “how” or “why” type questions about processes.

Works with RAG to fetch only the most relevant sections.

✅ Summary:

User Layer → submits queries.

Agent Layer → interprets, decides, retrieves, summarizes.

Data Layer → provides factual pricing + business logic.

Combined, this ensures fast, reliable, and context-aware responses for operations & FAs.


📝 How This  Works

User (Ops/FA) → submits query.

Context-Aware Quest module receives and routes it.

Cache check:

If answer exists → retrieve & summarize → return to user.

If not → check session memory for context.

LLM/SLM Selector → chooses the right model based on complexity.

Query Type Decision:

Pricing → Oracle DB

Business logic → Docs via RAG

RAG Retriever → pulls relevant context from external data.

Summarizer → cleans up raw results.

Response Generator → formats and delivers to user.

Answer is also stored in cache for reuse.