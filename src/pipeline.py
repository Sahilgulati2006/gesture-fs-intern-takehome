import argparse
import os
import sys
from typing import Callable, Dict, Any, List

from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from src.knowledge_base import build_knowledge_base


# ──────────────────────────────────────────────
# Provided: local LLM (no API key needed)
# ──────────────────────────────────────────────
def get_llm() -> Callable[[str], List[Dict[str, str]]]:
    """Return a callable local LLM using flan-t5-base.

    Downloads ~1GB on first run, then cached.
    """
    tokenizer = AutoTokenizer.from_pretrained("google/flan-t5-base")
    model = AutoModelForSeq2SeqLM.from_pretrained("google/flan-t5-base")

    def generate(prompt: str) -> List[Dict[str, str]]:
        inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=512)
        outputs = model.generate(**inputs, max_new_tokens=150)
        text = tokenizer.decode(outputs[0], skip_special_tokens=True)
        return [{"generated_text": text}]

    return generate


# ──────────────────────────────────────────────
# Provided: prompt template
# ──────────────────────────────────────────────
PROMPT_TEMPLATE = """You are a helpful assistant for a marketing agency. Use the following context to answer the client's question.
If the answer is not in the context, say "I don't have enough information to answer that."

Context:
{context}

Client question: {question}

Answer:"""


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# TODO 1: Implement ask_question
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
def ask_question(vector_store, llm: Callable[[str], List[Dict[str, str]]], question: str) -> Dict[str, Any]:
    """Retrieve relevant chunks and generate an answer."""
    if not question or not question.strip():
        raise ValueError("Question cannot be empty.")

    # 1. Retrieve the top 3 most relevant chunks
    docs = vector_store.similarity_search(question, k=3)
    sources = [doc.page_content for doc in docs]

    # 2. Combine chunk text into a single context string
    context = "\n\n".join(sources)

    # CRITICAL FIX: flan-t5-base restricts inputs to 512 tokens. 
    # If the context is too long, the tokenizer chops off the *end* of the prompt,
    # meaning the LLM never sees the actual question! We safely truncate the context.
    # ~300 tokens * 4 chars/token = 1200 character limit for context.
    if len(context) > 1200:
        context = context[:1200] + "..."

    # 3. Fill in the prompt template
    prompt = PROMPT_TEMPLATE.format(context=context, question=question.strip())

    # 4. Call the LLM and extract the generated text
    result = llm(prompt)
    answer = result[0]["generated_text"]

    return {"answer": answer, "sources": sources}


# ──────────────────────────────────────────────
# Helper: pretty-print a result in the CLI
# ──────────────────────────────────────────────
def print_result(result: Dict[str, Any]) -> None:
    print("\n📄 Sources:")
    for i, source in enumerate(result["sources"], start=1):
        preview = source.strip().replace("\n", " ")
        if len(preview) > 100:
            preview = preview[:100] + "..."
        print(f"  {i}. {preview}")
    print(f"\n💬 Answer: {result['answer']}\n")
    print("-" * 50)


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# TODO 2: Complete the interactive loop
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
def main() -> None:
    """Interactive Q&A loop with Bonus features."""
    parser = argparse.ArgumentParser(
        description="Ask questions about the marketing agency's services, pricing, and process."
    )
    # BONUS: --query CLI argument for single-question mode
    parser.add_argument(
        "--query",
        type=str,
        default=None,
        help="Ask a single question and print the answer, then exit.",
    )
    args = parser.parse_args()

    # Robust path construction ensuring it always points to the right dir
    base_dir = os.path.dirname(os.path.abspath(__file__))
    data_dir = os.path.join(base_dir, "..", "data")

    # BONUS: Missing files / directory error handling
    if not os.path.isdir(data_dir):
        print(f"❌ Error: Data directory not found at {data_dir}", file=sys.stderr)
        sys.exit(1)

    print("📚 Building knowledge base...")
    try:
        vector_store = build_knowledge_base(data_dir)
    except Exception as e:
        print(f"❌ Failed to build knowledge base: {e}", file=sys.stderr)
        sys.exit(1)

    print("🤖 Loading LLM (this may take a moment on first run)...")
    try:
        llm = get_llm()
    except Exception as e:
        print(f"❌ Failed to load LLM: {e}", file=sys.stderr)
        sys.exit(1)

    # Handle Single-Question Mode (--query)
    if args.query:
        try:
            result = ask_question(vector_store, llm, args.query)
            print_result(result)
        except Exception as e:
            print(f"❌ Error processing query: {e}", file=sys.stderr)
            sys.exit(1)
        return

    # Handle Interactive Mode
    print("\n✨ Ready! Ask a question about our services, pricing, or process.")
    print("Type 'quit' or 'exit' to stop.\n")

    while True:
        try:
            question = input("> ").strip()
        except (EOFError, KeyboardInterrupt):  # Graceful exit on Ctrl+C / Ctrl+D
            print("\nGoodbye!")
            break

        if question.lower() in ("quit", "exit"):
            print("Goodbye!")
            break

        # BONUS: Empty input error handling
        if not question:
            print("⚠️ Please enter a question.\n")
            continue

        try:
            result = ask_question(vector_store, llm, question)
            print_result(result)
        except Exception as e:
            print(f"❌ Something went wrong: {e}\n")


if __name__ == "__main__":
    main()