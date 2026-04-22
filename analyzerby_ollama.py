import re
import json
from collections import defaultdict
from OllamaLLM import OllamaLLM


class PageWiseAnalyzer:
    def __init__(self, model: str = "mistral"):
        self.llm = OllamaLLM(model=model)

    # =========================
    # LOAD FILE
    # =========================
    def load_text(self, file_path: str) -> str:
        with open(file_path, "r", encoding="utf-8") as f:
            return f.read()

    # =========================
    # SPLIT BY PAGE
    # =========================
    def split_pages(self, text: str) -> list[dict]:
        """
        Matches the actual format produced by PDFExtractor:

            ------------------------...
              PAGE X of Y  |  N words
            ------------------------...

        Two consecutive dash-divider lines sandwich the page label.
        Returns list of dicts: {page_num, content}
        """
        # Pattern: dashes-line, then "PAGE N" label line, then dashes-line
        pattern = re.compile(
            r"-{10,}\n"                    # opening divider  (---...---\n)
            r"\s+PAGE\s+(\d+)[^\n]*\n"     # "  PAGE X of Y  |  N words\n"
            r"-{10,}",                     # closing divider  (---...---)
            re.IGNORECASE
        )

        matches = list(pattern.finditer(text))

        if not matches:
            print("⚠️  No page headers found. Treating entire file as one block.")
            return [{"page_num": 1, "content": text.strip()}]

        pages = []
        for i, match in enumerate(matches):
            page_num = int(match.group(1))
            start    = match.end()          # content starts right after closing divider
            end      = matches[i + 1].start() if i + 1 < len(matches) else len(text)

            # Strip trailing dot-divider line  (·····) and any footer block
            content = text[start:end]
            content = re.sub(r"[·∙•]{5,}.*$", "", content, flags=re.DOTALL)
            content = content.strip()

            if len(content) > 30:           # skip near-empty pages
                pages.append({"page_num": page_num, "content": content})

        return pages

    # =========================
    # ANALYZE
    # =========================
    def analyze(self, text: str) -> dict:
        pages   = self.split_pages(text)
        results = []
        failed  = []

        print(f"📄 Total pages detected: {len(pages)}\n")

        for page in pages:
            pg = page["page_num"]
            try:
                print(f"🔍 Processing Page {pg}...")
                prompt   = self._build_prompt(page["content"])
                response = self.llm.generate(prompt)
                parsed   = self._safe_json(response)

                if parsed:
                    parsed["page"] = pg
                    results.append(parsed)
                else:
                    failed.append(pg)

            except Exception as e:
                print(f"  [ERROR Page {pg}]: {e}")
                failed.append(pg)

        if failed:
            print(f"\n⚠️  Pages with parse errors: {failed}")

        return self._merge(results)

    # =========================
    # PROMPT
    # =========================
    def _build_prompt(self, page_text: str) -> str:
        snippet = page_text[:3000]
        return f"""
You are a document analysis assistant.

Read the text below and extract:
1. The main academic/professional subject.
2. All topics covered, each with a difficulty rating: easy | medium | hard.

Rules:
- Return ONLY a valid JSON object — no prose, no markdown, no explanation.
- Use exactly this schema:

{{
  "subject": "<subject name>",
  "topics": [
    {{"name": "<topic>", "difficulty": "easy|medium|hard"}}
  ]
}}

Text:
\"\"\"
{snippet}
\"\"\"
"""

    # =========================
    # SAFE JSON PARSER
    # =========================
    def _safe_json(self, text: str) -> dict | None:
        text = re.sub(r"```(?:json)?", "", text).strip()
        try:
            start = text.index("{")
            end   = text.rindex("}") + 1
            return json.loads(text[start:end])
        except (ValueError, json.JSONDecodeError):
            print("  ⚠️  JSON parse failed")
            return None

    # =========================
    # MERGE + DEDUP
    # =========================
    def _merge(self, results: list[dict]) -> dict:
        DIFFICULTY_RANK = {"easy": 1, "medium": 2, "hard": 3}
        staging: dict[str, dict[str, str]] = defaultdict(dict)

        for r in results:
            subject = (r.get("subject") or "Unknown").strip().title()
            for t in r.get("topics", []):
                name = (t.get("name") or "").strip().title()
                diff = (t.get("difficulty") or "medium").strip().lower()
                if not name:
                    continue
                existing = staging[subject].get(name)
                if not existing or DIFFICULTY_RANK.get(diff, 0) > DIFFICULTY_RANK.get(existing, 0):
                    staging[subject][name] = diff

        merged = {}
        for subject, topics in staging.items():
            sorted_topics = sorted(
                [{"name": k, "difficulty": v} for k, v in topics.items()],
                key=lambda x: (-DIFFICULTY_RANK.get(x["difficulty"], 0), x["name"])
            )
            merged[subject] = {
                "topic_count": len(sorted_topics),
                "topics":      sorted_topics,
            }

        return merged

    # =========================
    # SAVE OUTPUT
    # =========================
    def save(self, result: dict, output_path: str = "analysis.json") -> None:
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(result, f, indent=2, ensure_ascii=False)
        print(f"\n💾 Results saved → {output_path}")


# =========================
# RUN
# =========================
if __name__ == "__main__":
    INPUT_FILE  = "extracted.txt"
    OUTPUT_FILE = "analysis.json"

    analyzer = PageWiseAnalyzer(model="mistral")
    text     = analyzer.load_text(INPUT_FILE)
    result   = analyzer.analyze(text)

    print("\n📊 FINAL OUTPUT:\n")
    print(json.dumps(result, indent=2, ensure_ascii=False))

    analyzer.save(result, OUTPUT_FILE)