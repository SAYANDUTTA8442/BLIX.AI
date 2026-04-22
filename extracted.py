import pdfplumber
import re
import textwrap
from datetime import datetime
from typing import Optional


class PDFExtractor:
    def __init__(self, file_path: str, line_width: int = 80):
        self.file_path = file_path
        self.line_width = line_width

    # =========================
    # CORE EXTRACTION
    # =========================
    def extract_text(self) -> Optional[list[dict]]:
        """
        Extract text page-wise from PDF.
        Returns list of dicts: {page_num, text, word_count}
        """
        pages = []
        try:
            with pdfplumber.open(self.file_path) as pdf:
                self.total_pages = len(pdf.pages)
                self.metadata   = pdf.metadata or {}
                for i, page in enumerate(pdf.pages):
                    raw = page.extract_text() or ""
                    pages.append({
                        "page_num":   i + 1,
                        "text":       raw,
                        "word_count": len(raw.split()) if raw else 0,
                    })
        except Exception as e:
            print(f"[EXTRACTOR ERROR]: {e}")
            return None
        return pages

    # =========================
    # CLEANING
    # =========================
    def clean_text(self, text: str) -> str:
        """
        Clean extracted text:
        - Normalize whitespace & newlines
        - Fix hyphenated line-breaks
        - Strip non-ASCII artefacts
        - Wrap long lines to self.line_width
        """
        text = re.sub(r"\r", "\n", text)
        text = re.sub(r"-\n",  "",  text)          # re-join hyphenated words
        text = re.sub(r"\n{3,}", "\n\n", text)     # collapse triple+ blank lines
        text = re.sub(r"[ \t]+", " ",   text)      # collapse horizontal whitespace
        text = re.sub(r"[^\x00-\x7F]+", " ", text) # drop non-ASCII artefacts
        text = re.sub(r" +\n", "\n", text)         # trailing spaces before newline

        # Wrap long paragraphs
        paragraphs = text.split("\n\n")
        wrapped = []
        for para in paragraphs:
            para = para.strip()
            if not para:
                continue
            # Preserve lines that look like headings / short lines (< 60 chars)
            lines = para.splitlines()
            if len(lines) == 1 and len(para) < 60:
                wrapped.append(para)
            else:
                wrapped.append(textwrap.fill(para, width=self.line_width))
        return "\n\n".join(wrapped)

    # =========================
    # FORMATTING HELPERS
    # =========================
    def _divider(self, char: str = "=") -> str:
        return char * self.line_width

    def _center(self, text: str) -> str:
        return text.center(self.line_width)

    def _build_document_header(self) -> str:
        title   = self.metadata.get("Title",   "Untitled Document")
        author  = self.metadata.get("Author",  "Unknown")
        subject = self.metadata.get("Subject", "")
        created = self.metadata.get("CreationDate", "")
        today   = datetime.now().strftime("%Y-%m-%d %H:%M")

        lines = [
            self._divider("="),
            self._center("EXTRACTED DOCUMENT"),
            self._divider("="),
            f"  File   : {self.file_path}",
            f"  Title  : {title}",
            f"  Author : {author}",
        ]
        if subject:
            lines.append(f"  Subject: {subject}")
        if created:
            lines.append(f"  Created: {created}")
        lines += [
            f"  Pages  : {self.total_pages}",
            f"  Extracted on: {today}",
            self._divider("="),
            "",
        ]
        return "\n".join(lines)

    def _build_toc(self, pages: list[dict]) -> str:
        lines = [
            self._divider("-"),
            self._center("TABLE OF CONTENTS"),
            self._divider("-"),
        ]
        for p in pages:
            label = f"  Page {p['page_num']:>3}"
            words = f"({p['word_count']} words)"
            dots  = "." * max(4, self.line_width - len(label) - len(words) - 2)
            lines.append(f"{label} {dots} {words}")
        lines += [self._divider("-"), ""]
        return "\n".join(lines)

    def _build_page_block(self, page: dict, clean: str) -> str:
        pg  = page["page_num"]
        wc  = page["word_count"]
        bar = self._divider("-")
        header = (
            f"\n{bar}\n"
            f"  PAGE {pg} of {self.total_pages}  |  {wc} words\n"
            f"{bar}\n"
        )
        if not clean:
            body = "  [No extractable text on this page]\n"
        else:
            # Indent every line by 2 spaces for readability
            body = "\n".join("  " + ln for ln in clean.splitlines()) + "\n"
        footer = f"\n{'·' * self.line_width}\n"
        return header + body + footer

    def _build_document_footer(self, total_words: int) -> str:
        return (
            f"\n{self._divider('=')}\n"
            f"{self._center('END OF DOCUMENT')}\n"
            f"  Total pages : {self.total_pages}\n"
            f"  Total words : {total_words}\n"
            f"{self._divider('=')}\n"
        )

    # =========================
    # FULL PIPELINE
    # =========================
    def extract_and_format(self) -> Optional[str]:
        pages = self.extract_text()
        if not pages:
            return None

        sections = [self._build_document_header(), self._build_toc(pages)]

        total_words = 0
        for page in pages:
            clean = self.clean_text(page["text"]) if page["text"] else ""
            total_words += page["word_count"]
            sections.append(self._build_page_block(page, clean))

        sections.append(self._build_document_footer(total_words))
        return "\n".join(sections)


# =========================
# CLI / TEST RUN
# =========================
if __name__ == "__main__":
    input_file  = "input.pdf"
    output_file = "extracted.txt"

    extractor = PDFExtractor(input_file, line_width=88)
    formatted = extractor.extract_and_format()

    if formatted:
        with open(output_file, "w", encoding="utf-8") as f:
            f.write(formatted)
        print(f"✅ Extraction complete → {output_file}")
    else:
        print("❌ Extraction failed")