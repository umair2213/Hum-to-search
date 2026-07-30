"""
Generate a professional PDF from the Markdown project report.

Usage:
    python docs/generate_pdf.py

Produces: docs/Hum_to_Search_Project_Report.pdf
"""

import re
import html
from pathlib import Path

import markdown
from weasyprint import HTML

BASE_DIR = Path(__file__).parent
MD_FILE = BASE_DIR / "Hum_to_Search_Project_Report.md"
PDF_FILE = BASE_DIR / "Hum_to_Search_Project_Report.pdf"
COVER_IMAGE = BASE_DIR / "ChatGPT Image Jul 30, 2026, 02_41_08 PM.png"

AUTHOR_NAME = "Umair Ahmed"
REPO_URL = "https://github.com/umair2213/Hum-to-search"


# ── Mermaid → HTML flowchart replacement ──────────────────────────

def _build_flowchart_box(label: str) -> str:
    return f'<div class="flow-box">{html.escape(label)}</div>'


def _build_flowchart_arrow() -> str:
    return '<div class="flow-arrow">&darr;</div>'


def _parse_mermaid_to_html(mermaid_block: str) -> str:
    """
    Parse simple mermaid flowchart TD blocks into HTML/CSS vertical flowchart.
    Handles: A[Label] --> B[Label]  and  A[Label<br/>subtitle] --> B[...]
    """
    lines = mermaid_block.strip().split('\n')
    if not lines or not lines[0].strip().startswith('flowchart'):
        return f'<pre>{html.escape(mermaid_block)}</pre>'

    # Extract nodes and edges
    edges = []
    node_labels = {}

    edge_pattern = re.compile(
        r'(\w+)\s*\[([^\]]+)\]\s*-->\s*(\w+)\s*\[([^\]]+)\]'
    )
    # Also handle nodes that appear only on one side
    single_pattern = re.compile(r'(\w+)\s*\[([^\]]+)\]')

    for line in lines[1:]:
        line = line.strip()
        if not line:
            continue

        match = edge_pattern.search(line)
        if match:
            src_id, src_label, dst_id, dst_label = match.groups()
            src_label = src_label.replace('<br/>', '<br>')
            dst_label = dst_label.replace('<br/>', '<br>')
            node_labels[src_id] = src_label
            node_labels[dst_id] = dst_label
            edges.append((src_id, dst_id))
        else:
            for sm in single_pattern.finditer(line):
                nid, nlabel = sm.groups()
                nlabel = nlabel.replace('<br/>', '<br>')
                node_labels[nid] = nlabel

    if not edges:
        return f'<pre>{html.escape(mermaid_block)}</pre>'

    # Build ordered sequence from edges (follow the chain)
    # Find start node (appears as src but never as dst)
    all_src = {e[0] for e in edges}
    all_dst = {e[1] for e in edges}
    start_nodes = all_src - all_dst
    start = next(iter(start_nodes)) if start_nodes else edges[0][0]

    # Build adjacency list
    adj = {}
    for src, dst in edges:
        adj.setdefault(src, []).append(dst)

    # Walk the chain
    ordered = []
    visited = set()
    current = start
    while current and current not in visited:
        visited.add(current)
        if current in node_labels:
            ordered.append(current)
        nexts = adj.get(current, [])
        current = nexts[0] if nexts else None

    # Build HTML
    parts = ['<div class="flowchart">']
    for i, nid in enumerate(ordered):
        if i > 0:
            parts.append(_build_flowchart_arrow())
        parts.append(_build_flowchart_box(node_labels[nid]))
    parts.append('</div>')

    return '\n'.join(parts)


def replace_mermaid_blocks(md_text: str) -> str:
    """Replace all ```mermaid ... ``` blocks with HTML flowcharts."""
    pattern = re.compile(r'```mermaid\n(.*?)```', re.DOTALL)
    return pattern.sub(lambda m: _parse_mermaid_to_html(m.group(1)), md_text)


# ── Table of Contents generation ──────────────────────────────────

def generate_toc(md_text: str) -> str:
    """Extract ## headings and build a TOC."""
    toc_items = []
    for match in re.finditer(r'^##\s+(\d+)\.\s+(.+)$', md_text, re.MULTILINE):
        num = match.group(1)
        title = match.group(2).strip()
        toc_items.append((num, title))

    if not toc_items:
        return ""

    items_html = []
    for num, title in toc_items:
        items_html.append(
            f'<div class="toc-item">'
            f'<span class="toc-num">{num}.</span>'
            f'<span class="toc-title">{html.escape(title)}</span>'
            f'</div>'
        )

    return (
        '<div class="toc-page">'
        '<h1 class="toc-heading">Table of Contents</h1>'
        + '\n'.join(items_html)
        + '</div>'
    )


# ── Main conversion ───────────────────────────────────────────────

def convert_md_to_pdf():
    md_text = MD_FILE.read_text(encoding="utf-8")

    # Replace mermaid blocks with HTML flowcharts
    md_text = replace_mermaid_blocks(md_text)

    # Convert markdown to HTML
    md = markdown.Markdown(extensions=['tables', 'fenced_code', 'attr_list'])
    body_html = md.convert(md_text)

    # Generate TOC
    toc_html = generate_toc(md_text)

    # Build cover page (full-bleed uploaded artwork)
    cover_image_uri = COVER_IMAGE.resolve().as_uri()
    cover_html = f"""
    <div class="cover-page">
        <img class="cover-image" src="{cover_image_uri}" alt="Hum-to-Search Cover">
    </div>
    """

    # Assemble full HTML document
    full_html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<style>
{CSS_STYLES}
</style>
</head>
<body>
{cover_html}
<div class="page-break"></div>
{toc_html}
<div class="page-break"></div>
<div class="report-body">
{body_html}
</div>
</body>
</html>"""

    # Generate PDF
    HTML(string=full_html, base_url=str(BASE_DIR)).write_pdf(str(PDF_FILE))
    print(f"PDF generated: {PDF_FILE}")


# ── CSS Styles ────────────────────────────────────────────────────

CSS_STYLES = """
/* ── Page setup ── */
@page {
    size: A4;
    margin: 2.2cm 2cm 2.2cm 2cm;
    @bottom-center {
        content: counter(page);
        font-family: 'Helvetica', 'Arial', sans-serif;
        font-size: 9pt;
        color: #888;
    }
}
@page :first {
    margin: 0;
    @bottom-center { content: none; }
}

/* ── Typography ── */
body {
    font-family: 'Helvetica', 'Arial', 'Noto Sans', sans-serif;
    font-size: 10.5pt;
    line-height: 1.6;
    color: #2c3e50;
}

/* ── Cover page ── */
.cover-page {
    page-break-after: always;
    width: 100%;
    height: 100vh;
    margin: 0;
    padding: 0;
    background: #0a0a14;
}
.cover-image {
    width: 100%;
    height: 100%;
    object-fit: cover;
    display: block;
}

/* ── Page break ── */
.page-break {
    page-break-before: always;
}

/* ── Table of Contents ── */
.toc-page {
    page-break-after: always;
}
.toc-heading {
    font-size: 22pt;
    font-weight: 700;
    color: #1a1a2e;
    border-bottom: 3px solid #e76f51;
    padding-bottom: 0.3cm;
    margin-bottom: 1cm;
}
.toc-item {
    display: flex;
    align-items: baseline;
    padding: 0.35cm 0;
    border-bottom: 1px dotted #ccc;
    font-size: 12pt;
}
.toc-num {
    font-weight: 700;
    color: #e76f51;
    width: 1.2cm;
    flex-shrink: 0;
}
.toc-title {
    color: #2c3e50;
}

/* ── Report body ── */
.report-body h1 {
    font-size: 18pt;
    font-weight: 700;
    color: #1a1a2e;
    border-bottom: 2px solid #e76f51;
    padding-bottom: 0.2cm;
    margin-top: 1.5cm;
    page-break-after: avoid;
}
.report-body h2 {
    font-size: 14pt;
    font-weight: 600;
    color: #16213e;
    margin-top: 1cm;
    page-break-after: avoid;
}
.report-body h3 {
    font-size: 12pt;
    font-weight: 600;
    color: #0f3460;
    margin-top: 0.8cm;
    page-break-after: avoid;
}
.report-body p {
    margin: 0.4cm 0;
    text-align: justify;
}
.report-body strong {
    color: #1a1a2e;
}
.report-body em {
    color: #0f3460;
}
.report-body code {
    font-family: 'Courier New', monospace;
    font-size: 9.5pt;
    background: #f0f0f0;
    padding: 1px 4px;
    border-radius: 3px;
    color: #c0392b;
}
.report-body pre {
    background: #f8f9fa;
    border: 1px solid #e0e0e0;
    border-radius: 4px;
    padding: 0.6cm;
    font-size: 9pt;
    overflow-x: auto;
    page-break-inside: avoid;
}
.report-body pre code {
    background: none;
    color: #2c3e50;
    padding: 0;
}

/* ── Tables ── */
.report-body table {
    width: 100%;
    border-collapse: collapse;
    margin: 0.6cm 0;
    font-size: 9.5pt;
    page-break-inside: auto;
}
.report-body thead {
    background: #1a1a2e;
    color: #ffffff;
}
.report-body th {
    padding: 0.3cm 0.4cm;
    text-align: left;
    font-weight: 600;
    border: 1px solid #1a1a2e;
}
.report-body td {
    padding: 0.25cm 0.4cm;
    border: 1px solid #ddd;
    vertical-align: top;
}
.report-body tbody tr:nth-child(even) {
    background: #f8f9fa;
}

/* ── Flowchart (mermaid replacement) ── */
.flowchart {
    margin: 0.8cm auto;
    text-align: center;
    page-break-inside: avoid;
}
.flow-box {
    display: inline-block;
    background: #eef2f8;
    border: 2px solid #0f3460;
    border-radius: 6px;
    padding: 0.3cm 0.8cm;
    font-size: 9.5pt;
    font-weight: 500;
    color: #1a1a2e;
    margin: 0.15cm 0;
    min-width: 5cm;
    max-width: 12cm;
}
.flow-arrow {
    font-size: 14pt;
    color: #e76f51;
    font-weight: 700;
    margin: 0.05cm 0;
}

/* ── Lists ── */
.report-body ul, .report-body ol {
    margin: 0.4cm 0;
    padding-left: 0.8cm;
}
.report-body li {
    margin: 0.2cm 0;
}

/* ── Links ── */
.report-body a {
    color: #0f3460;
    text-decoration: none;
}

/* ── Horizontal rule ── */
.report-body hr {
    border: none;
    border-top: 1px solid #e0e0e0;
    margin: 0.8cm 0;
}
"""


if __name__ == "__main__":
    convert_md_to_pdf()
