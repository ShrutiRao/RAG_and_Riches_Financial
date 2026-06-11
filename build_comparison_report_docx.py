from __future__ import annotations

from datetime import date
from pathlib import Path
from textwrap import dedent
from zipfile import ZipFile, ZIP_DEFLATED
from xml.sax.saxutils import escape


OUT_PATH = Path("docs/comparison_report_rerank_impact.docx")

NS_W = "http://schemas.openxmlformats.org/wordprocessingml/2006/main"
NS_R = "http://schemas.openxmlformats.org/officeDocument/2006/relationships"
NS_CP = "http://schemas.openxmlformats.org/package/2006/metadata/core-properties"
NS_DC = "http://purl.org/dc/elements/1.1/"
NS_DCTERMS = "http://purl.org/dc/terms/"
NS_DCMITYPE = "http://purl.org/dc/dcmitype/"
NS_XSI = "http://www.w3.org/2001/XMLSchema-instance"
NS_VT = "http://schemas.openxmlformats.org/officeDocument/2006/docPropsVTypes"


def twips(inches: float) -> int:
    return int(round(inches * 1440))


def pt_to_half_points(points: float) -> int:
    return int(round(points * 2))


def xml_attr(name: str, value: str | int) -> str:
    return f'{name}="{escape(str(value))}"'


def run_xml(text: str, *, size: float = 11, bold: bool = False, italic: bool = False, color: str = "000000") -> str:
    parts = [
        "<w:r>",
        "<w:rPr>",
        f'<w:rFonts w:ascii="Calibri" w:hAnsi="Calibri"/>',
        f'<w:sz w:val="{pt_to_half_points(size)}"/>',
        f'<w:color w:val="{color}"/>',
    ]
    if bold:
        parts.append("<w:b/>")
    if italic:
        parts.append("<w:i/>")
    parts.extend(
        [
            "</w:rPr>",
            f"<w:t xml:space=\"preserve\">{escape(text)}</w:t>",
            "</w:r>",
        ]
    )
    return "".join(parts)


def paragraph_xml(
    runs: list[str],
    *,
    align: str | None = None,
    before: int = 0,
    after: int = 120,
    line: int = 260,
    keep_with_next: bool = False,
    shading: str | None = None,
    border_bottom: str | None = None,
) -> str:
    ppr = [
        "<w:pPr>",
        f'<w:spacing w:before="{before}" w:after="{after}" w:line="{line}" w:lineRule="auto"/>',
    ]
    if align:
        ppr.append(f'<w:jc w:val="{align}"/>')
    if keep_with_next:
        ppr.append("<w:keepNext/>")
    if shading:
        ppr.append(
            f'<w:shd w:val="clear" w:color="auto" w:fill="{shading}"/>'
        )
    if border_bottom:
        ppr.append(
            f'<w:pBdr><w:bottom w:val="single" w:sz="6" w:space="1" w:color="{border_bottom}"/></w:pBdr>'
        )
    ppr.append("</w:pPr>")
    return "<w:p>" + "".join(ppr) + "".join(runs) + "</w:p>"


def heading_xml(text: str, level: int) -> str:
    if level == 1:
        return paragraph_xml(
            [run_xml(text, size=16, bold=True, color="1F4D78")],
            before=320,
            after=100,
            line=280,
            keep_with_next=True,
        )
    if level == 2:
        return paragraph_xml(
            [run_xml(text, size=13, bold=True, color="2E74B5")],
            before=180,
            after=60,
            line=260,
            keep_with_next=True,
        )
    return paragraph_xml(
        [run_xml(text, size=12, bold=True, color="1F4D78")],
        before=120,
        after=40,
        line=250,
        keep_with_next=True,
    )


def cell_xml(
    text: str,
    *,
    width: int,
    bold: bool = False,
    fill: str | None = None,
    align: str = "left",
    color: str = "000000",
) -> str:
    tcpr = [
        "<w:tcPr>",
        f'<w:tcW w:type="dxa" w:w="{width}"/>',
        '<w:tcMar w:top="80" w:start="120" w:bottom="80" w:end="120"/>',
        "<w:vAlign w:val=\"center\"/>",
    ]
    if fill:
        tcpr.append(f'<w:shd w:val="clear" w:color="auto" w:fill="{fill}"/>')
    tcpr.append("</w:tcPr>")
    return (
        "<w:tc>"
        + "".join(tcpr)
        + paragraph_xml(
            [run_xml(text, size=10.5 if bold else 10.5, bold=bold, color=color)],
            align=align,
            before=0,
            after=60,
            line=240,
        )
        + "</w:tc>"
    )


def table_xml(headers: list[str], rows: list[list[str]], widths: list[int]) -> str:
    grid = "".join(f'<w:gridCol w:w="{w}"/>' for w in widths)
    header_fill = "E8EEF5"
    border_xml = dedent(
        f"""
        <w:tblBorders>
          <w:top w:val="single" w:sz="6" w:space="0" w:color="D0D7DE"/>
          <w:left w:val="single" w:sz="6" w:space="0" w:color="D0D7DE"/>
          <w:bottom w:val="single" w:sz="6" w:space="0" w:color="D0D7DE"/>
          <w:right w:val="single" w:sz="6" w:space="0" w:color="D0D7DE"/>
          <w:insideH w:val="single" w:sz="6" w:space="0" w:color="D0D7DE"/>
          <w:insideV w:val="single" w:sz="6" w:space="0" w:color="D0D7DE"/>
        </w:tblBorders>
        """
    ).strip()
    tbl_pr = dedent(
        f"""
        <w:tblPr>
          <w:tblW w:type="dxa" w:w="9360"/>
          <w:tblInd w:type="dxa" w:w="120"/>
          <w:tblLayout w:type="fixed"/>
          {border_xml}
          <w:tblCellMar>
            <w:top w:w="80" w:type="dxa"/>
            <w:start w:w="120" w:type="dxa"/>
            <w:bottom w:w="80" w:type="dxa"/>
            <w:end w:w="120" w:type="dxa"/>
          </w:tblCellMar>
        </w:tblPr>
        """
    ).strip()

    header_row = "<w:tr>" + "".join(
        cell_xml(head, width=width, bold=True, fill=header_fill, align="center", color="1F1F1F")
        for head, width in zip(headers, widths, strict=True)
    ) + "</w:tr>"

    body_rows = []
    for row in rows:
        body_rows.append(
            "<w:tr>"
            + "".join(
                cell_xml(text, width=width, bold=False, align="left")
                for text, width in zip(row, widths, strict=True)
            )
            + "</w:tr>"
        )

    return "<w:tbl>" + tbl_pr + f"<w:tblGrid>{grid}</w:tblGrid>" + header_row + "".join(body_rows) + "</w:tbl>"


def callout_xml(text: str) -> str:
    return (
        '<w:tbl>'
        '<w:tblPr>'
        '<w:tblW w:type="dxa" w:w="9360"/>'
        '<w:tblInd w:type="dxa" w:w="120"/>'
        '<w:tblLayout w:type="fixed"/>'
        '<w:tblBorders>'
        '<w:top w:val="single" w:sz="6" w:space="0" w:color="D0D7DE"/>'
        '<w:left w:val="single" w:sz="6" w:space="0" w:color="D0D7DE"/>'
        '<w:bottom w:val="single" w:sz="6" w:space="0" w:color="D0D7DE"/>'
        '<w:right w:val="single" w:sz="6" w:space="0" w:color="D0D7DE"/>'
        '</w:tblBorders>'
        '<w:tblCellMar>'
        '<w:top w:w="120" w:type="dxa"/>'
        '<w:start w:w="140" w:type="dxa"/>'
        '<w:bottom w:w="120" w:type="dxa"/>'
        '<w:end w:w="140" w:type="dxa"/>'
        '</w:tblCellMar>'
        '</w:tblPr>'
        '<w:tblGrid><w:gridCol w:w="9360"/></w:tblGrid>'
        '<w:tr><w:tc>'
        '<w:tcPr>'
        '<w:tcW w:type="dxa" w:w="9360"/>'
        '<w:tcMar w:top="120" w:start="140" w:bottom="120" w:end="140"/>'
        '<w:vAlign w:val="center"/>'
        '<w:shd w:val="clear" w:color="auto" w:fill="F4F6F9"/>'
        '</w:tcPr>'
        + paragraph_xml([run_xml(text, size=11, bold=True, color="1F3A5F")], after=60, line=250)
        + '</w:tc></w:tr></w:tbl>'
    )


def document_xml() -> str:
    sections: list[str] = []
    sections.append(
        paragraph_xml(
            [run_xml("Comparison Report and Reranking Impact Analysis", size=26, color="0B2545", bold=True)],
            align="center",
            before=0,
            after=40,
            line=280,
        )
    )
    sections.append(
        paragraph_xml(
            [run_xml("A report on fixed-size, semantic, and hybrid retrieval quality across the financial corpus.", size=11, italic=True, color="5B6573")],
            align="center",
            before=0,
            after=80,
            line=250,
        )
    )
    sections.append(
        paragraph_xml(
            [run_xml("Prepared for benchmark review and retrieval quality analysis.", size=9.5, color="6B7280")],
            align="center",
            before=0,
            after=180,
            line=230,
        )
    )
    sections.append(
        callout_xml(
            "We used LlamaIndex to build 3 parallel RAG pipelines over SEC filings, earnings call transcripts, insurance claims, and loan agreements. The only difference between the pipelines was the chunking strategy - fixed-size versus semantic chunking - allowing us to directly measure the impact of chunking on retrieval quality, answer relevance, and faithfulness."
        )
    )

    sections.append(heading_xml("Executive Summary", 1))
    sections.append(
        paragraph_xml(
            [run_xml(
                "This benchmark isolates the retrieval layer so we can see how each design choice affects the quality of evidence returned to the answer generator. The report focuses on retrieval rank, first-hit quality, reranking lift, and faithfulness, which together tell the story of how well the app stays anchored to the source material.",
                size=11,
            )],
            after=80,
            line=250,
        )
    )
    sections.append(
        paragraph_xml(
            [run_xml(
                "The practical question is simple: when a financial chatbot needs to explain liquidity pressure, reserve movement, covenant terms, or claim activity, which retrieval path gives the clearest and most defensible evidence?",
                size=11,
            )],
            after=80,
            line=250,
        )
    )

    sections.append(heading_xml("Comparison Design", 1))
    sections.append(heading_xml("Fixed-size chunking", 2))
    sections.append(
        paragraph_xml(
            [run_xml(
                "Documents are split into overlapping word windows. This is the most mechanical baseline and is useful for broad keyword retrieval, but it can break sentence-level meaning when the answer depends on a precise clause, claim, or numeric detail.",
                size=11,
            )],
            after=60,
            line=250,
        )
    )
    sections.append(heading_xml("Semantic chunking", 2))
    sections.append(
        paragraph_xml(
            [run_xml(
                "Documents are split along sentence-aware boundaries so that each chunk better preserves a complete thought. This tends to improve retrieval precision when the query is looking for a statement, a risk disclosure, or a claim-specific detail.",
                size=11,
            )],
            after=60,
            line=250,
        )
    )
    sections.append(heading_xml("Hybrid retrieval with RRF", 2))
    sections.append(
        paragraph_xml(
            [run_xml(
                "Hybrid retrieval runs fixed-size and semantic retrieval in parallel, then merges their ranked results using Reciprocal Rank Fusion. This helps recover evidence that one strategy may miss while still promoting strong candidates from both retrieval paths.",
                size=11,
            )],
            after=100,
            line=250,
        )
    )

    sections.append(heading_xml("Tool Stack", 1))
    tool_rows = [
        ["LlamaIndex", "Orchestrates the RAG pipelines and keeps the comparison workflow aligned across strategies."],
        ["Pinecone", "Provides vector search for chunk retrieval, with a local fallback path when external services are unavailable."],
        ["Nebius", "Handles query rewrite and optional reranking so we can measure the impact of ordering improvements."],
        ["OpenAI embeddings", "Serve as one embedding baseline in the embedding comparison table."],
        ["BGE embeddings", "Serve as the second embedding baseline for side-by-side retrieval comparison."],
        ["LlamaParse", "Parses optional PDF sources when available, extending the corpus beyond the built-in sample set."],
        ["Streamlit", "Powers the interactive UI, comparison views, and benchmark tables."],
    ]
    sections.append(table_xml(["Tool", "Role in the report"], tool_rows, [2304, 7056]))

    sections.append(heading_xml("Reranking Impact", 1))
    sections.append(
        paragraph_xml(
            [run_xml(
                "Reranking does not replace retrieval. It reorders the retrieved candidates after the first pass so the most relevant passages rise to the top. In practice, this improves the usefulness of the context presented to the answer generator, especially when the first retrieval pass returns several plausible but uneven candidates.",
                size=11,
            )],
            after=60,
            line=250,
        )
    )
    sections.append(
        paragraph_xml(
            [run_xml(
                "The impact should be interpreted as an ordering improvement rather than a recall improvement. If reranking lifts the first relevant source, the answer often becomes more focused and easier to defend with evidence. If reranking improves rank but the answer still wanders, the generation layer may need tightening.",
                size=11,
            )],
            after=80,
            line=250,
        )
    )

    sections.append(heading_xml("Faithfulness and Answer Quality", 1))
    sections.append(
        paragraph_xml(
            [run_xml(
                "Faithfulness is measured by checking whether the answer is supported by the retrieved chunks. In this app, claim IDs, dates, numbers, and source-specific statements should all be traceable back to evidence that appears in the retrieved context.",
                size=11,
            )],
            after=60,
            line=250,
        )
    )
    sections.append(
        paragraph_xml(
            [run_xml(
                "That makes the benchmark useful for more than retrieval tuning. It also highlights whether the answer generator stays grounded or starts stretching beyond the source text.",
                size=11,
            )],
            after=80,
            line=250,
        )
    )

    sections.append(heading_xml("Recommended Demo Questions", 1))
    question_rows = [
        ["Liquidity", "What liquidity risks and funding pressure are disclosed across the SEC filings and loan documents?"],
        ["Margins", "What do the earnings calls say about margin pressure, pricing, and guidance?"],
        ["Claims", "How are claim reserves, settlement timing, and litigation exposure changing?"],
        ["Covenants", "What covenant obligations and default triggers should we watch in the loan documents?"],
        ["Hybrid", "Which documents mention both liquidity pressure and compliance or control remediation?"],
        ["Exact ID", "What happened with claim CLM-2026-1048?"],
        ["Exact ID", "What is the status of claim CLM-2026-1057?"],
    ]
    sections.append(table_xml(["Demo theme", "Question"], question_rows, [1500, 7860]))

    sections.append(heading_xml("Final reminder of refusal rule", 1))
    sections.append(
        callout_xml(
            "The assistant must stay grounded in the provided financial documents. If a question is not supported by the corpus, the answer should refuse briefly rather than inventing details."
        )
    )

    body = "".join(sections) + (
        "<w:sectPr>"
        "<w:pgSz w:w=\"12240\" w:h=\"15840\"/>"
        "<w:pgMar w:top=\"1440\" w:right=\"1440\" w:bottom=\"1440\" w:left=\"1440\" w:header=\"708\" w:footer=\"708\" w:gutter=\"0\"/>"
        "<w:cols w:space=\"708\"/>"
        "<w:docGrid w:linePitch=\"360\"/>"
        "</w:sectPr>"
    )
    return (
        f'<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
        f'<w:document xmlns:w="{NS_W}" xmlns:r="{NS_R}">'
        f"<w:body>{body}</w:body>"
        f"</w:document>"
    )


def styles_xml() -> str:
    return dedent(
        f"""<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<w:styles xmlns:w="{NS_W}">
  <w:docDefaults>
    <w:rPrDefault>
      <w:rPr>
        <w:rFonts w:ascii="Calibri" w:hAnsi="Calibri"/>
        <w:sz w:val="22"/>
        <w:szCs w:val="22"/>
      </w:rPr>
    </w:rPrDefault>
    <w:pPrDefault>
      <w:pPr>
        <w:spacing w:before="0" w:after="120" w:line="260" w:lineRule="auto"/>
      </w:pPr>
    </w:pPrDefault>
  </w:docDefaults>
  <w:style w:type="paragraph" w:default="1" w:styleId="Normal">
    <w:name w:val="Normal"/>
    <w:qFormat/>
    <w:rPr>
      <w:rFonts w:ascii="Calibri" w:hAnsi="Calibri"/>
      <w:sz w:val="22"/>
      <w:szCs w:val="22"/>
      <w:color w:val="000000"/>
    </w:rPr>
    <w:pPr>
      <w:spacing w:before="0" w:after="120" w:line="260" w:lineRule="auto"/>
    </w:pPr>
  </w:style>
</w:styles>
"""
    ).strip()


def content_types_xml() -> str:
    return dedent(
        f"""<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<Types xmlns="http://schemas.openxmlformats.org/package/2006/content-types">
  <Default Extension="rels" ContentType="application/vnd.openxmlformats-package.relationships+xml"/>
  <Default Extension="xml" ContentType="application/xml"/>
  <Override PartName="/word/document.xml" ContentType="application/vnd.openxmlformats-officedocument.wordprocessingml.document.main+xml"/>
  <Override PartName="/word/styles.xml" ContentType="application/vnd.openxmlformats-officedocument.wordprocessingml.styles+xml"/>
  <Override PartName="/docProps/core.xml" ContentType="application/vnd.openxmlformats-package.core-properties+xml"/>
  <Override PartName="/docProps/app.xml" ContentType="application/vnd.openxmlformats-officedocument.extended-properties+xml"/>
</Types>
"""
    ).strip()


def rels_xml() -> str:
    return dedent(
        f"""<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">
  <Relationship Id="rId1" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/officeDocument" Target="word/document.xml"/>
  <Relationship Id="rId2" Type="http://schemas.openxmlformats.org/package/2006/relationships/metadata/core-properties" Target="docProps/core.xml"/>
  <Relationship Id="rId3" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/extended-properties" Target="docProps/app.xml"/>
</Relationships>
"""
    ).strip()


def app_xml() -> str:
    return dedent(
        """<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<Properties xmlns="http://schemas.openxmlformats.org/officeDocument/2006/extended-properties" xmlns:vt="http://schemas.openxmlformats.org/officeDocument/2006/docPropsVTypes">
  <Application>OpenAI Codex</Application>
</Properties>
"""
    ).strip()


def core_xml() -> str:
    now = date(2026, 6, 10).isoformat() + "T00:00:00Z"
    return dedent(
        f"""<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<cp:coreProperties xmlns:cp="{NS_CP}" xmlns:dc="{NS_DC}" xmlns:dcterms="{NS_DCTERMS}" xmlns:dcmitype="{NS_DCMITYPE}" xmlns:xsi="{NS_XSI}">
  <dc:title>Comparison Report and Reranking Impact Analysis</dc:title>
  <dc:subject>RAG benchmark report</dc:subject>
  <dc:creator>OpenAI Codex</dc:creator>
  <cp:keywords>RAG, benchmark, reranking, faithfulness</cp:keywords>
  <dc:description>Comparison report and reranking impact analysis for the financial RAG app.</dc:description>
  <cp:lastModifiedBy>OpenAI Codex</cp:lastModifiedBy>
  <dcterms:created xsi:type="dcterms:W3CDTF">{now}</dcterms:created>
  <dcterms:modified xsi:type="dcterms:W3CDTF">{now}</dcterms:modified>
</cp:coreProperties>
"""
    ).strip()


def make_docx(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with ZipFile(path, "w", ZIP_DEFLATED) as zf:
        zf.writestr("[Content_Types].xml", content_types_xml())
        zf.writestr("_rels/.rels", rels_xml())
        zf.writestr("docProps/core.xml", core_xml())
        zf.writestr("docProps/app.xml", app_xml())
        zf.writestr("word/document.xml", document_xml())
        zf.writestr("word/styles.xml", styles_xml())


if __name__ == "__main__":
    make_docx(OUT_PATH)
    print(OUT_PATH)
