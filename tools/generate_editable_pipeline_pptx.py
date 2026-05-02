from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from zipfile import ZIP_DEFLATED, ZipFile

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'outputs' / 'modified_pipeline_editable.pptx'

SLIDE_CX = 12192000
SLIDE_CY = 6858000


def esc(s: str) -> str:
    return s.replace('&', '&amp;').replace('<', '&lt;').replace('>', '&gt;')


def emu(x_in: float) -> int:
    return int(round(x_in * 914400))


def rgb(hex_color: str) -> str:
    return hex_color.replace('#', '').upper()


BG = '#F5F8FC'
PANEL = '#EAF1FB'
WHITE = '#FFFFFF'
NAVY = '#17324D'
TEXT = '#1F2D3D'
SUBTEXT = '#5D7085'
BLUE = '#4F83FF'
BLUE_SOFT = '#DCE9FF'
ORANGE = '#F28C28'
GREEN = '#1E9B63'
GREEN_SOFT = '#E4F8EF'
RED = '#D64545'
RED_SOFT = '#FDE8E8'
GRAY = '#AAB6C3'
PROMPT = '#FFF7DB'
PROMPT_BORDER = '#EFC14D'
ARROW = '#7A93B8'
MASK = '#EEF3F8'


shape_id = 1


def next_id() -> int:
    global shape_id
    shape_id += 1
    return shape_id


def sp(shape_type: str, name: str, x: int, y: int, cx: int, cy: int, fill: str, line: str | None = None,
       line_w: int = 12700, radius_adj: int | None = None, text: str | None = None,
       font_size: int = 1800, bold: bool = False, color: str = TEXT, align: str = 'l',
       margin_l: int = 91440 // 4, margin_r: int = 91440 // 4, margin_t: int = 91440 // 6,
       margin_b: int = 91440 // 8, vert: str = 'horz', no_fill: bool = False) -> str:
    sid = next_id()
    fill_xml = '<a:noFill/>' if no_fill else f'<a:solidFill><a:srgbClr val="{rgb(fill)}"/></a:solidFill>'
    line_xml = '<a:ln><a:noFill/></a:ln>' if line is None else f'<a:ln w="{line_w}"><a:solidFill><a:srgbClr val="{rgb(line)}"/></a:solidFill></a:ln>'
    av = '' if radius_adj is None else f'<a:avLst><a:gd name="adj" fmla="val {radius_adj}"/></a:avLst>'
    tx_body = ''
    if text is not None:
        lines = text.split('\n')
        paras = []
        for i, line_text in enumerate(lines):
            paras.append(
                f'<a:p><a:pPr algn="{align}"/>'
                f'<a:r><a:rPr lang="en-US" sz="{font_size}" b="{1 if bold else 0}"><a:solidFill><a:srgbClr val="{rgb(color)}"/></a:solidFill></a:rPr><a:t>{esc(line_text)}</a:t></a:r>'
                f'<a:endParaRPr lang="en-US" sz="{font_size}" b="{1 if bold else 0}"><a:solidFill><a:srgbClr val="{rgb(color)}"/></a:solidFill></a:endParaRPr></a:p>'
            )
        tx_body = (
            '<p:txBody>'
            '<a:bodyPr wrap="square" rtlCol="0" anchor="ctr" vert="horz" lIns="%d" tIns="%d" rIns="%d" bIns="%d"/>'
            '<a:lstStyle/>'
            '%s'
            '</p:txBody>'
        ) % (margin_l, margin_t, margin_r, margin_b, ''.join(paras))
    return f'''
<p:sp>
  <p:nvSpPr>
    <p:cNvPr id="{sid}" name="{esc(name)}"/>
    <p:cNvSpPr/>
    <p:nvPr/>
  </p:nvSpPr>
  <p:spPr>
    <a:xfrm><a:off x="{x}" y="{y}"/><a:ext cx="{cx}" cy="{cy}"/></a:xfrm>
    <a:prstGeom prst="{shape_type}">{av}</a:prstGeom>
    {fill_xml}
    {line_xml}
  </p:spPr>
  {tx_body}
</p:sp>'''


def text_box(name: str, x: int, y: int, cx: int, cy: int, text: str, font_size: int = 1800,
             bold: bool = False, color: str = TEXT, align: str = 'l') -> str:
    return sp('rect', name, x, y, cx, cy, WHITE, None, text=text, font_size=font_size,
              bold=bold, color=color, align=align, no_fill=True)


def line_arrow(name: str, x1: int, y1: int, x2: int, y2: int, color: str = ARROW, width: int = 38100) -> str:
    sid = next_id()
    return f'''
<p:cxnSp>
  <p:nvCxnSpPr>
    <p:cNvPr id="{sid}" name="{esc(name)}"/>
    <p:cNvCxnSpPr/>
    <p:nvPr/>
  </p:nvCxnSpPr>
  <p:spPr>
    <a:xfrm><a:off x="{min(x1,x2)}" y="{min(y1,y2)}"/><a:ext cx="{abs(x2-x1)}" cy="{abs(y2-y1)}"/></a:xfrm>
    <a:prstGeom prst="line"><a:avLst/></a:prstGeom>
    <a:ln w="{width}" cmpd="sng" cap="rnd">
      <a:solidFill><a:srgbClr val="{rgb(color)}"/></a:solidFill>
      <a:tailEnd type="none"/>
      <a:headEnd type="triangle" w="med" len="med"/>
    </a:ln>
  </p:spPr>
</p:cxnSp>'''


def group_frame_triplet(x: int, y: int, title: str, note: str, border: str, dashed_inner: bool = False, w_each: int = emu(0.72), h_each: int = emu(0.58), gap: int = emu(0.10)) -> list[str]:
    items = []
    if title:
        items.append(text_box(title, x, y - emu(0.20), emu(2.0), emu(0.18), title, font_size=1600, bold=True, color=NAVY))
    for i, lbl in enumerate(['Frame 1', 'Frame 2', 'Frame 3']):
        fx = x + i * (w_each + gap)
        items.append(sp('roundRect', f'{title} frame {i+1}', fx, y, w_each, h_each, WHITE, border, line_w=19050, radius_adj=16667,
                        text=f'{lbl}\n(Add frame)', font_size=1400, color=SUBTEXT, align='ctr'))
        if dashed_inner:
            items.append(sp('roundRect', f'{title} inner {i+1}', fx + emu(0.10), y + emu(0.08), w_each - emu(0.20), h_each - emu(0.16),
                            MASK, GRAY, line_w=12700, radius_adj=16667, no_fill=False))
    items.append(text_box(f'{title} note', x, y + h_each + emu(0.04), emu(2.1), emu(0.28), note, font_size=1100, color=SUBTEXT))
    return items


def build_slide() -> str:
    items: list[str] = []

    items.append(sp('rect', 'Background', 0, 0, SLIDE_CX, SLIDE_CY, BG, None))
    items.append(text_box('Title', emu(0.45), emu(0.18), emu(8.0), emu(0.45),
                          'Foreground Motion Learning with Background Preservation', font_size=2500, bold=True, color=NAVY))
    items.append(text_box('Subtitle', emu(0.46), emu(0.58), emu(8.8), emu(0.22),
                          'Train foreground motion from a generated reference, then apply it on the original video to preserve the real background.',
                          font_size=1200, color=SUBTEXT))

    # Panels
    items.append(sp('roundRect', 'Panel A', emu(0.35), emu(0.95), emu(5.35), emu(2.55), PANEL, None, radius_adj=9000))
    items.append(sp('roundRect', 'Panel B', emu(5.85), emu(0.95), emu(5.75), emu(2.55), PANEL, None, radius_adj=9000))
    items.append(sp('roundRect', 'Panel C', emu(0.35), emu(3.95), emu(11.25), emu(2.95), PANEL, None, radius_adj=9000))

    items.append(text_box('A title', emu(0.52), emu(1.05), emu(3.6), emu(0.22), 'A. Motion Proposal from First Frame + Prompt', font_size=1800, bold=True, color=NAVY))
    items += group_frame_triplet(emu(0.55), emu(1.38), 'Input: Edited first frame', 'Insert your edited first-frame examples here.', BLUE)
    items.append(sp('roundRect', 'Prompt box A', emu(0.55), emu(2.68), emu(1.95), emu(0.50), PROMPT, PROMPT_BORDER, line_w=12700, radius_adj=16667,
                    text='Prompt\nExample: girl passes the potted plant to the person on her left.', font_size=1080, color='#6B5200'))
    items.append(line_arrow('Arrow A1', emu(2.65), emu(2.02), emu(2.95), emu(2.02)))
    items.append(sp('roundRect', 'WAN area', emu(3.02), emu(1.33), emu(2.35), emu(1.88), WHITE, ORANGE, line_w=19050, radius_adj=16667))
    items.append(text_box('WAN header', emu(3.16), emu(1.42), emu(1.8), emu(0.18), 'WAN / generative prior output', font_size=1450, bold=True, color=NAVY))
    items += group_frame_triplet(emu(3.16), emu(1.70), '', 'Desired handoff motion appears here. Background may drift from the source video.', ORANGE, dashed_inner=True, w_each=emu(0.52), h_each=emu(0.48), gap=emu(0.08))
    items.append(sp('roundRect', 'Motion ref chip', emu(3.16), emu(2.92), emu(1.95), emu(0.24), '#FFF0DD', '#F0B264', line_w=9525,
                    text='Use this clip only as a motion reference', font_size=1050, color='#8A4F00', align='ctr'))

    items.append(text_box('B title', emu(6.02), emu(1.05), emu(3.4), emu(0.22), 'B. Foreground Motion LoRA Training', font_size=1800, bold=True, color=NAVY))
    items += group_frame_triplet(emu(6.03), emu(1.44), 'Mask WAN video foreground', 'Mask only the moving foreground: girl + receiver + object.', BLUE, dashed_inner=True, w_each=emu(0.58), h_each=emu(0.50), gap=emu(0.08))
    items.append(line_arrow('Arrow B1', emu(8.38), emu(2.02), emu(8.72), emu(2.02)))
    items.append(sp('roundRect', 'Lora body train', emu(8.80), emu(1.50), emu(1.15), emu(0.86), WHITE, BLUE, line_w=19050, radius_adj=16667))
    items.append(sp('roundRect', 'Lora chip train', emu(9.02), emu(1.26), emu(0.66), emu(0.22), ORANGE, ORANGE, text='LoRA', font_size=1450, bold=True, color=WHITE, align='ctr'))
    items.append(sp('chevron', 'Left chevron train', emu(8.86), emu(1.67), emu(0.24), emu(0.40), BLUE_SOFT, BLUE_SOFT))
    items.append(sp('chevron', 'Right chevron train', emu(9.66), emu(1.67), emu(0.24), emu(0.40), BLUE_SOFT, BLUE_SOFT))
    items.append(sp('rect', 'Lora bar1 train', emu(9.10), emu(1.76), emu(0.08), emu(0.18), '#9CB7EE', None))
    items.append(sp('rect', 'Lora bar2 train', emu(9.24), emu(1.72), emu(0.08), emu(0.26), ORANGE, None))
    items.append(sp('rect', 'Lora bar3 train', emu(9.38), emu(1.76), emu(0.08), emu(0.18), '#9CB7EE', None))
    items.append(text_box('Lora train label', emu(8.62), emu(2.40), emu(1.55), emu(0.20), 'Learned representation', font_size=1450, bold=True, color=NAVY, align='ctr'))
    items.append(text_box('Lora train sub', emu(8.60), emu(2.57), emu(1.60), emu(0.15), 'LoRA learns foreground motion only', font_size=1000, color=SUBTEXT, align='ctr'))
    items.append(line_arrow('Arrow B2', emu(10.04), emu(2.02), emu(10.42), emu(2.02)))
    items.append(sp('roundRect', 'Training target', emu(10.47), emu(1.40), emu(0.98), emu(1.04), WHITE, GREEN, line_w=19050, radius_adj=16667,
                    text='Training Target\n\nForeground-motion\nrepresentation\nonly', font_size=1320, bold=True, color=NAVY, align='ctr'))
    items.append(sp('roundRect', 'Key change box', emu(6.03), emu(2.90), emu(5.45), emu(0.30), GREEN_SOFT, '#9FD9BF', line_w=9525,
                    text='Key change: learn foreground motion from the generated masked clip, not from the source-video foreground motion.',
                    font_size=980, color='#196847', align='ctr'))

    items.append(text_box('C title', emu(0.52), emu(4.05), emu(3.0), emu(0.22), 'C. Inference on the Original Video', font_size=1800, bold=True, color=NAVY))
    items += group_frame_triplet(emu(0.55), emu(4.42), 'Original video frames', 'Source video supplies the true background.', BLUE, w_each=emu(0.88), h_each=emu(0.62), gap=emu(0.10))
    items += group_frame_triplet(emu(0.55), emu(5.82), 'Same foreground mask on original video', 'Apply the same foreground mask used during training.', ORANGE, dashed_inner=True, w_each=emu(0.88), h_each=emu(0.44), gap=emu(0.10))
    items.append(sp('roundRect', 'Prompt box C', emu(3.28), emu(4.68), emu(1.62), emu(0.48), PROMPT, PROMPT_BORDER, line_w=12700, radius_adj=16667,
                    text='Prompt at inference\nReuse the handoff action prompt.', font_size=1080, color='#6B5200'))
    items.append(line_arrow('Arrow C1', emu(4.98), emu(4.92), emu(5.33), emu(4.92)))
    items.append(sp('roundRect', 'Lora body infer', emu(5.40), emu(4.42), emu(1.28), emu(0.94), WHITE, BLUE, line_w=19050, radius_adj=16667))
    items.append(sp('roundRect', 'Lora chip infer', emu(5.66), emu(4.22), emu(0.78), emu(0.24), BLUE, BLUE, text='LoRA', font_size=1450, bold=True, color=WHITE, align='ctr'))
    items.append(sp('chevron', 'Left chevron infer', emu(5.53), emu(4.74), emu(0.26), emu(0.42), BLUE_SOFT, BLUE_SOFT))
    items.append(sp('chevron', 'Right chevron infer', emu(6.33), emu(4.74), emu(0.26), emu(0.42), BLUE_SOFT, BLUE_SOFT))
    items.append(sp('rect', 'Lora bar1 infer', emu(5.83), emu(4.82), emu(0.08), emu(0.18), ORANGE, None))
    items.append(sp('rect', 'Lora bar2 infer', emu(5.99), emu(4.77), emu(0.08), emu(0.28), '#9CB7EE', None))
    items.append(sp('rect', 'Lora bar3 infer', emu(6.15), emu(4.82), emu(0.08), emu(0.18), ORANGE, None))
    items.append(text_box('Infer label', emu(5.25), emu(5.52), emu(1.60), emu(0.20), 'Inference module', font_size=1500, bold=True, color=NAVY, align='ctr'))
    items.append(text_box('Infer sub', emu(5.14), emu(5.70), emu(1.85), emu(0.14), 'Base model + frozen foreground-motion LoRA', font_size=980, color=SUBTEXT, align='ctr'))
    items.append(line_arrow('Arrow C2', emu(6.78), emu(4.92), emu(7.16), emu(4.92)))
    items.append(sp('roundRect', 'Result area', emu(7.20), emu(4.30), emu(2.55), emu(2.20), WHITE, GREEN, line_w=19050, radius_adj=16667))
    items.append(text_box('Result header', emu(7.32), emu(4.40), emu(1.5), emu(0.18), 'Edited result video', font_size=1500, bold=True, color=NAVY))
    items += group_frame_triplet(emu(7.34), emu(4.80), '', 'Novel foreground motion with the original background preserved.', GREEN, w_each=emu(0.66), h_each=emu(0.52), gap=emu(0.10))
    items.append(sp('roundRect', 'Output chip', emu(7.34), emu(5.92), emu(2.24), emu(0.26), GREEN_SOFT, '#9FD9BF', line_w=9525,
                    text='Output: new action, same real scene', font_size=1050, color='#196847', align='ctr'))

    items.append(sp('roundRect', 'Why box', emu(9.92), emu(4.30), emu(1.28), emu(2.20), WHITE, GRAY, line_w=12700, radius_adj=16667))
    items.append(text_box('Why title', emu(10.02), emu(4.40), emu(0.95), emu(0.18), 'Why this matters', font_size=1500, bold=True, color=NAVY))
    items.append(sp('roundRect', 'Our method box', emu(10.04), emu(4.80), emu(0.98), emu(0.66), GREEN_SOFT, '#9FD9BF', line_w=9525, radius_adj=16667,
                    text='Our method\nUses the original video at inference, so the real background is preserved.', font_size=980, color='#196847'))
    items.append(sp('roundRect', 'Baseline box', emu(10.04), emu(5.62), emu(0.98), emu(0.78), RED_SOFT, '#EF9A9A', line_w=9525, radius_adj=16667,
                    text='Prompt-only / first-frame baselines\nCan generate the action, but do not explicitly preserve the exact source-video background trajectory.', font_size=900, color=RED))

    items.append(text_box('Footer', emu(0.48), emu(7.02), emu(9.2), emu(0.18),
                          'Suggested caption: generate the target foreground motion with a first-frame-guided model, learn only that masked foreground motion with LoRA, then apply it on the original video to keep the real background while enabling novel actions.',
                          font_size=950, color=SUBTEXT))

    return f'''<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<p:sld xmlns:a="http://schemas.openxmlformats.org/drawingml/2006/main"
       xmlns:r="http://schemas.openxmlformats.org/officeDocument/2006/relationships"
       xmlns:p="http://schemas.openxmlformats.org/presentationml/2006/main">
  <p:cSld>
    <p:spTree>
      <p:nvGrpSpPr>
        <p:cNvPr id="1" name=""/>
        <p:cNvGrpSpPr/>
        <p:nvPr/>
      </p:nvGrpSpPr>
      <p:grpSpPr>
        <a:xfrm>
          <a:off x="0" y="0"/><a:ext cx="0" cy="0"/><a:chOff x="0" y="0"/><a:chExt cx="0" cy="0"/>
        </a:xfrm>
      </p:grpSpPr>
      {''.join(items)}
    </p:spTree>
  </p:cSld>
  <p:clrMapOvr><a:masterClrMapping/></p:clrMapOvr>
</p:sld>'''


def build_pptx() -> None:
    created = datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace('+00:00', 'Z')
    title = 'Editable Pipeline Diagram'
    slide_xml = build_slide()

    files = {
        '[Content_Types].xml': '''<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<Types xmlns="http://schemas.openxmlformats.org/package/2006/content-types">
  <Default Extension="rels" ContentType="application/vnd.openxmlformats-package.relationships+xml"/>
  <Default Extension="xml" ContentType="application/xml"/>
  <Override PartName="/ppt/presentation.xml" ContentType="application/vnd.openxmlformats-officedocument.presentationml.presentation.main+xml"/>
  <Override PartName="/ppt/slides/slide1.xml" ContentType="application/vnd.openxmlformats-officedocument.presentationml.slide+xml"/>
  <Override PartName="/ppt/slideMasters/slideMaster1.xml" ContentType="application/vnd.openxmlformats-officedocument.presentationml.slideMaster+xml"/>
  <Override PartName="/ppt/slideLayouts/slideLayout1.xml" ContentType="application/vnd.openxmlformats-officedocument.presentationml.slideLayout+xml"/>
  <Override PartName="/ppt/theme/theme1.xml" ContentType="application/vnd.openxmlformats-officedocument.theme+xml"/>
  <Override PartName="/ppt/presProps.xml" ContentType="application/vnd.openxmlformats-officedocument.presentationml.presProps+xml"/>
  <Override PartName="/ppt/viewProps.xml" ContentType="application/vnd.openxmlformats-officedocument.presentationml.viewProps+xml"/>
  <Override PartName="/ppt/tableStyles.xml" ContentType="application/vnd.openxmlformats-officedocument.presentationml.tableStyles+xml"/>
  <Override PartName="/docProps/core.xml" ContentType="application/vnd.openxmlformats-package.core-properties+xml"/>
  <Override PartName="/docProps/app.xml" ContentType="application/vnd.openxmlformats-officedocument.extended-properties+xml"/>
</Types>''',
        '_rels/.rels': '''<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">
  <Relationship Id="rId1" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/officeDocument" Target="ppt/presentation.xml"/>
  <Relationship Id="rId2" Type="http://schemas.openxmlformats.org/package/2006/relationships/metadata/core-properties" Target="docProps/core.xml"/>
  <Relationship Id="rId3" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/extended-properties" Target="docProps/app.xml"/>
</Relationships>''',
        'docProps/app.xml': '''<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<Properties xmlns="http://schemas.openxmlformats.org/officeDocument/2006/extended-properties"
            xmlns:vt="http://schemas.openxmlformats.org/officeDocument/2006/docPropsVTypes">
  <Application>OpenAI Codex</Application>
  <Slides>1</Slides>
  <Notes>0</Notes>
  <HiddenSlides>0</HiddenSlides>
  <MMClips>0</MMClips>
  <ScaleCrop>false</ScaleCrop>
  <HeadingPairs><vt:vector size="2" baseType="variant"><vt:variant><vt:lpstr>Slides</vt:lpstr></vt:variant><vt:variant><vt:i4>1</vt:i4></vt:variant></vt:vector></HeadingPairs>
  <TitlesOfParts><vt:vector size="1" baseType="lpstr"><vt:lpstr>Editable Pipeline Diagram</vt:lpstr></vt:vector></TitlesOfParts>
  <Company>OpenAI</Company>
  <AppVersion>1.0</AppVersion>
</Properties>''',
        'docProps/core.xml': f'''<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<cp:coreProperties xmlns:cp="http://schemas.openxmlformats.org/package/2006/metadata/core-properties" xmlns:dc="http://purl.org/dc/elements/1.1/" xmlns:dcterms="http://purl.org/dc/terms/" xmlns:dcmitype="http://purl.org/dc/dcmitype/" xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance">
  <dc:title>{esc(title)}</dc:title>
  <dc:creator>OpenAI Codex</dc:creator>
  <cp:lastModifiedBy>OpenAI Codex</cp:lastModifiedBy>
  <dcterms:created xsi:type="dcterms:W3CDTF">{created}</dcterms:created>
  <dcterms:modified xsi:type="dcterms:W3CDTF">{created}</dcterms:modified>
</cp:coreProperties>''',
        'ppt/presentation.xml': f'''<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<p:presentation xmlns:a="http://schemas.openxmlformats.org/drawingml/2006/main" xmlns:r="http://schemas.openxmlformats.org/officeDocument/2006/relationships" xmlns:p="http://schemas.openxmlformats.org/presentationml/2006/main" saveSubsetFonts="1" autoCompressPictures="0">
  <p:sldMasterIdLst><p:sldMasterId id="2147483648" r:id="rId1"/></p:sldMasterIdLst>
  <p:sldIdLst><p:sldId id="256" r:id="rId2"/></p:sldIdLst>
  <p:sldSz cx="{SLIDE_CX}" cy="{SLIDE_CY}"/>
  <p:notesSz cx="6858000" cy="9144000"/>
  <p:defaultTextStyle><a:defPPr/></p:defaultTextStyle>
</p:presentation>''',
        'ppt/_rels/presentation.xml.rels': '''<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">
  <Relationship Id="rId1" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/slideMaster" Target="slideMasters/slideMaster1.xml"/>
  <Relationship Id="rId2" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/slide" Target="slides/slide1.xml"/>
  <Relationship Id="rId3" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/presProps" Target="presProps.xml"/>
  <Relationship Id="rId4" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/viewProps" Target="viewProps.xml"/>
  <Relationship Id="rId5" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/theme" Target="theme/theme1.xml"/>
  <Relationship Id="rId6" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/tableStyles" Target="tableStyles.xml"/>
</Relationships>''',
        'ppt/slides/slide1.xml': slide_xml,
        'ppt/slides/_rels/slide1.xml.rels': '''<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">
  <Relationship Id="rId1" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/slideLayout" Target="../slideLayouts/slideLayout1.xml"/>
</Relationships>''',
        'ppt/slideMasters/slideMaster1.xml': '''<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<p:sldMaster xmlns:a="http://schemas.openxmlformats.org/drawingml/2006/main" xmlns:r="http://schemas.openxmlformats.org/officeDocument/2006/relationships" xmlns:p="http://schemas.openxmlformats.org/presentationml/2006/main">
  <p:cSld name="Office Theme"><p:spTree><p:nvGrpSpPr><p:cNvPr id="1" name=""/><p:cNvGrpSpPr/><p:nvPr/></p:nvGrpSpPr><p:grpSpPr><a:xfrm><a:off x="0" y="0"/><a:ext cx="0" cy="0"/><a:chOff x="0" y="0"/><a:chExt cx="0" cy="0"/></a:xfrm></p:grpSpPr></p:spTree></p:cSld>
  <p:clrMap bg1="lt1" tx1="dk1" bg2="lt2" tx2="dk2" accent1="accent1" accent2="accent2" accent3="accent3" accent4="accent4" accent5="accent5" accent6="accent6" hlink="hlink" folHlink="folHlink"/>
  <p:sldLayoutIdLst><p:sldLayoutId id="1" r:id="rId1"/></p:sldLayoutIdLst>
  <p:txStyles><p:titleStyle/><p:bodyStyle/><p:otherStyle/></p:txStyles>
</p:sldMaster>''',
        'ppt/slideMasters/_rels/slideMaster1.xml.rels': '''<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">
  <Relationship Id="rId1" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/slideLayout" Target="../slideLayouts/slideLayout1.xml"/>
  <Relationship Id="rId2" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/theme" Target="../theme/theme1.xml"/>
</Relationships>''',
        'ppt/slideLayouts/slideLayout1.xml': '''<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<p:sldLayout xmlns:a="http://schemas.openxmlformats.org/drawingml/2006/main" xmlns:r="http://schemas.openxmlformats.org/officeDocument/2006/relationships" xmlns:p="http://schemas.openxmlformats.org/presentationml/2006/main" type="blank" preserve="1">
  <p:cSld name="Blank"><p:spTree><p:nvGrpSpPr><p:cNvPr id="1" name=""/><p:cNvGrpSpPr/><p:nvPr/></p:nvGrpSpPr><p:grpSpPr><a:xfrm><a:off x="0" y="0"/><a:ext cx="0" cy="0"/><a:chOff x="0" y="0"/><a:chExt cx="0" cy="0"/></a:xfrm></p:grpSpPr></p:spTree></p:cSld>
  <p:clrMapOvr><a:masterClrMapping/></p:clrMapOvr>
</p:sldLayout>''',
        'ppt/slideLayouts/_rels/slideLayout1.xml.rels': '''<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">
  <Relationship Id="rId1" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/slideMaster" Target="../slideMasters/slideMaster1.xml"/>
</Relationships>''',
        'ppt/theme/theme1.xml': '''<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<a:theme xmlns:a="http://schemas.openxmlformats.org/drawingml/2006/main" name="Office Theme"><a:themeElements><a:clrScheme name="Office"><a:dk1><a:srgbClr val="000000"/></a:dk1><a:lt1><a:srgbClr val="FFFFFF"/></a:lt1><a:dk2><a:srgbClr val="1F497D"/></a:dk2><a:lt2><a:srgbClr val="EEECE1"/></a:lt2><a:accent1><a:srgbClr val="4F81BD"/></a:accent1><a:accent2><a:srgbClr val="C0504D"/></a:accent2><a:accent3><a:srgbClr val="9BBB59"/></a:accent3><a:accent4><a:srgbClr val="8064A2"/></a:accent4><a:accent5><a:srgbClr val="4BACC6"/></a:accent5><a:accent6><a:srgbClr val="F79646"/></a:accent6><a:hlink><a:srgbClr val="0000FF"/></a:hlink><a:folHlink><a:srgbClr val="800080"/></a:folHlink></a:clrScheme><a:fontScheme name="Office"><a:majorFont><a:latin typeface="Aptos Display"/><a:ea typeface=""/><a:cs typeface=""/></a:majorFont><a:minorFont><a:latin typeface="Aptos"/><a:ea typeface=""/><a:cs typeface=""/></a:minorFont></a:fontScheme><a:fmtScheme name="Office"><a:fillStyleLst><a:solidFill><a:schemeClr val="phClr"/></a:solidFill></a:fillStyleLst><a:lnStyleLst><a:ln w="9525"><a:solidFill><a:schemeClr val="phClr"/></a:solidFill></a:ln></a:lnStyleLst><a:effectStyleLst><a:effectStyle><a:effectLst/></a:effectStyle></a:effectStyleLst><a:bgFillStyleLst><a:solidFill><a:schemeClr val="phClr"/></a:solidFill></a:bgFillStyleLst></a:fmtScheme></a:themeElements><a:objectDefaults/><a:extraClrSchemeLst/></a:theme>''',
        'ppt/presProps.xml': '<?xml version="1.0" encoding="UTF-8" standalone="yes"?><p:presentationPr xmlns:a="http://schemas.openxmlformats.org/drawingml/2006/main" xmlns:r="http://schemas.openxmlformats.org/officeDocument/2006/relationships" xmlns:p="http://schemas.openxmlformats.org/presentationml/2006/main"/>',
        'ppt/viewProps.xml': '<?xml version="1.0" encoding="UTF-8" standalone="yes"?><p:viewPr xmlns:a="http://schemas.openxmlformats.org/drawingml/2006/main" xmlns:r="http://schemas.openxmlformats.org/officeDocument/2006/relationships" xmlns:p="http://schemas.openxmlformats.org/presentationml/2006/main" lastView="sldView"><p:normalViewPr/><p:slideViewPr><p:cSldViewPr snapToGrid="1" snapToObjects="1"/></p:slideViewPr><p:notesTextViewPr/><p:gridSpacing cx="72008" cy="72008"/></p:viewPr>',
        'ppt/tableStyles.xml': '<?xml version="1.0" encoding="UTF-8" standalone="yes"?><a:tblStyleLst xmlns:a="http://schemas.openxmlformats.org/drawingml/2006/main" def="{5C22544A-7EE6-4342-B048-85BDC9FD1C3A}"/>',
    }

    with ZipFile(OUT, 'w', compression=ZIP_DEFLATED) as zf:
        for path, content in files.items():
            zf.writestr(path, content)


if __name__ == '__main__':
    build_pptx()
    print(f'Wrote {OUT}')
