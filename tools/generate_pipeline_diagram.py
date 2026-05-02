from __future__ import annotations

from pathlib import Path
from textwrap import wrap

from PIL import Image, ImageDraw, ImageFont


ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = ROOT / "outputs"
PNG_PATH = OUT_DIR / "modified_pipeline_diagram.png"
SVG_PATH = OUT_DIR / "modified_pipeline_diagram.svg"

W, H = 2600, 1500
BG = "#f5f8fc"
PANEL = "#eaf1fb"
WHITE = "#ffffff"
NAVY = "#17324d"
TEXT = "#1f2d3d"
SUBTEXT = "#4a6178"
BLUE = "#4f83ff"
BLUE_SOFT = "#dce9ff"
ORANGE = "#f28c28"
ORANGE_SOFT = "#fff0dd"
GREEN = "#1e9b63"
GREEN_SOFT = "#e4f8ef"
RED = "#d64545"
RED_SOFT = "#fde8e8"
GRAY = "#9aa8b6"
MASK = "#8f9399"
PROMPT = "#fff7db"
PROMPT_BORDER = "#efc14d"
ARROW = "#7a93b8"


def get_font(size: int, bold: bool = False) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    candidates = []
    if bold:
        candidates.extend(
            [
                "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
                "/usr/share/fonts/truetype/liberation2/LiberationSans-Bold.ttf",
            ]
        )
    else:
        candidates.extend(
            [
                "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
                "/usr/share/fonts/truetype/liberation2/LiberationSans-Regular.ttf",
            ]
        )

    for path in candidates:
        p = Path(path)
        if p.exists():
            return ImageFont.truetype(str(p), size=size)
    return ImageFont.load_default()


FONT_TITLE = get_font(42, bold=True)
FONT_SECTION = get_font(28, bold=True)
FONT_LABEL = get_font(24, bold=True)
FONT_BODY = get_font(22, bold=False)
FONT_SMALL = get_font(18, bold=False)
FONT_TINY = get_font(16, bold=False)


def rounded_rect(draw: ImageDraw.ImageDraw, box, fill, outline=None, width=2, radius=24):
    draw.rounded_rectangle(box, radius=radius, fill=fill, outline=outline, width=width)


def centered_text(draw: ImageDraw.ImageDraw, box, text, font, fill=TEXT):
    left, top, right, bottom = box
    bbox = draw.multiline_textbbox((0, 0), text, font=font, spacing=4, align="center")
    tw = bbox[2] - bbox[0]
    th = bbox[3] - bbox[1]
    x = left + (right - left - tw) / 2
    y = top + (bottom - top - th) / 2
    draw.multiline_text((x, y), text, font=font, fill=fill, spacing=4, align="center")


def wrapped_text(draw: ImageDraw.ImageDraw, xy, text, font, fill=TEXT, width_chars=42, line_spacing=4):
    wrapped = "\n".join(wrap(text, width=width_chars))
    draw.multiline_text(xy, wrapped, font=font, fill=fill, spacing=line_spacing)


def add_frame_strip(draw: ImageDraw.ImageDraw, x, y, w, h, title, note, accent, dashed=False):
    if title:
        draw.text((x, y - 38), title, font=FONT_LABEL, fill=NAVY)
    frame_gap = 18
    frame_w = int((w - 2 * frame_gap) / 3)
    labels = ["Frame 1", "Frame 2", "Frame 3"]
    for i, lbl in enumerate(labels):
        fx = x + i * (frame_w + frame_gap)
        fy = y
        box = (fx, fy, fx + frame_w, fy + h)
        if dashed:
            draw.rounded_rectangle(box, radius=20, outline=accent, width=4)
            # Fake dashed mask overlay.
            draw.rounded_rectangle(
                (fx + 18, fy + 18, fx + frame_w - 18, fy + h - 18),
                radius=16,
                outline=GRAY,
                width=3,
            )
        else:
            rounded_rect(draw, box, WHITE, outline=accent, width=4, radius=20)
        centered_text(draw, (fx + 20, fy + 18, fx + frame_w - 20, fy + h - 18), f"{lbl}\n(Add video frame)", FONT_BODY, fill=SUBTEXT)
    wrapped_text(draw, (x, y + h + 14), note, FONT_SMALL, fill=SUBTEXT, width_chars=max(24, int(w / 18)))


def add_prompt_box(draw: ImageDraw.ImageDraw, x, y, w, h, title, body):
    rounded_rect(draw, (x, y, x + w, y + h), PROMPT, outline=PROMPT_BORDER, width=3, radius=22)
    draw.text((x + 18, y + 12), title, font=FONT_SMALL, fill="#8b5b00")
    wrapped_text(draw, (x + 18, y + 42), body, FONT_SMALL, fill="#6b5200", width_chars=max(18, int(w / 13)))


def add_arrow(draw: ImageDraw.ImageDraw, x1, y1, x2, y2, color=ARROW, width=8):
    draw.line((x1, y1, x2, y2), fill=color, width=width)
    ah = 16
    draw.polygon([(x2, y2), (x2 - 26, y2 - ah), (x2 - 26, y2 + ah)], fill=color)


def add_lora_block(draw: ImageDraw.ImageDraw, x, y, w, h, title, subtitle, train=True):
    rounded_rect(draw, (x, y, x + w, y + h), WHITE, outline=BLUE, width=4, radius=26)
    for i, bar_x in enumerate([x + 44, x + 112, x + w - 132, x + w - 64]):
        draw.rounded_rectangle((bar_x, y + 70 + (i % 2) * 18, bar_x + 24, y + h - 70 - (i % 2) * 18), radius=10, fill="#9cb7ee")
    for bar_x in [x + 182, x + w - 202]:
        draw.rounded_rectangle((bar_x, y + 58, bar_x + 28, y + h - 58), radius=10, fill=ORANGE)
    draw.polygon([(x + 26, y + h / 2), (x + 88, y + 42), (x + 88, y + h - 42)], fill=BLUE_SOFT)
    draw.polygon([(x + w - 26, y + h / 2), (x + w - 88, y + 42), (x + w - 88, y + h - 42)], fill=BLUE_SOFT)
    chip_fill = ORANGE if train else BLUE
    rounded_rect(draw, (x + w / 2 - 82, y - 34, x + w / 2 + 82, y + 30), chip_fill, outline=None, width=1, radius=18)
    draw.text((x + w / 2 - 44, y - 18), "LoRA", font=FONT_SECTION, fill=WHITE)
    if train:
        draw.text((x + w / 2 + 46, y - 28), "🔥", font=FONT_SECTION, fill=ORANGE)
    else:
        draw.text((x + w / 2 + 44, y - 28), "❄", font=FONT_SECTION, fill=BLUE)
    centered_text(draw, (x + 14, y + h + 8, x + w - 14, y + h + 68), title, FONT_LABEL, fill=NAVY)
    centered_text(draw, (x + 10, y + h + 48, x + w - 10, y + h + 108), subtitle, FONT_SMALL, fill=SUBTEXT)


def draw_png():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    img = Image.new("RGBA", (W, H), BG)
    draw = ImageDraw.Draw(img)

    draw.text((70, 36), "Foreground Motion Learning with Background Preservation", font=FONT_TITLE, fill=NAVY)
    draw.text((72, 92), "Train foreground motion from a generated reference, then apply it on the original video to preserve the real background.", font=FONT_BODY, fill=SUBTEXT)

    # Top left panel: motion proposal
    rounded_rect(draw, (48, 150, 1220, 700), PANEL, outline=None, radius=34)
    draw.text((80, 176), "A. Motion Proposal from First Frame + Prompt", font=FONT_SECTION, fill=NAVY)
    add_frame_strip(
        draw,
        82,
        244,
        430,
        180,
        "Input: Edited first frame",
        "Insert your edited first-frame examples here.",
        BLUE,
    )
    add_prompt_box(
        draw,
        82,
        505,
        430,
        122,
        "Prompt",
        "Example: girl passes the potted plant to the person on her left.",
    )
    add_arrow(draw, 540, 392, 620, 392)
    rounded_rect(draw, (650, 238, 1170, 630), WHITE, outline=ORANGE, width=4, radius=28)
    draw.text((680, 260), "WAN / generative prior output", font=FONT_LABEL, fill=NAVY)
    add_frame_strip(
        draw,
        680,
        310,
        460,
        180,
        "Generated motion video",
        "Desired handoff motion appears here. Background may drift from the source video.",
        ORANGE,
        dashed=True,
    )
    rounded_rect(draw, (680, 556, 1140, 610), ORANGE_SOFT, outline="#f0b264", width=2, radius=18)
    centered_text(draw, (700, 564, 1120, 604), "Use this clip only as a motion reference", FONT_SMALL, fill="#8a4f00")

    # Top right panel: LoRA training
    rounded_rect(draw, (1260, 150, 2550, 700), PANEL, outline=None, radius=34)
    draw.text((1294, 176), "B. Foreground Motion LoRA Training", font=FONT_SECTION, fill=NAVY)
    add_frame_strip(
        draw,
        1294,
        244,
        470,
        180,
        "Mask WAN video foreground",
        "Mask only the moving foreground: girl + receiver + object.",
        BLUE,
        dashed=True,
    )
    add_arrow(draw, 1795, 392, 1880, 392)
    add_lora_block(
        draw,
        1910,
        270,
        250,
        190,
        "Learned representation",
        "LoRA learns foreground motion only",
        train=True,
    )
    add_arrow(draw, 2188, 392, 2272, 392)
    rounded_rect(draw, (2300, 250, 2512, 530), WHITE, outline=GREEN, width=4, radius=24)
    centered_text(draw, (2320, 286, 2492, 352), "Training Target", FONT_LABEL, fill=NAVY)
    centered_text(draw, (2320, 354, 2492, 488), "Foreground-motion\nrepresentation\nonly", FONT_BODY, fill=SUBTEXT)
    rounded_rect(draw, (1294, 548, 2512, 626), GREEN_SOFT, outline="#8dd4b2", width=2, radius=20)
    centered_text(draw, (1320, 564, 2486, 612), "Key change: learn foreground motion from the generated masked clip, not from the source-video foreground motion.", FONT_SMALL, fill="#196847")

    # Bottom main panel: inference
    rounded_rect(draw, (48, 760, 2550, 1432), "#eef4fb", outline=None, radius=34)
    draw.text((80, 786), "C. Inference on the Original Video", font=FONT_SECTION, fill=NAVY)
    add_frame_strip(
        draw,
        82,
        858,
        620,
        190,
        "Original video frames",
        "Source video supplies the true background.",
        BLUE,
    )
    add_frame_strip(
        draw,
        82,
        1148,
        620,
        160,
        "Same foreground mask on original video",
        "Apply the same foreground mask used during training.",
        ORANGE,
        dashed=True,
    )
    add_prompt_box(
        draw,
        742,
        912,
        360,
        116,
        "Prompt at inference",
        "Reuse the handoff action prompt.",
    )
    add_arrow(draw, 1118, 968, 1200, 968)
    add_lora_block(
        draw,
        1230,
        864,
        280,
        210,
        "Inference module",
        "Base model + frozen foreground-motion LoRA",
        train=False,
    )
    add_arrow(draw, 1538, 968, 1625, 968)
    rounded_rect(draw, (1656, 838, 2208, 1312), WHITE, outline=GREEN, width=4, radius=28)
    draw.text((1684, 866), "Edited result video", font=FONT_LABEL, fill=NAVY)
    add_frame_strip(
        draw,
        1684,
        918,
        494,
        190,
        "",
        "Novel foreground motion with the original background preserved.",
        GREEN,
    )
    rounded_rect(draw, (1684, 1170, 2178, 1246), GREEN_SOFT, outline="#8dd4b2", width=2, radius=18)
    centered_text(draw, (1704, 1182, 2158, 1238), "Output: new action, same real scene", FONT_SMALL, fill="#196847")

    # Comparison callout
    rounded_rect(draw, (2240, 840, 2518, 1312), WHITE, outline=GRAY, width=3, radius=24)
    draw.text((2264, 866), "Why this matters", font=FONT_LABEL, fill=NAVY)
    rounded_rect(draw, (2264, 922, 2494, 1066), GREEN_SOFT, outline="#9fd9bf", width=2, radius=18)
    draw.text((2284, 944), "Our method", font=FONT_SMALL, fill="#196847")
    wrapped_text(
        draw,
        (2284, 978),
        "Uses the original video at inference, so the real background is preserved.",
        FONT_TINY,
        fill="#196847",
        width_chars=25,
    )
    rounded_rect(draw, (2264, 1092, 2494, 1264), RED_SOFT, outline="#ef9a9a", width=2, radius=18)
    draw.text((2284, 1114), "Prompt-only / first-frame baselines", font=FONT_SMALL, fill=RED)
    wrapped_text(
        draw,
        (2284, 1148),
        "Can generate the action, but do not explicitly preserve the exact source-video background trajectory.",
        FONT_TINY,
        fill=RED,
        width_chars=25,
    )

    # Footer note
    footer = (
        "Suggested caption: generate the target foreground motion with a first-frame-guided model, learn only that masked foreground motion with LoRA, then apply it on the original video to keep the real background while enabling novel actions."
    )
    wrapped_text(draw, (72, 1452), footer, FONT_SMALL, fill=SUBTEXT, width_chars=160)

    img.save(PNG_PATH)


def esc(text: str) -> str:
    return (
        text.replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
        .replace('"', "&quot;")
    )


def svg_text(x, y, text, size=22, weight="400", fill=TEXT, anchor="start"):
    return f'<text x="{x}" y="{y}" font-family="DejaVu Sans, Arial, sans-serif" font-size="{size}" font-weight="{weight}" fill="{fill}" text-anchor="{anchor}">{esc(text)}</text>'


def svg_multiline(x, y, text, size=20, weight="400", fill=TEXT, width_chars=32, line_height=1.3):
    lines = wrap(text, width_chars)
    tspans = []
    for i, line in enumerate(lines):
        dy = "0" if i == 0 else f"{line_height}em"
        tspans.append(f'<tspan x="{x}" dy="{dy}">{esc(line)}</tspan>')
    return f'<text x="{x}" y="{y}" font-family="DejaVu Sans, Arial, sans-serif" font-size="{size}" font-weight="{weight}" fill="{fill}">{"".join(tspans)}</text>'


def svg_round_rect(x, y, w, h, rx, fill, stroke="none", stroke_width=1):
    return f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="{rx}" fill="{fill}" stroke="{stroke}" stroke-width="{stroke_width}"/>'


def svg_frame_group(x, y, w, h, accent, title, note, dashed=False):
    gap = 18
    frame_w = int((w - 2 * gap) / 3)
    parts = []
    if title:
        parts.append(svg_text(x, y - 16, title, size=24, weight="700", fill=NAVY))
    for i, lbl in enumerate(["Frame 1", "Frame 2", "Frame 3"]):
        fx = x + i * (frame_w + gap)
        dash = ' stroke-dasharray="12 10"' if dashed else ""
        parts.append(
            f'<rect x="{fx}" y="{y}" width="{frame_w}" height="{h}" rx="20" fill="{WHITE}" stroke="{accent}" stroke-width="4"{dash}/>'
        )
        if dashed:
            parts.append(
                f'<rect x="{fx + 18}" y="{y + 18}" width="{frame_w - 36}" height="{h - 36}" rx="16" fill="none" stroke="{GRAY}" stroke-width="3" stroke-dasharray="10 8"/>'
            )
        parts.append(svg_text(fx + frame_w / 2, y + h / 2 - 8, lbl, size=22, weight="700", fill=SUBTEXT, anchor="middle"))
        parts.append(svg_text(fx + frame_w / 2, y + h / 2 + 24, "(Add video frame)", size=18, fill=SUBTEXT, anchor="middle"))
    parts.append(svg_multiline(x, y + h + 28, note, size=18, fill=SUBTEXT, width_chars=max(22, int(w / 18))))
    return "\n".join(parts)


def arrow_svg(x1, y1, x2, y2, color=ARROW):
    return (
        f'<line x1="{x1}" y1="{y1}" x2="{x2}" y2="{y2}" stroke="{color}" stroke-width="8" stroke-linecap="round"/>'
        f'<polygon points="{x2},{y2} {x2-26},{y2-16} {x2-26},{y2+16}" fill="{color}"/>'
    )


def lora_svg(x, y, w, h, train=True, title="", subtitle=""):
    chip = ORANGE if train else BLUE
    icon = "🔥" if train else "❄"
    parts = [svg_round_rect(x, y, w, h, 26, WHITE, stroke=BLUE, stroke_width=4)]
    parts.append(f'<polygon points="{x+26},{y+h/2} {x+88},{y+42} {x+88},{y+h-42}" fill="{BLUE_SOFT}"/>')
    parts.append(f'<polygon points="{x+w-26},{y+h/2} {x+w-88},{y+42} {x+w-88},{y+h-42}" fill="{BLUE_SOFT}"/>')
    for bar_x, y1, y2, color in [
        (x + 44, y + 70, y + h - 70, "#9cb7ee"),
        (x + 112, y + 88, y + h - 88, "#9cb7ee"),
        (x + 182, y + 58, y + h - 58, ORANGE),
        (x + w - 202, y + 58, y + h - 58, ORANGE),
        (x + w - 132, y + 88, y + h - 88, "#9cb7ee"),
        (x + w - 64, y + 70, y + h - 70, "#9cb7ee"),
    ]:
        parts.append(f'<rect x="{bar_x}" y="{y1}" width="26" height="{y2-y1}" rx="10" fill="{color}"/>')
    parts.append(svg_round_rect(x + w / 2 - 82, y - 34, 164, 64, 18, chip))
    parts.append(svg_text(x + w / 2, y + 8, "LoRA", size=28, weight="700", fill=WHITE, anchor="middle"))
    parts.append(svg_text(x + w / 2 + 58, y + 8, icon, size=26, fill=chip, anchor="start"))
    if title:
        parts.append(svg_text(x + w / 2, y + h + 30, title, size=24, weight="700", fill=NAVY, anchor="middle"))
    if subtitle:
        parts.append(svg_text(x + w / 2, y + h + 60, subtitle, size=18, fill=SUBTEXT, anchor="middle"))
    return "\n".join(parts)


def draw_svg():
    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{W}" height="{H}" viewBox="0 0 {W} {H}">',
        svg_round_rect(0, 0, W, H, 0, BG),
        svg_text(70, 80, "Foreground Motion Learning with Background Preservation", size=42, weight="700", fill=NAVY),
        svg_text(72, 122, "Train foreground motion from a generated reference, then apply it on the original video to preserve the real background.", size=22, fill=SUBTEXT),
        svg_round_rect(48, 150, 1172, 550, 34, PANEL),
        svg_text(80, 194, "A. Motion Proposal from First Frame + Prompt", size=28, weight="700", fill=NAVY),
        svg_frame_group(82, 244, 430, 180, BLUE, "Input: Edited first frame", "Insert your edited first-frame examples here."),
        svg_round_rect(82, 505, 430, 122, 22, PROMPT, stroke=PROMPT_BORDER, stroke_width=3),
        svg_text(100, 536, "Prompt", size=18, weight="700", fill="#8b5b00"),
        svg_multiline(100, 566, "Example: girl passes the potted plant to the person on her left.", size=18, fill="#6b5200", width_chars=32),
        arrow_svg(540, 392, 620, 392),
        svg_round_rect(650, 238, 520, 392, 28, WHITE, stroke=ORANGE, stroke_width=4),
        svg_text(680, 276, "WAN / generative prior output", size=24, weight="700", fill=NAVY),
        svg_frame_group(680, 310, 460, 180, ORANGE, "Generated motion video", "Desired handoff motion appears here. Background may drift from the source video.", dashed=True),
        svg_round_rect(680, 556, 460, 54, 18, ORANGE_SOFT, stroke="#f0b264", stroke_width=2),
        svg_text(910, 589, "Use this clip only as a motion reference", size=18, fill="#8a4f00", anchor="middle"),
        svg_round_rect(1260, 150, 1290, 550, 34, PANEL),
        svg_text(1294, 194, "B. Foreground Motion LoRA Training", size=28, weight="700", fill=NAVY),
        svg_frame_group(1294, 244, 470, 180, BLUE, "Mask WAN video foreground", "Mask only the moving foreground: girl + receiver + object.", dashed=True),
        arrow_svg(1795, 392, 1880, 392),
        lora_svg(1910, 270, 250, 190, True, "Learned representation", "LoRA learns foreground motion only"),
        arrow_svg(2188, 392, 2272, 392),
        svg_round_rect(2300, 250, 212, 280, 24, WHITE, stroke=GREEN, stroke_width=4),
        svg_text(2406, 320, "Training Target", size=24, weight="700", fill=NAVY, anchor="middle"),
        svg_text(2406, 390, "Foreground-motion", size=22, fill=SUBTEXT, anchor="middle"),
        svg_text(2406, 424, "representation", size=22, fill=SUBTEXT, anchor="middle"),
        svg_text(2406, 458, "only", size=22, fill=SUBTEXT, anchor="middle"),
        svg_round_rect(1294, 548, 1218, 78, 20, GREEN_SOFT, stroke="#8dd4b2", stroke_width=2),
        svg_multiline(1320, 580, "Key change: learn foreground motion from the generated masked clip, not from the source-video foreground motion.", size=18, fill="#196847", width_chars=105),
        svg_round_rect(48, 760, 2502, 672, 34, "#eef4fb"),
        svg_text(80, 804, "C. Inference on the Original Video", size=28, weight="700", fill=NAVY),
        svg_frame_group(82, 858, 620, 190, BLUE, "Original video frames", "Source video supplies the true background."),
        svg_frame_group(82, 1148, 620, 160, ORANGE, "Same foreground mask on original video", "Apply the same foreground mask used during training.", dashed=True),
        svg_round_rect(742, 912, 360, 116, 22, PROMPT, stroke=PROMPT_BORDER, stroke_width=3),
        svg_text(760, 944, "Prompt at inference", size=18, weight="700", fill="#8b5b00"),
        svg_multiline(760, 976, "Reuse the handoff action prompt.", size=18, fill="#6b5200", width_chars=28),
        arrow_svg(1118, 968, 1200, 968),
        lora_svg(1230, 864, 280, 210, False, "Inference module", "Base model + frozen foreground-motion LoRA"),
        arrow_svg(1538, 968, 1625, 968),
        svg_round_rect(1656, 838, 552, 474, 28, WHITE, stroke=GREEN, stroke_width=4),
        svg_text(1684, 876, "Edited result video", size=24, weight="700", fill=NAVY),
        svg_frame_group(1684, 918, 494, 190, GREEN, "", "Novel foreground motion with the original background preserved."),
        svg_round_rect(1684, 1170, 494, 76, 18, GREEN_SOFT, stroke="#8dd4b2", stroke_width=2),
        svg_text(1931, 1214, "Output: new action, same real scene", size=18, fill="#196847", anchor="middle"),
        svg_round_rect(2240, 840, 278, 472, 24, WHITE, stroke=GRAY, stroke_width=3),
        svg_text(2264, 876, "Why this matters", size=24, weight="700", fill=NAVY),
        svg_round_rect(2264, 922, 230, 144, 18, GREEN_SOFT, stroke="#9fd9bf", stroke_width=2),
        svg_text(2284, 950, "Our method", size=18, weight="700", fill="#196847"),
        svg_multiline(2284, 980, "Uses the original video at inference, so the real background is preserved.", size=16, fill="#196847", width_chars=24),
        svg_round_rect(2264, 1092, 230, 172, 18, RED_SOFT, stroke="#ef9a9a", stroke_width=2),
        svg_text(2284, 1120, "Prompt-only / first-frame baselines", size=18, weight="700", fill=RED),
        svg_multiline(2284, 1152, "Can generate the action, but do not explicitly preserve the exact source-video background trajectory.", size=16, fill=RED, width_chars=24),
        svg_multiline(72, 1468, "Suggested caption: generate the target foreground motion with a first-frame-guided model, learn only that masked foreground motion with LoRA, then apply it on the original video to keep the real background while enabling novel actions.", size=18, fill=SUBTEXT, width_chars=160),
        "</svg>",
    ]
    SVG_PATH.write_text("\n".join(parts), encoding="utf-8")


if __name__ == "__main__":
    draw_png()
    draw_svg()
    print(f"Wrote {PNG_PATH}")
    print(f"Wrote {SVG_PATH}")
