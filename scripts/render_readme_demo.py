from __future__ import annotations

from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

WIDTH = 1100
HEIGHT = 620
OUT = Path("docs/assets/mcp-dataframe-qa-demo.gif")

COMMAND = (
    '$ uv run mcp-dataframe-qa --ask\n'
    '  "What are the top metros by median list price?"'
)

ROWS = [
    ("1", "Vineyard Haven, MA", "$2.00M"),
    ("2", "Jackson, WY", "$1.58M"),
    ("3", "Edwards, CO", "$1.36M"),
]

STEPS = [
    ("Compact profile", "schema + stats, not raw rows"),
    ("AnalysisPlan", "group, median, sort, limit"),
    ("Validate", "columns, types, caps, policy"),
    ("Read-only Pandas", "local deterministic execution"),
    ("Structured result", "table + chart hint + audit id"),
]

COLORS = {
    "bg": "#f5f7fb",
    "ink": "#172033",
    "muted": "#5a6475",
    "terminal": "#111827",
    "terminal_2": "#0b1220",
    "terminal_border": "#263244",
    "green": "#25c481",
    "green_dark": "#0b8f5a",
    "cyan": "#3fb8ff",
    "yellow": "#ffd166",
    "red": "#ff6b6b",
    "white": "#f8fafc",
    "line": "#d8dee9",
    "card": "#ffffff",
    "card_border": "#d9e0ea",
    "soft_green": "#dff7eb",
    "soft_cyan": "#e5f5ff",
}


def load_font(size: int, *, mono: bool = False, bold: bool = False) -> ImageFont.FreeTypeFont:
    if mono:
        candidates = [
            "/System/Library/Fonts/Menlo.ttc",
            "/System/Library/Fonts/Monaco.ttf",
            "/System/Library/Fonts/SFNSMono.ttf",
            "/usr/share/fonts/truetype/dejavu/DejaVuSansMono.ttf",
        ]
    elif bold:
        candidates = [
            "/System/Library/Fonts/Supplemental/Arial Bold.ttf",
            "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
            "/System/Library/Fonts/SFNS.ttf",
        ]
    else:
        candidates = [
            "/System/Library/Fonts/Supplemental/Arial.ttf",
            "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
            "/System/Library/Fonts/SFNS.ttf",
        ]
    for candidate in candidates:
        try:
            return ImageFont.truetype(candidate, size)
        except OSError:
            continue
    return ImageFont.load_default()


FONT = {
    "title": load_font(30, bold=True),
    "subtitle": load_font(17),
    "label": load_font(14, bold=True),
    "body": load_font(15),
    "small": load_font(12),
    "mono": load_font(17, mono=True),
    "mono_small": load_font(14, mono=True),
    "mono_bold": load_font(17, mono=True),
}


def rounded(
    draw: ImageDraw.ImageDraw,
    box: tuple[int, int, int, int],
    radius: int,
    fill: str,
    outline: str | None = None,
    width: int = 1,
) -> None:
    draw.rounded_rectangle(box, radius=radius, fill=fill, outline=outline, width=width)


def text(
    draw: ImageDraw.ImageDraw,
    xy: tuple[int, int],
    value: str,
    font: ImageFont.ImageFont,
    fill: str,
) -> None:
    draw.text(xy, value, font=font, fill=fill)


def draw_header(draw: ImageDraw.ImageDraw) -> None:
    text(draw, (48, 32), "MCP DataFrame QA", FONT["title"], COLORS["ink"])
    text(
        draw,
        (48, 68),
        "Ask in English. Validate a typed plan. Execute local Pandas. Return focused facts.",
        FONT["subtitle"],
        COLORS["muted"],
    )
    rounded(draw, (824, 36, 1046, 72), 18, COLORS["soft_green"], COLORS["green_dark"])
    text(draw, (844, 46), "No raw dataframe prompt dump", FONT["label"], COLORS["green_dark"])


def draw_terminal_shell(draw: ImageDraw.ImageDraw) -> None:
    rounded(draw, (44, 100, 704, 580), 16, COLORS["terminal"], COLORS["terminal_border"], 2)
    rounded(draw, (44, 100, 704, 138), 16, COLORS["terminal_2"], COLORS["terminal_border"], 2)
    draw.rectangle((46, 122, 702, 139), fill=COLORS["terminal_2"])
    for i, color in enumerate([COLORS["red"], COLORS["yellow"], COLORS["green"]]):
        draw.ellipse((66 + i * 24, 114, 78 + i * 24, 126), fill=color)
    text(draw, (150, 111), "local MCP server demo", FONT["small"], "#9ca3af")


def draw_command(draw: ImageDraw.ImageDraw, chars: int) -> int:
    y = 158
    visible = COMMAND[:chars]
    lines = visible.split("\n")
    for line in lines:
        text(draw, (66, y), line, FONT["mono"], COLORS["white"])
        y += 28
    if chars < len(COMMAND):
        line = lines[-1] if lines else ""
        cursor_x = 66 + int(draw.textlength(line, font=FONT["mono"])) + 3
        draw.rectangle((cursor_x, y - 27, cursor_x + 9, y - 8), fill=COLORS["green"])
    return 226


def draw_terminal_output(draw: ImageDraw.ImageDraw, stage: int) -> None:
    y = 238
    if stage >= 1:
        text(draw, (66, y), "dataframe://default/profile", FONT["mono_small"], COLORS["cyan"])
        text(draw, (324, y), "91,872 rows | 11 columns", FONT["mono_small"], "#cbd5e1")
        y += 26
    if stage >= 2:
        rounded(draw, (64, y, 676, y + 78), 10, "#172033", "#334155")
        text(draw, (82, y + 13), "AnalysisPlan", FONT["label"], COLORS["green"])
        text(draw, (82, y + 36), "group_by: region_name", FONT["mono_small"], "#e5e7eb")
        text(
            draw,
            (322, y + 36),
            "metric: median(median_list_price)",
            FONT["mono_small"],
            "#e5e7eb",
        )
        y += 96
    if stage >= 3:
        text(draw, (66, y), "validated:", FONT["mono_small"], COLORS["green"])
        text(draw, (162, y), "columns, metric, sort, and row cap", FONT["mono_small"], "#cbd5e1")
        y += 30
    if stage >= 4:
        text(draw, (66, y), "Answer: Returned 10 rows", FONT["mono_bold"], COLORS["white"])
        y += 30
        text(draw, (66, y), "Top metros by median list price", FONT["mono_small"], COLORS["yellow"])
        y += 28
        for rank, name, amount in ROWS:
            text(draw, (84, y), f"{rank}.", FONT["mono_small"], "#94a3b8")
            text(draw, (124, y), name, FONT["mono_small"], COLORS["white"])
            text(draw, (394, y), amount, FONT["mono_small"], COLORS["green"])
            y += 24
        text(
            draw,
            (84, y),
            "... 7 more rows in structured table output",
            FONT["mono_small"],
            "#94a3b8",
        )
        y += 24
    if stage >= 5:
        y += 8
        text(
            draw,
            (66, y),
            "audit_id: qry_20260708_211915_d93541c1",
            FONT["mono_small"],
            COLORS["cyan"],
        )


def draw_architecture(draw: ImageDraw.ImageDraw, stage: int) -> None:
    rounded(draw, (732, 100, 1056, 580), 16, COLORS["card"], COLORS["card_border"], 2)
    text(draw, (758, 124), "Execution path", FONT["label"], COLORS["ink"])
    text(
        draw,
        (758, 146),
        "The LLM describes; the server validates.",
        FONT["small"],
        COLORS["muted"],
    )

    y = 186
    for index, (title, detail) in enumerate(STEPS, start=1):
        active = stage >= index
        fill = COLORS["soft_green"] if active else "#f8fafc"
        outline = COLORS["green"] if active else COLORS["line"]
        rounded(draw, (758, y, 1030, y + 54), 12, fill, outline, 2 if active else 1)
        circle_fill = COLORS["green"] if active else "#e2e8f0"
        draw.ellipse((774, y + 15, 798, y + 39), fill=circle_fill)
        text(
            draw,
            (782, y + 18),
            str(index),
            FONT["small"],
            COLORS["white"] if active else COLORS["muted"],
        )
        text(draw, (812, y + 11), title, FONT["label"], COLORS["ink"])
        text(draw, (812, y + 31), detail, FONT["small"], COLORS["muted"])
        if index < len(STEPS):
            draw.line((894, y + 56, 894, y + 70), fill=COLORS["line"], width=2)
        y += 72



def make_frame(stage: int, chars: int | None = None) -> Image.Image:
    image = Image.new("RGB", (WIDTH, HEIGHT), COLORS["bg"])
    draw = ImageDraw.Draw(image)
    draw_header(draw)
    draw_terminal_shell(draw)
    draw_command(draw, chars if chars is not None else len(COMMAND))
    draw_terminal_output(draw, stage)
    draw_architecture(draw, stage)
    return image


def main() -> None:
    OUT.parent.mkdir(parents=True, exist_ok=True)

    frames: list[Image.Image] = []
    durations: list[int] = []

    for chars in [0, 18, 39, 61, 82, len(COMMAND)]:
        frames.append(make_frame(0, chars))
        durations.append(130)

    for stage, duration in [(1, 500), (2, 650), (3, 650), (4, 1000), (5, 1800)]:
        frames.append(make_frame(stage))
        durations.append(duration)

    frames[0].save(
        OUT,
        save_all=True,
        append_images=frames[1:],
        duration=durations,
        loop=0,
        optimize=True,
    )
    print(OUT)


if __name__ == "__main__":
    main()
