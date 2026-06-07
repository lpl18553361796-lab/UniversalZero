import json
import os
from PIL import Image, ImageDraw, ImageFont


ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FIG_DIR = os.path.join(ROOT, "experiment_results", "paper_figures")


def font(size, bold=False):
    candidates = [
        "C:/Windows/Fonts/arialbd.ttf" if bold else "C:/Windows/Fonts/arial.ttf",
        "C:/Windows/Fonts/calibrib.ttf" if bold else "C:/Windows/Fonts/calibri.ttf",
    ]
    for path in candidates:
        if os.path.exists(path):
            return ImageFont.truetype(path, size=size)
    return ImageFont.load_default()


TITLE = font(42, bold=True)
SUBTITLE = font(28, bold=True)
TEXT = font(24)
SMALL = font(20)


def text_size(draw, text, fnt):
    box = draw.multiline_textbbox((0, 0), text, font=fnt, spacing=4)
    return box[2] - box[0], box[3] - box[1]


def draw_centered(draw, xy, text, fnt, fill=(35, 35, 35)):
    x, y = xy
    w, h = text_size(draw, text, fnt)
    draw.multiline_text((x - w / 2, y - h / 2), text, font=fnt, fill=fill, spacing=4, align="center")


def draw_panel(draw, box, title, labels, values, value_labels, colors, max_value=None):
    x0, y0, x1, y1 = box
    draw_centered(draw, ((x0 + x1) / 2, y0 + 24), title, SUBTITLE)

    plot_left = x0 + 95
    plot_top = y0 + 75
    plot_right = x1 - 45
    plot_bottom = y1 - 95
    axis_color = (90, 90, 90)
    grid_color = (225, 225, 225)

    max_y = max_value or max(values) * 1.15
    max_y = max(max_y, 1)

    for i in range(5):
        t = i / 4
        y = plot_bottom - t * (plot_bottom - plot_top)
        draw.line((plot_left, y, plot_right, y), fill=grid_color, width=2)
        tick = max_y * t
        tick_text = f"{tick:.0f}"
        tw, th = text_size(draw, tick_text, SMALL)
        draw.text((plot_left - tw - 12, y - th / 2), tick_text, font=SMALL, fill=(80, 80, 80))

    draw.line((plot_left, plot_top, plot_left, plot_bottom), fill=axis_color, width=3)
    draw.line((plot_left, plot_bottom, plot_right, plot_bottom), fill=axis_color, width=3)

    n = len(values)
    gap = 55
    bar_area = plot_right - plot_left
    bar_w = min(120, (bar_area - gap * (n + 1)) / n)

    for idx, (label, value, value_label, color) in enumerate(zip(labels, values, value_labels, colors)):
        cx = plot_left + gap + bar_w / 2 + idx * (bar_w + gap)
        bar_h = (value / max_y) * (plot_bottom - plot_top)
        bx0 = cx - bar_w / 2
        by0 = plot_bottom - bar_h
        bx1 = cx + bar_w / 2
        by1 = plot_bottom
        draw.rounded_rectangle((bx0, by0, bx1, by1), radius=8, fill=color)
        draw_centered(draw, (cx, by0 - 22), value_label, SMALL)
        draw_centered(draw, (cx, plot_bottom + 32), label, SMALL)


def save_chart(filename, title, win_labels, win_values, win_text, elo_values, elo_text, colors):
    img = Image.new("RGB", (2200, 1040), "white")
    draw = ImageDraw.Draw(img)
    draw_centered(draw, (1100, 62), title, TITLE)
    draw_panel(
        draw,
        (80, 120, 1060, 980),
        "Win Rate",
        win_labels,
        win_values,
        win_text,
        colors,
        max_value=100,
    )
    draw_panel(
        draw,
        (1140, 120, 2120, 980),
        "Estimated Elo",
        win_labels,
        elo_values,
        elo_text,
        colors,
    )

    png_path = os.path.join(FIG_DIR, filename + ".png")
    pdf_path = os.path.join(FIG_DIR, filename + ".pdf")
    img.save(png_path, "PNG", dpi=(300, 300))
    img.save(pdf_path, "PDF", resolution=300)
    return png_path, pdf_path


def save_single_benchmark_chart(filename, title, label, win_value, win_text, elo_value, elo_text, color):
    return save_chart(
        filename,
        title,
        [label],
        [win_value],
        [win_text],
        [elo_value],
        [elo_text],
        [color],
    )


def othello_data():
    report_path = os.path.join(ROOT, "experiment_results", "othello_expert_eval_robust.json")
    report = json.load(open(report_path, encoding="utf-8"))
    order = [
        ("vs_random", "Random"),
        ("vs_pure_mcts_50", "Pure MCTS-50"),
        ("vs_pure_mcts_100", "Pure MCTS-100"),
    ]
    labels, wins, elos, win_text, elo_text = [], [], [], [], []
    for key, label in order:
        item = report["benchmarks"][key]
        labels.append(label)
        wins.append(item["win_rate"] * 100)
        elos.append(item["elo_rating"])
        win_text.append(f"{item['win_rate'] * 100:.1f}%")
        elo_text.append(f"{item['elo_rating']:.1f}")
    return labels, wins, win_text, elos, elo_text


def hex_data():
    report_path = os.path.join(ROOT, "experiment_results", "hex_transfer_benefit_fixed_quick_eval.json")
    report = json.load(open(report_path, encoding="utf-8"))
    order = [
        ("transfer_vs_random", "Transfer\nvs Random"),
        ("scratch_vs_random", "Scratch\nvs Random"),
        ("transfer_vs_scratch", "Transfer\nvs Scratch"),
    ]
    labels, wins, elos, win_text, elo_text = [], [], [], [], []
    for key, label in order:
        item = report[key]
        labels.append(label)
        wins.append(item["win_rate"] * 100)
        elos.append(item["elo_rating"])
        win_text.append(f"{item['win_rate'] * 100:.1f}%")
        elo_text.append(f"{item['elo_rating']:.1f}")
    return labels, wins, win_text, elos, elo_text


def hex_report():
    report_path = os.path.join(ROOT, "experiment_results", "hex_transfer_benefit_fixed_quick_eval.json")
    return json.load(open(report_path, encoding="utf-8"))


def main():
    os.makedirs(FIG_DIR, exist_ok=True)
    othello_colors = [(47, 111, 143), (208, 140, 63), (63, 143, 95)]
    hex_colors = [(47, 111, 143), (208, 140, 63), (63, 143, 95)]

    paths = []
    paths.extend(save_chart(
        "fig_othello_credibility",
        "Othello Expert Model Credibility",
        *othello_data(),
        colors=othello_colors,
    ))
    paths.extend(save_chart(
        "fig_hex_transfer_benefit",
        "Hex Transfer Model Evaluation",
        *hex_data(),
        colors=hex_colors,
    ))
    report = hex_report()
    for path in paths:
        print(path)


if __name__ == "__main__":
    main()
