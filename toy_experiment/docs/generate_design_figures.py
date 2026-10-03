"""Regenerate the design's illustrative SVG diagrams using only Python."""
from html import escape
from math import cos, pi, sin
from pathlib import Path

DESTINATION = Path(__file__).parent
BLUE = '#245d9b'
ORANGE = '#a94f13'


def text(x, y, content, size=17, anchor='middle', color='#243447'):
    return f'<text x="{x}" y="{y}" font-size="{size}" text-anchor="{anchor}" fill="{color}">{escape(content)}</text>'


def arrow(x1, y1, x2, y2, color=BLUE):
    return f'<path d="M {x1} {y1} L {x2} {y2}" stroke="{color}" stroke-width="2" fill="none" marker-end="url(#arrow)"/>'


def box(x, y, width, height, lines):
    result = f'<rect x="{x}" y="{y}" width="{width}" height="{height}" rx="8" fill="#eef4fa" stroke="{BLUE}"/>'
    for i, line in enumerate(lines):
        result += text(x + width / 2, y + height / 2 + (i - (len(lines) - 1) / 2) * 23 + 6, line)
    return result


def save(name, width, height, title, parts):
    svg = f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {width} {height}" role="img" aria-labelledby="title"><title id="title">{escape(title)}</title>'
    svg += '<defs><marker id="arrow" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="6" markerHeight="6" orient="auto-start-reverse"><path d="M 0 0 L 10 5 L 0 10 z" fill="#245d9b"/></marker></defs>'
    svg += f'<rect width="{width}" height="{height}" fill="white"/><g font-family="Arial, sans-serif">' + ''.join(parts) + '</g></svg>\n'
    (DESTINATION / name).write_text(svg)


def data_diagram():
    parts = [text(250, 32, 'A. Draw a clean spiral point'), text(760, 32, 'B. Add noise outside the clean plane')]
    # Exact local curve, drawn with uniformly spaced parameter values for illustration.
    points = []
    for i in range(801):
        u = i / 800
        r, angle = 2 * u, 4 * pi * u
        points.append(f'{250 + 75*r*cos(angle):.3f},{210 - 75*r*sin(angle):.3f}')
    parts += [arrow(70, 210, 430, 210), arrow(250, 380, 250, 55), text(430, 237, 'coordinate 1'), text(275, 64, 'coordinate 2', 15, 'start')]
    parts += [f'<polyline points="{" ".join(points)}" fill="none" stroke="{BLUE}" stroke-width="2.5"/>', '<circle cx="212.5" cy="210" r="6" fill="#a94f13"/>', text(125, 184, 'u = 0.25', 17, color=ORANGE), text(250, 408, 'Clean point: (−0.5, 0). Two turns, radius limit 2.')]
    # Side view of the worked 3D case: all example points have coordinate 2 = 0.
    parts += [arrow(590, 340, 965, 340), arrow(865, 355, 865, 65), text(960, 368, 'coordinate 1'), text(888, 72, 'coordinate 3', 15, 'start')]
    parts += [text(720, 330, 'Clean plane: coordinate 3 = 0', 16), arrow(865, 140, 765, 340)]
    for x, y, label, tx, ty in [(765, 340, 'clean x = (−0.5, 0, 0)', 750, 393), (865, 140, 'noise ε = (0, 0, 1)', 745, 106), (840, 190, 'input z = (−0.125, 0, 0.75)', 752, 174)]:
        parts += [f'<circle cx="{x}" cy="{y}" r="6" fill="{ORANGE}"/>', text(tx, ty, label, 16)]
    parts += [text(760, 425, 'At t = 0.25, velocity v = (−0.5, 0, −1).', 17), text(510, 459, 'Illustration of exact formulas, not trained samples. Panel B shows the coordinate-2 = 0 slice.', 16)]
    save('data-path.svg', 1020, 485, 'Spiral construction and one noisy three-coordinate example', parts)


def model_diagram():
    parts = [text(550, 30, 'One network. Five Linear layers. Output meaning selected outside the network.')]
    positions = [20, 168, 316, 464, 612, 760, 908]
    labels = [['Noisy input z', 'and time t', 'D + 1 values'], ['Linear 1', 'D + 1 → 256', 'ReLU'], ['Linear 2', '256 → 256', 'ReLU'], ['Linear 3', '256 → 256', 'ReLU'], ['Linear 4', '256 → 256', 'ReLU'], ['Linear 5', '256 → D', 'No activation'], ['Raw output', 'D values', 'x̂, ε̂, or v̂']]
    for i, (x, lines) in enumerate(zip(positions, labels)):
        parts.append(box(x, 70, 130, 102, lines))
        if i < len(positions) - 1:
            parts.append(arrow(x + 130, 121, positions[i + 1], 121))
    parts += [text(550, 210, 'The model receives no projection matrix and no clean point. There is no input-to-output bypass.', 17)]
    save('model.svg', 1100, 245, 'The five-layer time-conditioned network', parts)


def workflow_diagram():
    parts = [text(550, 28, 'Training: compare velocity predictions with a known pair'), text(550, 258, 'Generation: advance a state with the learned velocity')]
    training = [(20, ['Clean batch x', 'time t, noise ε']), (235, ['Construct z', 'target v = x − ε']), (450, ['Model(z, t)', 'raw prediction']), (665, ['Convert to v̂', 'compare with v']), (880, ['Backward', 'optimizer step'])]
    sampling = [(20, ['Gaussian state', 'start at t = δ']), (235, ['Model(z, t)', 'same weights']), (450, ['Convert to v̂', 'same formula']), (665, ['Euler or Heun', 'advance z and t']), (880, ['Final samples', 'then evaluate'])]
    for y, row in [(65, training), (295, sampling)]:
        for i, (x, lines) in enumerate(row):
            parts.append(box(x, y, 190, 80, lines))
            if i < len(row)-1:
                parts.append(arrow(x+190, y+40, row[i+1][0], y+40))
    parts += ['<path d="M 975 145 V 188 H 545 V 145" stroke="#a94f13" stroke-width="2" stroke-dasharray="6 4" fill="none" marker-end="url(#arrow)"/>', text(760, 216, 'Updated weights. Start each new update at the left.', 16, color=ORANGE)]
    parts += ['<path d="M 760 375 V 411 H 330 V 375" stroke="#245d9b" stroke-width="2" fill="none" marker-end="url(#arrow)"/>', text(550, 443, 'Repeat until t = 1 − δ. No fresh noise or weight update inside this loop.', 16)]
    save('workflow.svg', 1100, 470, 'Training updates weights, while generation updates the sample state', parts)


if __name__ == '__main__':
    data_diagram()
    model_diagram()
    workflow_diagram()
