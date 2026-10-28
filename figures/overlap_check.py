"""Automatic layout check: reports text that touches other text, data lines, markers, bars, legends or panel boxes.

check(fig, name) is called by style.savefig before saving.  It works in display coordinates after a draw:
  * text vs text                      (tick labels, axis labels, titles, annotations, panel letters, legend texts)
  * text vs Line2D segments / markers (data curves, reference lines; spines and the text's own legend handles excluded)
  * text vs scatter points
  * text vs bar / box patches         (a text must lie fully inside or fully outside a patch)
  * legend box vs data                (lines, markers, scatter, bars inside the legend frame)
Images (sphere maps) and gridlines are ignored.  A small padding (PAD px) is added around text boxes.
"""
import numpy as np
from matplotlib.text import Text
from matplotlib.lines import Line2D
from matplotlib.collections import PathCollection, PolyCollection, LineCollection
from matplotlib.patches import Rectangle, FancyBboxPatch, Patch
from matplotlib.legend import Legend
from matplotlib.transforms import Bbox

PAD = 1.0
MARK_R = 2.0   # marker radius in px used for point tests (at check dpi)


def _tick_filter(fig):
    """ids of all tick-label Text objects, and of those actually drawn"""
    every, drawn = set(), set()
    for ax in fig.axes:
        for axis in (ax.xaxis, ax.yaxis):
            for tk in axis.get_major_ticks() + axis.get_minor_ticks():
                every.update((id(tk.label1), id(tk.label2)))
            if not ax.axison:
                continue
            try:
                for tk in axis._update_ticks():
                    for lb in (tk.label1, tk.label2):
                        if lb.get_visible():
                            drawn.add(id(lb))
            except Exception:
                pass
    return every, drawn


def _texts(fig, r):
    out = []
    every, drawn = _tick_filter(fig)
    for t in fig.findobj(Text):
        if id(t) in every and id(t) not in drawn:
            continue
        if not t.get_visible() or not t.get_text().strip():
            continue
        if t.axes is not None and not t.axes.get_visible():
            continue
        try:
            bb = Text.get_window_extent(t, r)
        except Exception:
            continue
        if bb.width < 1 or bb.height < 1:
            continue
        if any(o.get_text() == t.get_text() and abs(ob.x0 - bb.x0 + PAD) < 1 and abs(ob.y0 - bb.y0 + PAD) < 1 for o, ob in out):
            continue
        out.append((t, bb.padded(PAD)))
    return out


def _legend_of(fig):
    legs = []
    for ax in fig.axes:
        lg = ax.get_legend()
        if lg is not None and lg.get_visible():
            legs.append(lg)
    legs += [l for l in fig.legends if l.get_visible()]
    return legs


def _seg_hits_box(p, q, bb):
    """Liang-Barsky clip test: does segment p-q intersect bbox?"""
    x0, y0, x1, y1 = bb.x0, bb.y0, bb.x1, bb.y1
    dx, dy = q[0] - p[0], q[1] - p[1]
    t0, t1 = 0.0, 1.0
    for pp, qq in ((-dx, p[0] - x0), (dx, x1 - p[0]), (-dy, p[1] - y0), (dy, y1 - p[1])):
        if pp == 0:
            if qq < 0:
                return False
        else:
            t = qq / pp
            if pp < 0:
                if t > t1:
                    return False
                t0 = max(t0, t)
            else:
                if t < t0:
                    return False
                t1 = min(t1, t)
    return True


def _line_display(line):
    xy = line.get_xydata()
    if xy is None or len(xy) == 0:
        return None
    xy = np.asarray(xy, float)
    ok = np.all(np.isfinite(xy), 1)
    xy = line.get_transform().transform(np.where(ok[:, None], xy, 0.0))
    xy[~ok] = np.nan
    return xy


def _data_artists(ax):
    lines = [l for l in ax.get_lines() if l.get_visible() and l.get_label() != "_nolegend_grid"]
    scat = [c for c in ax.collections if isinstance(c, PathCollection) and c.get_visible()]
    lcol = [c for c in ax.collections if isinstance(c, LineCollection) and c.get_visible()]
    bars = [p for p in ax.patches if isinstance(p, (Rectangle, FancyBboxPatch)) and p.get_visible() and p is not ax.patch]
    return lines, scat, lcol, bars


def check(fig, name="", verbose=True):
    fig.canvas.draw()
    r = fig.canvas.get_renderer()
    texts = _texts(fig, r)
    legends = _legend_of(fig)
    leg_texts = {}
    for lg in legends:
        for t in lg.get_texts():
            leg_texts[id(t)] = lg
        if lg.get_title() is not None:
            leg_texts[id(lg.get_title())] = lg
    issues = []

    def label(t):
        s = t.get_text().replace("\n", " ")
        return (s[:28] + "...") if len(s) > 31 else s

    # ---- text vs text
    for i in range(len(texts)):
        ti, bi = texts[i]
        for j in range(i + 1, len(texts)):
            tj, bj = texts[j]
            if bi.overlaps(bj):
                # shrink padding back for the decision
                if ti.get_window_extent(r).overlaps(tj.get_window_extent(r)) or True:
                    ov = Bbox.intersection(bi, bj)
                    if ov is not None and ov.width > 1.5 and ov.height > 1.5:
                        issues.append(f"text/text   '{label(ti)}'  x  '{label(tj)}'")

    # ---- text vs data inside each axes (texts of any axes can hit data of any axes geometrically)
    for ax in fig.axes:
        if not ax.get_visible():
            continue
        if ax.name not in ("rectilinear",):
            continue
        lines, scat, lcol, bars = _data_artists(ax)
        axbb = ax.get_window_extent(r)
        for t, bb in texts:
            if not bb.overlaps(axbb):
                continue
            own_leg = leg_texts.get(id(t))
            # lines
            for ln in lines:
                if own_leg is not None and ln in getattr(own_leg, "legend_handles", []) + getattr(own_leg, "legendHandles", []):
                    continue
                xy = _line_display(ln)
                if xy is None:
                    continue
                hit = False
                if ln.get_linestyle() not in ("None", "", " ") and ln.get_linewidth() > 0:
                    for k in range(len(xy) - 1):
                        p, q = xy[k], xy[k + 1]
                        if np.any(np.isnan(p)) or np.any(np.isnan(q)):
                            continue
                        # clip segment to axes box (lines are clipped)
                        if _seg_hits_box(p, q, bb) and _seg_hits_box(p, q, axbb):
                            # make sure the intersection lies inside the axes
                            hit = True
                            break
                if not hit and ln.get_marker() not in (None, "None", "", " "):
                    ms = max(ln.get_markersize() * fig.dpi / 72 / 2, MARK_R)
                    inb = (xy[:, 0] > bb.x0 - ms) & (xy[:, 0] < bb.x1 + ms) & (xy[:, 1] > bb.y0 - ms) & (xy[:, 1] < bb.y1 + ms)
                    inax = (xy[:, 0] >= axbb.x0) & (xy[:, 0] <= axbb.x1) & (xy[:, 1] >= axbb.y0) & (xy[:, 1] <= axbb.y1)
                    hit = bool(np.any(inb & inax))
                if hit:
                    issues.append(f"text/line   '{label(t)}'  x  line '{ln.get_label()}' ({ax.get_label() or 'ax'})")
            # scatter
            for c in scat:
                off = c.get_offsets()
                if len(off) == 0:
                    continue
                xy = c.get_offset_transform().transform(off)
                inb = (xy[:, 0] > bb.x0 - MARK_R) & (xy[:, 0] < bb.x1 + MARK_R) & (xy[:, 1] > bb.y0 - MARK_R) & (xy[:, 1] < bb.y1 + MARK_R)
                inax = (xy[:, 0] >= axbb.x0) & (xy[:, 0] <= axbb.x1) & (xy[:, 1] >= axbb.y0) & (xy[:, 1] <= axbb.y1)
                if np.any(inb & inax):
                    issues.append(f"text/points '{label(t)}'  x  scatter ({int((inb & inax).sum())} pts)")
            # bars / boxes: partial overlap only
            for p in bars:
                pb = p.get_window_extent(r)
                if pb.width < 1 or pb.height < 1:
                    continue
                if own_leg is not None:
                    continue
                inter = Bbox.intersection(bb, pb)
                if inter is None or inter.width < 1.5 or inter.height < 1.5:
                    continue
                inside = (bb.x0 >= pb.x0 - 0.5 and bb.x1 <= pb.x1 + 0.5 and bb.y0 >= pb.y0 - 0.5 and bb.y1 <= pb.y1 + 0.5)
                if not inside:
                    issues.append(f"text/patch  '{label(t)}'  partly over a bar/box")

    # ---- legend frame vs data
    for lg in legends:
        lb = lg.get_window_extent(r)
        ax = lg.axes
        if ax is None or ax.name != "rectilinear":
            continue
        lines, scat, lcol, bars = _data_artists(ax)
        handles = getattr(lg, "legend_handles", []) or getattr(lg, "legendHandles", [])
        axbb = ax.get_window_extent(r)
        for ln in lines:
            if ln in handles:
                continue
            xy = _line_display(ln)
            if xy is None:
                continue
            hit = False
            if ln.get_linestyle() not in ("None", "", " ") and ln.get_linewidth() > 0:
                for k in range(len(xy) - 1):
                    p, q = xy[k], xy[k + 1]
                    if np.any(np.isnan(p)) or np.any(np.isnan(q)):
                        continue
                    if _seg_hits_box(p, q, lb):
                        hit = True
                        break
            if not hit and ln.get_marker() not in (None, "None", "", " "):
                inb = (xy[:, 0] > lb.x0) & (xy[:, 0] < lb.x1) & (xy[:, 1] > lb.y0) & (xy[:, 1] < lb.y1)
                hit = bool(np.any(inb))
            if hit:
                issues.append(f"legend/line legend of {ax.get_label() or 'ax'} x line '{ln.get_label()}'")
        for c in scat:
            off = c.get_offsets()
            if len(off) == 0:
                continue
            xy = c.get_offset_transform().transform(off)
            if np.any((xy[:, 0] > lb.x0) & (xy[:, 0] < lb.x1) & (xy[:, 1] > lb.y0) & (xy[:, 1] < lb.y1)):
                issues.append("legend/points")
        for p in bars:
            if p in handles or (p.get_alpha() is not None and p.get_alpha() <= 0.3):
                continue
            pb = p.get_window_extent(r)
            inter = Bbox.intersection(lb, pb)
            if inter is not None and inter.width > 1.5 and inter.height > 1.5:
                issues.append("legend/bar")
                break
    issues = sorted(set(issues))
    if verbose:
        print(f"[layout] {name}: {'OK' if not issues else str(len(issues)) + ' issue(s)'}")
        for s in issues:
            print("    ", s)
    return issues
