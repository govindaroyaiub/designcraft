//! Object › Effects › Add Product Shadow: a soft contact shadow under each selected product image,
//! shaped like the product's real base rather than a box or an oval.
//!
//! The product frame is rendered on its own with a transparent background (what the frame shows:
//! cropping, rotation and effects included). From the lowest opaque point a flood fill finds the
//! product itself, so other things in the frame can't count, and the bottom edge of every column
//! of it gives the base contour. The shadow is a lens along that contour, drawn behind the product:
//! Paper fill multiplied onto the page, with a soft drop shadow below. Like a designer's, it has
//! few points (10 at most by default): cubic curves are fitted to the contour (Schneider's
//! algorithm, as drawing tools fit pencil strokes) with the tolerance that keeps within the budget,
//! so it stays quick to adjust by hand. Without opaque pixels (an empty frame) it falls back to a
//! flat bar under the frame.

use std::collections::{HashSet, VecDeque};

use designcraft_color::BlendMode;
use designcraft_doc::{Document, Fill, Item, ItemId, Shape, SpreadRef, Stroke};
use designcraft_geom::{Affine, Anchor, PathData, Point, Rect, SubPath};
use serde_json::{Value, json};

use super::{CommandSpec, bad, cmd, has_selection};
use crate::{Result, Session};

/// The object style every shadow gets (created on first use: Paper fill, no stroke and the soft
/// drop shadow, so editing the style restyles every shadow).
pub const STYLE: &str = "Product Shadow";

/// Points on a shadow, by default and at most (`maxPoints`).
const POINTS: usize = 10;
const MAX_POINTS: usize = 40;
/// Pixels at least this opaque belong to the product.
const ALPHA_MIN: u8 = 128;
/// Rendering resolution for tracing (300 ppi), lowered for very large frames.
const PX_PER_PT: f64 = 300.0 / 72.0;
const MAX_SIDE_PX: f64 = 4000.0;
/// Shadow thickness: this fraction of the product's height, at most `HEIGHT_MAX` points.
const HEIGHT_RATIO: f64 = 0.02;
const HEIGHT_MAX: f64 = 3.0;
/// Moves the shadow's bottom edge (points; negative = up, under the product's edge).
const GAP: f64 = -1.0;
/// Narrows the shadow on each side (points) so it stays inside the product's silhouette.
const INSET: f64 = 1.0;
/// The bottom edge breaks into separate bases where it jumps by more than this (points): one
/// product standing in front of another.
const JUMP: f64 = 2.0;
/// A base reaches up its sides no higher than this fraction of its width (an ellipse in perspective).
const MAX_RISE: f64 = 0.25;
/// Curves follow the base to within this (points) when the point budget allows: closer is invisible
/// under a soft shadow, and every extra point is one more to adjust by hand.
const FIT: f64 = 0.75;
/// With several products selected, a base this much covered by things in front of its product
/// (0–1) gets no shadow: out of sight. A product picked on its own always gets its shadow.
const HIDDEN: f64 = 0.8;
/// Bases narrower than this (points) get no shadow.
const MIN_BASE: f64 = 6.0;
/// Fallback bar width as a fraction of the frame width.
const WIDTH_RATIO: f64 = 0.95;

pub fn specs() -> Vec<CommandSpec> {
    vec![cmd!(
        "object.addProductShadow",
        "Add Product Shadow",
        ["Object", "Effects"],
        None,
        "{ids?, maxPoints?: 4–40 (10)} — a contact shadow under each selected product image (or group holding one), traced from the product's base and placed behind it, with at most maxPoints points → {ids, traced}",
        has_selection,
        add_product_shadow
    )]
}

/// The bottom edge of the product in a rendering: `contour` holds (column, rows above `bottom`) for
/// every column of the product, left to right.
#[derive(Clone, Debug, PartialEq)]
pub(crate) struct Base {
    pub width: u32,
    pub height: u32,
    pub bottom: u32,
    pub contour: Vec<(u32, u32)>,
}

/// Trace the base of the product in straight or premultiplied RGBA pixels `w` × `h`: flood fill
/// (8-connected) from the middle of the widest opaque run in the lowest opaque row, then take the
/// lowest filled row of each column of the filled region. None without opaque pixels.
pub(crate) fn trace_base(rgba: &[u8], w: u32, h: u32) -> Option<Base> {
    let (wu, hu) = (usize::try_from(w).ok()?, usize::try_from(h).ok()?);
    let n = wu.checked_mul(hu)?;
    if wu == 0 || hu == 0 || rgba.len() < n.checked_mul(4)? {
        return None;
    }
    let opaque = |x: usize, y: usize| rgba.get((y * wu + x) * 4 + 3).is_some_and(|a| *a >= ALPHA_MIN);
    // Seed: the widest opaque run in the lowest row that has one.
    let mut seed = None;
    for y in (0..hu).rev() {
        let mut best: Option<(usize, usize)> = None;
        let mut x = 0;
        while x < wu {
            if !opaque(x, y) {
                x += 1;
                continue;
            }
            let start = x;
            while x < wu && opaque(x, y) {
                x += 1;
            }
            if best.is_none_or(|(a, b)| x - 1 - start > b - a) {
                best = Some((start, x - 1));
            }
        }
        if let Some((a, b)) = best {
            seed = Some(((a + b) / 2, y));
            break;
        }
    }
    let (sx, sy) = seed?;
    let mut visited = vec![false; n];
    let mut column_bottom: Vec<Option<usize>> = vec![None; wu];
    let mut queue = VecDeque::from([(sx, sy)]);
    visited[sy * wu + sx] = true;
    while let Some((x, y)) = queue.pop_front() {
        if let Some(c) = column_bottom.get_mut(x) {
            *c = Some(c.map_or(y, |b| b.max(y)));
        }
        for (dx, dy) in [(1i64, 0i64), (-1, 0), (0, 1), (0, -1), (1, 1), (1, -1), (-1, 1), (-1, -1)] {
            let (nx, ny) = (x as i64 + dx, y as i64 + dy);
            if nx < 0 || ny < 0 || nx >= wu as i64 || ny >= hu as i64 {
                continue;
            }
            let (nx, ny) = (nx as usize, ny as usize);
            let i = ny * wu + nx;
            if !visited[i] && opaque(nx, ny) {
                visited[i] = true;
                queue.push_back((nx, ny));
            }
        }
    }
    let bottom = column_bottom.iter().flatten().copied().max()?;
    let left = column_bottom.iter().position(Option::is_some)?;
    let right = column_bottom.iter().rposition(Option::is_some)?;
    // A column inside the region without filled pixels (a gap) takes the nearest filled one.
    let nearest = |x: usize| -> usize {
        for d in 0..=right - left {
            for c in [x.checked_sub(d), x.checked_add(d)].into_iter().flatten() {
                if (left..=right).contains(&c)
                    && let Some(Some(b)) = column_bottom.get(c)
                {
                    return *b;
                }
            }
        }
        bottom
    };
    let contour = (left..=right).map(|x| (x as u32, (bottom - nearest(x)) as u32)).collect();
    Some(Base { width: w, height: h, bottom: bottom as u32, contour })
}

/// One cubic Bézier segment: start, its handle, the end's handle, end.
type Cubic = [Point; 4];

fn bez(c: &Cubic, t: f64) -> Point {
    let mt = 1.0 - t;
    let (a, b, cc, d) = (mt * mt * mt, 3.0 * mt * mt * t, 3.0 * mt * t * t, t * t * t);
    Point::new(a * c[0].x + b * c[1].x + cc * c[2].x + d * c[3].x, a * c[0].y + b * c[1].y + cc * c[2].y + d * c[3].y)
}

fn unit(x: f64, y: f64) -> (f64, f64) {
    let l = x.hypot(y);
    if l > 1e-12 { (x / l, y / l) } else { (0.0, 0.0) }
}

/// The cubic from `pts[0]` to its last point with end tangents `t0` and `t1` (unit vectors pointing
/// into the curve) that best fits `pts` at parameters `u`, by least squares (Schneider, Graphics Gems).
fn fit_one(pts: &[Point], u: &[f64], t0: (f64, f64), t1: (f64, f64)) -> Cubic {
    let (p0, p3) = (pts[0], pts[pts.len() - 1]);
    let (mut c00, mut c01, mut c11, mut x0, mut x1) = (0.0, 0.0, 0.0, 0.0, 0.0);
    for (p, &t) in pts.iter().zip(u) {
        let mt = 1.0 - t;
        let (b0, b1, b2, b3) = (mt * mt * mt, 3.0 * mt * mt * t, 3.0 * mt * t * t, t * t * t);
        let a0 = (t0.0 * b1, t0.1 * b1);
        let a1 = (t1.0 * b2, t1.1 * b2);
        c00 += a0.0 * a0.0 + a0.1 * a0.1;
        c01 += a0.0 * a1.0 + a0.1 * a1.1;
        c11 += a1.0 * a1.0 + a1.1 * a1.1;
        let tmp = (p.x - (p0.x * (b0 + b1) + p3.x * (b2 + b3)), p.y - (p0.y * (b0 + b1) + p3.y * (b2 + b3)));
        x0 += a0.0 * tmp.0 + a0.1 * tmp.1;
        x1 += a1.0 * tmp.0 + a1.1 * tmp.1;
    }
    let det = c00 * c11 - c01 * c01;
    let seg = p0.distance(p3);
    let (mut al, mut ar) = if det.abs() > 1e-12 { ((x0 * c11 - x1 * c01) / det, (c00 * x1 - c01 * x0) / det) } else { (0.0, 0.0) };
    // Degenerate or backwards handles: a third of the chord, as the heuristic fallback.
    if !(al > seg * 1e-6 && ar > seg * 1e-6 && al.is_finite() && ar.is_finite()) {
        (al, ar) = (seg / 3.0, seg / 3.0);
    }
    [p0, Point::new(p0.x + t0.0 * al, p0.y + t0.1 * al), Point::new(p3.x + t1.0 * ar, p3.y + t1.1 * ar), p3]
}

/// Cubics through `pts` (in order) staying within `tol` of every point, split where they don't:
/// Schneider's curve fitting, iterative. The segments share tangents, so inner joins are smooth.
fn fit_curve(pts: &[Point], tol: f64) -> Vec<Cubic> {
    let n = pts.len();
    if n < 2 {
        return Vec::new();
    }
    let tangent = |a: Point, b: Point| unit(b.x - a.x, b.y - a.y);
    let mut out = Vec::new();
    // (range, start tangent, end tangent), worked off in order.
    let mut todo = vec![(0usize, n - 1, tangent(pts[0], pts[1]), tangent(pts[n - 1], pts[n - 2]))];
    while let Some((a, b, t0, t1)) = todo.pop() {
        let seg = &pts[a..=b];
        if seg.len() <= 2 {
            let d = seg[0].distance(seg[seg.len() - 1]) / 3.0;
            out.push([
                seg[0],
                Point::new(seg[0].x + t0.0 * d, seg[0].y + t0.1 * d),
                Point::new(seg[seg.len() - 1].x + t1.0 * d, seg[seg.len() - 1].y + t1.1 * d),
                seg[seg.len() - 1],
            ]);
            continue;
        }
        // Chord-length parameters.
        let mut u = vec![0.0; seg.len()];
        for k in 1..seg.len() {
            u[k] = u[k - 1] + seg[k].distance(seg[k - 1]);
        }
        let total = u[seg.len() - 1].max(1e-12);
        u.iter_mut().for_each(|v| *v /= total);
        let c = fit_one(seg, &u, t0, t1);
        let (worst, err) =
            seg.iter().zip(&u).enumerate().map(|(k, (p, t))| (k, bez(&c, *t).distance(*p))).fold((0, 0.0), |m, x| if x.1 > m.1 { x } else { m });
        if err <= tol || b - a < 3 {
            out.push(c);
            continue;
        }
        let k = worst.clamp(1, seg.len() - 2) + a;
        let mid = tangent(pts[k + 1], pts[k - 1]);
        // Right part first on the stack's top after the left: pushed in reverse order.
        todo.push((k, b, (-mid.0, -mid.1), t1));
        todo.push((a, k, t0, mid));
    }
    out
}

/// The fewest cubics fitting `pts` to within [`FIT`]; if that takes more than `max`, the `max`
/// that fit best (the smallest tolerance that gets there).
fn fit_within(pts: &[Point], max: usize) -> Vec<Cubic> {
    // A few hundred points are plenty for a curve made of a handful of segments.
    let step = pts.len().div_ceil(400).max(1);
    let mut p: Vec<Point> = pts.iter().step_by(step).copied().collect();
    if let (Some(last), Some(end)) = (p.last(), pts.last())
        && last != end
    {
        p.push(*end);
    }
    let good = fit_curve(&p, FIT);
    if good.len() <= max {
        return good;
    }
    let span = p.iter().fold(0.0f64, |m, q| m.max(q.distance(p[0]))).max(1e-6);
    let (mut lo, mut hi) = (FIT, span);
    let mut best = fit_curve(&p, hi);
    for _ in 0..30 {
        let mid = (lo + hi) / 2.0;
        let f = fit_curve(&p, mid);
        if f.len() <= max {
            (hi, best) = (mid, f);
        } else {
            lo = mid;
        }
    }
    best
}

/// The bases in `b`, one per product standing in it: the bottom edge splits where it jumps by more
/// than `jump` pixels; in each piece, the stretch around its lowest point that rises no more than
/// [`MAX_RISE`] of its width (plus `jump`, for the wobble of a real edge) is the base (the rest
/// climbs the product's sides). Pieces narrower than
/// `min` pixels are left out. Each comes back as a [`Base`] of the same rendering.
pub(crate) fn bases(b: &Base, jump: f64, min: f64) -> Vec<Base> {
    let mut pieces: Vec<&[(u32, u32)]> = Vec::new();
    let mut start = 0;
    for k in 1..=b.contour.len() {
        let breaks =
            k == b.contour.len() || (b.contour[k].1 as f64 - b.contour[k - 1].1 as f64).abs() > jump || b.contour[k].0 != b.contour[k - 1].0 + 1;
        if breaks {
            pieces.push(&b.contour[start..k]);
            start = k;
        }
    }
    let mut out = Vec::new();
    for piece in pieces {
        let Some(low) = piece.iter().map(|c| c.1).min() else { continue };
        let Some(at) = piece.iter().position(|c| c.1 == low) else { continue };
        // Grow around the lowest point while the rise stays within MAX_RISE of the width so far.
        let (mut a, mut z) = (at, at);
        loop {
            let width = (z - a + 1) as f64;
            let ok = |k: usize| (piece[k].1 - low) as f64 <= MAX_RISE * (width + 1.0) + jump;
            let left = a > 0 && ok(a - 1);
            let right = z + 1 < piece.len() && ok(z + 1);
            if !left && !right {
                break;
            }
            if left {
                a -= 1;
            }
            if right {
                z += 1;
            }
        }
        if ((z - a + 1) as f64) < min {
            continue;
        }
        // Offsets from this base's own lowest point.
        let bottom = b.bottom - low;
        out.push(Base { width: b.width, height: b.height, bottom, contour: piece[a..=z].iter().map(|(x, off)| (*x, off - low)).collect() });
    }
    out
}

/// The shadow outline (spread coordinates) for a product framed by `r`, whose rendering has the
/// traced `base` (None: the fallback bar), in at most `max_points` points (≥ 4). The bottom edge is
/// curves fitted to the base contour, smooth where they meet; the top edge joins the two ends
/// raised by the shadow's thickness (it lies behind the product, out of sight).
pub(crate) fn shadow_outline(r: Rect, base: Option<&Base>, max_points: usize) -> Vec<Anchor> {
    let h = (r.height() * HEIGHT_RATIO).min(HEIGHT_MAX);
    let Some(b) = base.filter(|b| b.width > 0 && b.height > 0 && !b.contour.is_empty()) else {
        let inset = (r.width() - r.width() * WIDTH_RATIO) / 2.0 + INSET;
        let (x0, x1, y1) = (r.x0 + inset, r.x1 - inset, r.y1 + GAP);
        return [Point::new(x0, y1 - h), Point::new(x1, y1 - h), Point::new(x1, y1), Point::new(x0, y1)].into_iter().map(Anchor::corner).collect();
    };
    let (kx, ky) = (r.width() / b.width as f64, r.height() / b.height as f64);
    // Pixel centres, so a one-column product still has width.
    let xs: Vec<f64> = b.contour.iter().map(|(x, _)| r.x0 + (*x as f64 + 0.5) * kx).collect();
    let ys: Vec<f64> = b.contour.iter().map(|(_, off)| r.y0 + (b.bottom as f64 + 1.0 - *off as f64) * ky + GAP).collect();
    // Narrow about the centre, keeping the shape: INSET in from each side.
    let (lo, hi) = xs.iter().fold((f64::INFINITY, f64::NEG_INFINITY), |(lo, hi), x| (lo.min(*x), hi.max(*x)));
    let width = hi - lo;
    let (k, cx) = if width > 2.0 * INSET { ((width - 2.0 * INSET) / width, (lo + hi) / 2.0) } else { (1.0, 0.0) };
    let bottom: Vec<Point> = xs.iter().zip(&ys).map(|(x, y)| Point::new(if k < 1.0 { cx + (x - cx) * k } else { *x }, *y)).collect();
    // Two points for the top edge; the rest follow the base, right to left as the outline runs.
    let mut bottom = bottom;
    bottom.reverse();
    let (Some(&right), Some(&left)) = (bottom.first(), bottom.last()) else { return Vec::new() };
    let mut out = vec![Anchor::corner(Point::new(left.x, left.y - h)), Anchor::corner(Point::new(right.x, right.y - h))];
    // A flat base is one straight segment; anything else, as few curves as the budget allows.
    let flat = bottom.iter().all(|p| (p.y - right.y).abs() < 1e-9);
    let curves = if flat || bottom.len() < 2 { Vec::new() } else { fit_within(&bottom, max_points.max(4) - 3) };
    if curves.is_empty() {
        out.push(Anchor::corner(right));
        out.push(Anchor::corner(left));
        return out;
    }
    // Anchor k joins curve k-1 (its end handle comes in) and curve k (its start handle goes out).
    for k in 0..=curves.len() {
        let p = if k < curves.len() { curves[k][0] } else { curves[k - 1][3] };
        let h_in = if k > 0 { curves[k - 1][2] } else { p };
        let h_out = if k < curves.len() { curves[k][1] } else { p };
        out.push(Anchor::with_handles(p, h_in, h_out));
    }
    out
}

/// The name a shadow takes from its product: `<name>_shadow`, where the name is the image frame's
/// own (Layers panel) name, else its placed file's name without the extension, else "Product".
fn shadow_name(d: &Document, frame: ItemId) -> String {
    let Some(it) = d.item(frame) else { return "Product_shadow".into() };
    let file = it.graphic().and_then(|g| d.assets.get(&g.asset)).map(|a| a.name.as_str()).unwrap_or("");
    let stem = std::path::Path::new(file).file_stem().and_then(|s| s.to_str()).unwrap_or("");
    let name = [clean_name(&it.name), stem.trim()].into_iter().find(|n| !n.is_empty()).unwrap_or("Product");
    format!("{name}_shadow")
}

/// A frame name as a shadow name: without a short label prefix such as `rl:` (as production
/// tools name frames after their image) or the image file's extension.
fn clean_name(name: &str) -> &str {
    let mut n = name.trim();
    if let Some((prefix, rest)) = n.split_once(':')
        && prefix.len() <= 4
        && prefix.chars().all(|c| c.is_ascii_alphanumeric())
    {
        n = rest.trim();
    }
    const EXTENSIONS: [&str; 10] = [".psd", ".png", ".jpg", ".jpeg", ".tif", ".tiff", ".ai", ".pdf", ".eps", ".svg"];
    for ext in EXTENSIONS {
        if n.len() > ext.len() && n.to_ascii_lowercase().ends_with(ext) {
            return &n[..n.len() - ext.len()];
        }
    }
    n
}

/// The tilt (radians, within ±45°) at which an image transformed by `img` stands on the page: its
/// rotation less any whole quarter turns.
fn standing_tilt(img: Affine) -> f64 {
    let c = img.as_coeffs();
    let theta = c[1].atan2(c[0]);
    if !theta.is_finite() {
        return 0.0;
    }
    let q = std::f64::consts::FRAC_PI_2;
    theta - (theta / q).round() * q
}

/// The shadow's effects: a soft drop shadow straight down (black, 55 %, 2.83 pt away, 3.4 pt soft).
fn shadow_effects() -> designcraft_doc::Effects {
    let mut e = designcraft_doc::Effects::default();
    let ds = &mut e.drop_shadow;
    (ds.on, ds.opacity, ds.angle, ds.global_light) = (true, 0.55, 90.0, false);
    (ds.distance, ds.size, ds.spread) = (2.834645669291339, 3.401574803149606, 0.0);
    e
}

/// The image frame `id` is or holds (the first one, depth first).
fn graphic_frame(d: &Document, id: ItemId) -> Option<ItemId> {
    fn find(it: &Item) -> Option<ItemId> {
        if it.graphic().is_some() {
            return Some(it.id);
        }
        it.children().iter().find_map(|c| find(c))
    }
    find(d.item(id)?)
}

/// Every item id in the document's spreads and parent spreads, at any depth.
fn all_ids(d: &Document) -> Vec<ItemId> {
    fn walk(items: &[std::sync::Arc<Item>], out: &mut Vec<ItemId>) {
        for it in items {
            out.push(it.id);
            walk(it.children(), out);
        }
    }
    let mut out = Vec::new();
    for sp in d.spreads.iter().chain(d.parents.iter()) {
        walk(&sp.items, &mut out);
    }
    out
}

/// Render frame `frame` alone (its group's other items hidden) upright in its own space: over its
/// inner bounds `r`, with `xf` (inner → spread) undone, so a rotated product stands straight and
/// its base is at the bottom. Transparent background.
fn render_alone(s: &Session, sr: SpreadRef, frame: ItemId, xf: Affine, r: Rect) -> Option<(Vec<u8>, u32, u32)> {
    let d = &s.doc().ok()?.doc;
    if xf.determinant().abs() < 1e-12 {
        return None;
    }
    let scale = PX_PER_PT.min(MAX_SIDE_PX / r.width().max(r.height()).max(1e-6));
    let (w, h) = ((r.width() * scale).round().max(1.0), (r.height() * scale).round().max(1.0));
    let keep = alone(d, frame)?;
    render_kept(s, sr, keep, view_of(xf, r, scale), w as u32, h as u32)
}

/// What draws the frame alone: the frame, the groups holding it and what it holds.
fn alone(d: &Document, frame: ItemId) -> Option<HashSet<ItemId>> {
    let mut keep: HashSet<ItemId> = HashSet::new();
    let loc = d.find(frame)?;
    let mut items: &[std::sync::Arc<Item>] = &d.spread(loc.spread)?.items;
    for &i in &loc.path {
        let it = items.get(i)?;
        keep.insert(it.id);
        items = it.children();
    }
    descendants(d.item(frame)?, &mut keep);
    Some(keep)
}

fn descendants(it: &Item, keep: &mut HashSet<ItemId>) {
    for c in it.children() {
        keep.insert(c.id);
        descendants(c, keep);
    }
}

/// What lies in front of top-level item `top` of spread `sr`: items on layers above its layer, and
/// later items on its own layer (with what they hold).
fn in_front(d: &Document, sr: SpreadRef, top: ItemId) -> Option<HashSet<ItemId>> {
    let sp = d.spread(sr)?;
    let at = sp.items.iter().position(|i| i.id == top)?;
    // Layers draw back to front from the end of the list: a lower index is further in front.
    let rank = |l: designcraft_doc::LayerId| d.layers.iter().position(|x| x.id == l).unwrap_or(usize::MAX);
    let mine = rank(sp.items[at].layer);
    let mut keep = HashSet::new();
    for (k, it) in sp.items.iter().enumerate() {
        let r = rank(it.layer);
        if r < mine || (r == mine && k > at) {
            keep.insert(it.id);
            descendants(it, &mut keep);
        }
    }
    Some(keep)
}

/// Spread → upright frame pixels: `r` (the frame's inner bounds) at `scale` px/pt, `xf` undone.
fn view_of(xf: Affine, r: Rect, scale: f64) -> Affine {
    Affine::scale(scale) * Affine::translate((-r.x0, -r.y0)) * xf.inverse()
}

/// Render only the items `keep` of spread `sr` through `view`, `w` × `h`, transparent.
fn render_kept(s: &Session, sr: SpreadRef, keep: HashSet<ItemId>, view: Affine, w: u32, h: u32) -> Option<(Vec<u8>, u32, u32)> {
    let d = &s.doc().ok()?.doc;
    let hidden = all_ids(d).into_iter().filter(|id| !keep.contains(id)).collect();
    let opts = designcraft_render::RenderOptions { background: None, paper: false, hidden, ..Default::default() };
    let mut rr = designcraft_render::Renderer::new();
    let img = rr.render(d, &s.cache, &[designcraft_render::Placed { spread: sr, xf: Affine::IDENTITY }], w, h, view, &opts);
    Some((img.pixels, img.width, img.height))
}

/// How much of `base`'s bottom edge (0–1) is covered in `front` (RGBA, the base's rendering size):
/// sampled just inside the product along the edge.
fn covered(base: &Base, front: &[u8]) -> f64 {
    let w = base.width as usize;
    let n = base.contour.len().max(1) as f64;
    let hits = base
        .contour
        .iter()
        .filter(|(x, off)| {
            let y = (base.bottom.saturating_sub(*off)).saturating_sub(1) as usize;
            front.get((y * w + *x as usize) * 4 + 3).is_some_and(|a| *a >= ALPHA_MIN)
        })
        .count();
    hits as f64 / n
}

fn add_product_shadow(s: &mut Session, p: &Value) -> Result<Value> {
    let ids = super::targets(s, p)?;
    // One product picked on its own is shadowed whatever stands in front of it.
    let skip_hidden = ids.len() > 1;
    let max_points = match p.get("maxPoints") {
        None | Some(Value::Null) => POINTS,
        Some(v) => v
            .as_u64()
            .and_then(|n| usize::try_from(n).ok())
            .filter(|n| (4..=MAX_POINTS).contains(n))
            .ok_or_else(|| bad("object.addProductShadow", format!("maxPoints must be a whole number from 4 to {MAX_POINTS}")))?,
    };
    // For each selected product: where it is, its image frame's bounds and the traced base.
    struct Plan {
        top: ItemId,
        /// What the shadows are called (`<product>_shadow`).
        name: String,
        /// The image frame's inner bounds and its inner → spread transform.
        bounds: Rect,
        xf: Affine,
        /// Whether the product could be traced; its bases (those out of sight left out).
        traced: bool,
        bases: Vec<Base>,
    }
    let mut plans = Vec::new();
    {
        let d = &s.doc()?.doc;
        for id in &ids {
            let Some(frame) = graphic_frame(d, *id) else { continue };
            let Some(loc) = d.find(frame) else { continue };
            let Some(it) = d.item_at(&loc) else { continue };
            // Traced upright as the product stands on the page: its image's rotation without whole
            // quarter turns (a photo stored sideways and turned upright in the layout), so a
            // product tilted with the layout keeps its tilt and its base is its page bottom.
            let full = d.parent_xf(&loc) * it.xf;
            let xf = Affine::rotate(standing_tilt(it.graphic().map_or(full, |g| full * g.xf)));
            let bounds = (xf.inverse() * full).transform_rect_bbox(it.inner_bounds());
            if !(bounds.width() > 0.0 && bounds.height() > 0.0 && bounds.width().is_finite() && bounds.height().is_finite()) {
                continue;
            }
            let base = render_alone(s, loc.spread, frame, xf, bounds).and_then(|(px, w, h)| trace_base(&px, w, h));
            // One shadow per product base (pixels per point from the rendering), unless what stands
            // in front of the product hides it.
            let bases = match &base {
                Some(b) => {
                    let ppt = b.width as f64 / bounds.width();
                    let found = bases(b, JUMP * ppt, MIN_BASE * ppt);
                    let top = d.top_level_of(frame).unwrap_or(frame);
                    let front = in_front(d, loc.spread, top)
                        .filter(|k| skip_hidden && !k.is_empty())
                        .and_then(|k| render_kept(s, loc.spread, k, view_of(xf, bounds, ppt), b.width, b.height));
                    match front {
                        Some((px, _, _)) => found.into_iter().filter(|b| covered(b, &px) < HIDDEN).collect(),
                        None => found,
                    }
                }
                None => Vec::new(),
            };
            plans.push(Plan { top: *id, name: shadow_name(d, frame), bounds, xf, traced: base.is_some(), bases });
        }
    }
    if plans.is_empty() {
        return Err(bad("object.addProductShadow", "select one or more product images (or groups holding one)"));
    }
    let traced = plans.iter().filter(|p| p.traced).count();
    s.edit(move |d, sel| {
        if d.styles.object_style(STYLE).is_none() {
            std::sync::Arc::make_mut(&mut d.styles).object.push(designcraft_doc::ObjectStyle {
                name: STYLE.into(),
                based_on: None,
                fill: Some(Fill::swatch(designcraft_color::swatch::PAPER)),
                stroke: Some(Stroke::none()),
                paragraph_style: None,
                text_frame: None,
                effects: Some(shadow_effects()),
            });
        }
        let mut made = Vec::new();
        for plan in &plans {
            let Some(loc) = d.find(plan.top) else { continue };
            let Some(layer) = d.item_at(&loc).map(|it| it.layer) else { continue };
            // A shadow per visible base; nothing traceable (nothing opaque): the fallback bar.
            let outlines: Vec<Vec<Anchor>> = if !plan.traced {
                vec![shadow_outline(plan.bounds, None, max_points)]
            } else {
                plan.bases.iter().map(|b| shadow_outline(plan.bounds, Some(b), max_points)).collect()
            };
            for (n, outline) in outlines.into_iter().filter(|o| o.len() >= 3).enumerate() {
                // Back into spread space, turned with the product.
                let outline: Vec<Anchor> = outline.iter().map(|a| a.transform(plan.xf)).collect();
                let mut it = Item::new(ItemId(d.alloc()), layer, Shape::Polygon, PathData::single(SubPath::new(outline, true)));
                // A second base of the same image: `…_shadow_2`.
                it.name = if n == 0 { plan.name.clone() } else { format!("{}_{}", plan.name, n + 1) };
                it.object_style = STYLE.into();
                it.fill = Fill::swatch(designcraft_color::swatch::PAPER);
                it.stroke = Stroke::none();
                it.blend = BlendMode::Multiply;
                // The style's effects (an existing style's, as the user may have changed them).
                it.effects = d.styles.object_style(STYLE).and_then(|os| os.effects.clone()).unwrap_or_else(shadow_effects);
                // Behind the product, in its spread.
                let at = d.find(plan.top).map(|l| l.top()).unwrap_or(loc.top());
                made.push(d.insert_item(loc.spread, it, Some(at))?);
            }
        }
        *sel = designcraft_doc::Selection::items(made.clone());
        Ok(json!({"ids": made.iter().map(|i| i.0).collect::<Vec<_>>(), "traced": traced}))
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    /// An RGBA image `w` × `h`, opaque where `f(x, y)`.
    fn image(w: u32, h: u32, f: impl Fn(u32, u32) -> bool) -> Vec<u8> {
        (0..h).flat_map(|y| (0..w).map(move |x| (x, y))).flat_map(|(x, y)| if f(x, y) { [0, 0, 0, 255] } else { [0, 0, 0, 0] }).collect()
    }

    #[test]
    fn a_flat_box_has_a_flat_base() {
        let px = image(40, 30, |x, y| (5..35).contains(&x) && (2..25).contains(&y));
        let b = trace_base(&px, 40, 30).unwrap();
        assert_eq!(b.bottom, 24);
        assert_eq!(b.contour.first().map(|c| c.0), Some(5));
        assert_eq!(b.contour.last().map(|c| c.0), Some(34));
        assert!(b.contour.iter().all(|c| c.1 == 0), "{:?}", b.contour);
    }

    #[test]
    fn a_round_bottle_curves_up_at_its_sides() {
        // A disc: the base is lowest in the middle and rises towards the edges.
        let px = image(41, 41, |x, y| (x as f64 - 20.0).powi(2) + (y as f64 - 20.0).powi(2) <= 400.0);
        let b = trace_base(&px, 41, 41).unwrap();
        let (n, mid) = (b.contour.len(), b.contour[b.contour.len() / 2].1);
        assert_eq!(n, 41, "every column");
        assert!(b.contour[0].1 > mid + 5 && b.contour[n - 1].1 > mid + 5, "{:?}", b.contour);
    }

    #[test]
    fn something_else_in_the_frame_does_not_count() {
        // The product, and a speck above and to the right that touches nothing.
        let px = image(60, 40, |x, y| (5..30).contains(&x) && (5..35).contains(&y) || (50..58).contains(&x) && (2..6).contains(&y));
        let b = trace_base(&px, 60, 40).unwrap();
        assert_eq!(b.contour.last().map(|c| c.0), Some(29), "the speck is not part of the base");
    }

    #[test]
    fn nothing_opaque_traces_nothing() {
        assert!(trace_base(&image(10, 10, |_, _| false), 10, 10).is_none());
        assert!(trace_base(&[], 10, 10).is_none(), "too few bytes");
    }

    #[test]
    fn products_in_one_image_get_a_base_each_and_sides_are_left_out() {
        // Two products side by side, the right one standing higher (further back); the left one's
        // outer columns climb its side (a tilted can).
        let mut contour: Vec<(u32, u32)> = (0..60).map(|x| (x, if x < 5 { 40 - x * 4 } else { 0 })).collect();
        contour.extend((60..100).map(|x| (x, 20)));
        let b = Base { width: 100, height: 100, bottom: 99, contour };
        let found = bases(&b, 3.0, 6.0);
        assert_eq!(found.len(), 2, "{found:?}");
        assert_eq!(found[0].contour.first().map(|c| c.0), Some(5), "the side is not part of the base");
        assert_eq!((found[1].contour.first().map(|c| c.0), found[1].bottom), (Some(60), 79), "the back product's own base");
        // A flat base with a few pixels of wobble (a product cut off by its frame) is one base.
        let noisy: Vec<(u32, u32)> = (0..200).map(|x| (x, [0, 1, 3, 2, 1][x as usize % 5])).collect();
        let b = Base { width: 200, height: 100, bottom: 99, contour: noisy };
        let found = bases(&b, 8.0, 25.0);
        assert_eq!(found.iter().map(|b| b.contour.len()).collect::<Vec<_>>(), vec![200], "the whole width");
        // A sliver is too small for a shadow.
        let b = Base { width: 100, height: 100, bottom: 99, contour: (0..4).map(|x| (x, 0)).collect() };
        assert!(bases(&b, 3.0, 6.0).is_empty());
    }

    #[test]
    fn shadow_names_drop_label_prefixes_and_extensions() {
        assert_eq!(clean_name("rl:FKZRWGYXE3J.psd"), "FKZRWGYXE3J");
        assert_eq!(clean_name("Hero bottle"), "Hero bottle");
        assert_eq!(clean_name("Bottle.PNG"), "Bottle");
        assert_eq!(clean_name("Note: keep"), "keep");
        assert_eq!(clean_name("A long label: kept"), "A long label: kept", "only a short prefix is a label");
    }

    #[test]
    fn a_flat_base_is_a_four_point_bar_inside_the_product() {
        let r = Rect::new(100.0, 100.0, 200.0, 300.0);
        let b = Base { width: 100, height: 200, bottom: 190, contour: (10..80).map(|x| (x, 0)).collect() };
        let pts = shadow_outline(r, Some(&b), POINTS);
        assert_eq!(pts.len(), 4, "a box needs no more");
        assert!(pts.iter().all(|a| !a.has_in() && !a.has_out()), "straight edges");
        let (lo, hi) = pts.iter().fold((f64::INFINITY, f64::NEG_INFINITY), |(lo, hi), a| (lo.min(a.p.x), hi.max(a.p.x)));
        assert!((lo - (110.5 + INSET)).abs() < 1e-9 && (hi - (179.5 - INSET)).abs() < 1e-9, "{lo} {hi}");
        let bottom = pts.iter().map(|a| a.p.y).fold(f64::NEG_INFINITY, f64::max);
        let top = pts.iter().map(|a| a.p.y).fold(f64::INFINITY, f64::min);
        assert!((bottom - (100.0 + 191.0 + GAP)).abs() < 1e-9, "on the base, {GAP} pt up");
        assert!((bottom - top - HEIGHT_MAX).abs() < 1e-9, "3 pt thick at most");
        // Without a trace: a bar under the frame.
        let bar = shadow_outline(r, None, POINTS);
        assert_eq!(bar.len(), 4);
        assert!((bar[2].p.y - (300.0 + GAP)).abs() < 1e-9);
    }

    #[test]
    fn a_round_base_takes_few_smooth_points_that_follow_it() {
        // The lower half of an ellipse, 200 px wide and 40 deep: a jar seen from slightly above.
        let contour: Vec<(u32, u32)> = (0..=200u32)
            .map(|x| {
                let t = (x as f64 - 100.0) / 100.0;
                (x, (40.0 - 40.0 * (1.0 - t * t).max(0.0).sqrt()).round() as u32)
            })
            .collect();
        let b = Base { width: 201, height: 100, bottom: 99, contour };
        let r = Rect::new(0.0, 0.0, 201.0, 100.0);
        for max in [POINTS, 6] {
            let pts = shadow_outline(r, Some(&b), max);
            assert!(pts.len() <= max && pts.len() >= 5, "{max}: {}", pts.len());
            let smooth = pts.iter().filter(|a| a.kind == designcraft_geom::AnchorKind::Smooth).count();
            assert_eq!(smooth, pts.len() - 4, "curves between the four corners");
            // Drawn as curves, the bottom edge stays within a couple of points of the real base.
            let mut path = PathData::single(SubPath::new(pts, true)).to_bezpath();
            path.apply_affine(Affine::IDENTITY);
            // (The middle 70 %, where the shadow shows: at the steep ends a sideways hair is a big
            // vertical difference.)
            for (x, off) in b.contour.iter().skip(30).step_by(10).take(15) {
                let (px, want) = (*x as f64 + 0.5, 100.0 - *off as f64 + GAP);
                let got = (0..2000)
                    .map(|k| k as f64 / 20.0)
                    .filter(|y| designcraft_geom::hit::fill_contains(&path, designcraft_geom::FillRule::NonZero, Point::new(px, *y)))
                    .fold(f64::NEG_INFINITY, f64::max);
                assert!((got - want).abs() < if max == POINTS { 2.0 } else { 4.0 }, "{max} points, x {px}: {got} vs {want}");
            }
        }
    }

    #[test]
    fn a_rotated_product_gets_its_shadow_on_its_own_base() {
        // The flyers this is for tilt their products about 10°: the shadow turns with the product.
        let svg =
            r##"<svg xmlns="http://www.w3.org/2000/svg" width="80" height="40"><rect x="10" y="20" width="60" height="20" fill="#c33"/></svg>"##;
        let mut s = Session::new();
        s.execute("file.new", &json!({})).unwrap();
        let placed = s
            .execute("file.place", &json!({"base64": super::super::file::base64_encode(svg.as_bytes()), "name": "p.svg", "x": 200, "y": 200}))
            .unwrap();
        let id = ItemId(placed["id"].as_u64().unwrap());
        s.execute("transform.rotate", &json!({"angle": 10, "ids": [id.0]})).unwrap();
        let r = s.execute("object.addProductShadow", &json!({"ids": [id.0]})).unwrap();
        assert_eq!(r["traced"], 1);
        let d = &s.doc().unwrap().doc;
        let (frame, shadow) = (d.item(id).unwrap(), d.item(ItemId(r["ids"][0].as_u64().unwrap())).unwrap());
        // In the product's own space the shadow is a level bar on its bottom edge.
        let back = frame.xf.inverse();
        let local: Vec<Point> = shadow.path.subpaths[0].anchors.iter().map(|a| back * (shadow.xf * a.p)).collect();
        let ib = frame.inner_bounds();
        let bottom = local.iter().map(|p| p.y).fold(f64::NEG_INFINITY, f64::max);
        assert!((bottom - (ib.y1 + GAP)).abs() < 0.5, "{bottom} vs {}", ib.y1 + GAP);
        let lowest: Vec<&Point> = local.iter().filter(|p| (p.y - bottom).abs() < 0.01).collect();
        assert_eq!(lowest.len(), 2, "both bottom corners level in the product's space: {local:?}");
    }

    #[test]
    fn a_photo_stored_sideways_gets_its_shadow_at_its_page_bottom() {
        // The opaque part is a bar along the image's right edge; the frame is turned a quarter so
        // the bar lies along the bottom or the top on the page. The shadow goes under the bar
        // as it stands on the page, not along the image's own bottom (a side on the page).
        let svg = r##"<svg xmlns="http://www.w3.org/2000/svg" width="80" height="80"><rect x="60" width="20" height="80" fill="#c33"/></svg>"##;
        let mut s = Session::new();
        s.execute("file.new", &json!({})).unwrap();
        let placed = s
            .execute("file.place", &json!({"base64": super::super::file::base64_encode(svg.as_bytes()), "name": "p.svg", "x": 200, "y": 200}))
            .unwrap();
        let id = ItemId(placed["id"].as_u64().unwrap());
        s.execute("transform.rotate", &json!({"angle": 90, "ids": [id.0]})).unwrap();
        let r = s.execute("object.addProductShadow", &json!({"ids": [id.0]})).unwrap();
        assert_eq!(r["traced"], 1);
        let d = &s.doc().unwrap().doc;
        let it = d.item(id).unwrap();
        let g = it.graphic().unwrap();
        // The bar on the page (60..80 × 0..80 in the image, 0.75 pt per pixel).
        let bar = (it.xf * g.xf).transform_rect_bbox(Rect::new(45.0, 0.0, 60.0, 60.0));
        let shadow = d.item(ItemId(r["ids"][0].as_u64().unwrap())).unwrap().bounds();
        assert!(bar.width() > bar.height(), "the bar lies flat on the page: {bar:?}");
        assert!((shadow.y1 - (bar.y1 + GAP)).abs() < 0.5, "under the bar: {shadow:?} {bar:?}");
        assert!((shadow.width() - (bar.width() - 2.0 * INSET)).abs() < 1.0, "as wide as the bar: {shadow:?} {bar:?}");
    }

    #[test]
    fn a_base_hidden_by_a_product_in_front_gets_no_shadow() {
        let svg =
            r##"<svg xmlns="http://www.w3.org/2000/svg" width="80" height="40"><rect x="10" y="20" width="60" height="20" fill="#c33"/></svg>"##;
        let place = |s: &mut Session, x: f64, y: f64| {
            ItemId(
                s.execute("file.place", &json!({"base64": super::super::file::base64_encode(svg.as_bytes()), "name": "p.svg", "x": x, "y": y}))
                    .unwrap()["id"]
                    .as_u64()
                    .unwrap(),
            )
        };
        let mut s = Session::new();
        s.execute("file.new", &json!({})).unwrap();
        let back = place(&mut s, 200.0, 200.0);
        // In front of it, lower down, covering its whole base.
        let front = s.execute("frame.create", &json!({"rect": [190, 210, 270, 260], "shape": "rectangle"})).unwrap()["id"].as_u64().unwrap();
        s.execute("object.fill", &json!({"ids": [front], "swatch": "[Black]"})).unwrap();
        // Picked on its own, it gets its shadow (behind what covers it).
        let r = s.execute("object.addProductShadow", &json!({"ids": [back.0]})).unwrap();
        assert_eq!(r["ids"].as_array().map(Vec::len), Some(1));
        s.execute("edit.undo", &json!({})).unwrap();
        // With other products selected too, its hidden base is skipped (the front one is no image:
        // nothing to shadow there).
        let r = s.execute("object.addProductShadow", &json!({"ids": [back.0, front]})).unwrap();
        assert_eq!(r["traced"], 1);
        assert_eq!(r["ids"].as_array().map(Vec::len), Some(0), "out of sight: no shadow, not a fallback bar");
    }

    #[test]
    fn adds_a_traced_shadow_behind_a_placed_image() {
        // A product whose upper half is transparent: only the lower bar is opaque.
        let svg =
            r##"<svg xmlns="http://www.w3.org/2000/svg" width="80" height="40"><rect x="10" y="20" width="60" height="20" fill="#c33"/></svg>"##;
        let mut s = Session::new();
        s.execute("file.new", &json!({})).unwrap();
        let placed = s
            .execute("file.place", &json!({"base64": super::super::file::base64_encode(svg.as_bytes()), "name": "p.svg", "x": 100, "y": 100}))
            .unwrap();
        let id = ItemId(placed["id"].as_u64().unwrap());
        let r = s.execute("object.addProductShadow", &json!({"ids": [id.0]})).unwrap();
        assert_eq!(r["traced"], 1);
        let anchors = s.doc().unwrap().doc.item(ItemId(r["ids"][0].as_u64().unwrap())).unwrap().path.subpaths[0].anchors.len();
        assert_eq!(anchors, 4, "a flat-bottomed product: a four-point bar");
        let sid = ItemId(r["ids"][0].as_u64().unwrap());
        let d = &s.doc().unwrap().doc;
        let (frame, shadow) = (d.item(id).unwrap().bounds(), d.item(sid).unwrap().bounds());
        let sp = d.spread(d.find(sid).unwrap().spread).unwrap();
        let at = |x: ItemId| sp.items.iter().position(|i| i.id == x).unwrap();
        assert!(at(sid) < at(id), "behind the product");
        let it = d.item(sid).unwrap();
        assert_eq!(it.name, "p_shadow", "named after the product's file");
        assert_eq!((it.object_style.as_str(), it.blend), (STYLE, BlendMode::Multiply));
        assert!(it.effects.drop_shadow.on);
        let style = d.styles.object_style(STYLE).unwrap();
        assert!(style.effects.as_ref().is_some_and(|e| e.drop_shadow.on && e.drop_shadow.opacity == 0.55), "the style holds the drop shadow");
        assert_eq!(it.effects, style.effects.clone().unwrap(), "the shadow follows its style");
        assert!((shadow.y1 - (frame.y1 + GAP)).abs() < 0.5, "on the product's base: {shadow:?} {frame:?}");
        // Traced, not the fallback bar: as wide as the opaque bar (60 of 80), not 95 % of the frame.
        let w = frame.width() * 60.0 / 80.0 - 2.0 * INSET;
        assert!((shadow.width() - w).abs() < 1.0, "{} vs {w}", shadow.width());
        // One undo step removes it.
        s.execute("edit.undo", &json!({})).unwrap();
        assert!(s.doc().unwrap().doc.item(sid).is_none());
    }
}
