export type Region = {
  x0: number;
  y0: number;
  x1: number;
  y1: number;
  area: number;
  /**
   * Boundary pixels, as flat x,y pairs. Only filled when asked for: the
   * convex hull is decided entirely by the outline, so interior pixels are
   * dead weight, and for text they outnumber the outline about ten to one.
   */
  edge?: number[];
};

/**
 * 8-connected component labelling over a binary mask, union-find with path
 * compression. Replaces cv.findContours + boundingRect.
 *
 * With `collectEdges` each region also carries its outline, which boxes.ts
 * turns into a minimum-area rectangle for rotated text.
 */
export function connectedRegions(
  mask: Uint8Array,
  width: number,
  height: number,
  collectEdges = false,
): Region[] {
  const labels = new Int32Array(width * height);
  const parent: number[] = [0];

  const find = (a: number): number => {
    let r = a;
    while (parent[r] !== r) r = parent[r];
    while (parent[a] !== r) {
      const next = parent[a];
      parent[a] = r;
      a = next;
    }
    return r;
  };
  const union = (a: number, b: number) => {
    const ra = find(a);
    const rb = find(b);
    if (ra !== rb) parent[Math.max(ra, rb)] = Math.min(ra, rb);
  };

  for (let y = 0; y < height; y++) {
    for (let x = 0; x < width; x++) {
      const i = y * width + x;
      if (!mask[i]) continue;
      // Only the already-scanned half of the 8-neighbourhood matters.
      let best = 0;
      const consider = (j: number) => {
        const l = labels[j];
        if (!l) return;
        if (!best) best = l;
        else union(best, l);
      };
      if (x > 0) consider(i - 1);
      if (y > 0) {
        consider(i - width);
        if (x > 0) consider(i - width - 1);
        if (x + 1 < width) consider(i - width + 1);
      }
      if (!best) {
        best = parent.length;
        parent.push(best);
      }
      labels[i] = best;
    }
  }

  const boxes = new Map<number, Region>();
  for (let y = 0; y < height; y++) {
    for (let x = 0; x < width; x++) {
      const i = y * width + x;
      const l = labels[i];
      if (!l) continue;
      const root = find(l);
      let r = boxes.get(root);
      if (!r) {
        r = { x0: x, y0: y, x1: x, y1: y, area: 1, edge: collectEdges ? [] : undefined };
        boxes.set(root, r);
      } else {
        if (x < r.x0) r.x0 = x;
        if (x > r.x1) r.x1 = x;
        if (y < r.y0) r.y0 = y;
        if (y > r.y1) r.y1 = y;
        r.area++;
      }
      if (!collectEdges) continue;
      const onEdge = x === 0 || y === 0 || x === width - 1 || y === height - 1 ||
        !mask[i - 1] || !mask[i + 1] || !mask[i - width] || !mask[i + width];
      if (onEdge) r.edge!.push(x, y);
    }
  }
  return [...boxes.values()];
}
