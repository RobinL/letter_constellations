import {
  buildHandwritingPath,
  compileTracingPath,
  type PreparedStroke,
  type WritingPath,
} from 'letterpaths';

export type SupportedLetterStyle = 'print' | 'pre-cursive';

export type LetterPlotPoint = {
  order: number;
  x: number;
  y: number;
  strokeStart?: boolean;
};

export type LetterPlot = {
  name: string;
  points: LetterPlotPoint[];
};

export const availableLetters = 'abcdefghijklmnopqrstuvwxyz'.split('');
export const availableLetterStyles: SupportedLetterStyle[] = ['print', 'pre-cursive'];
export const defaultLetterStyle: SupportedLetterStyle = 'print';

const tracingSampleRateByStyle: Record<SupportedLetterStyle, number> = {
  print: 64,
  'pre-cursive': 80,
};

const plotCache = new Map<string, LetterPlot>();

const createCacheKey = (letter: string, style: SupportedLetterStyle): string =>
  `${style}:${letter}`;

type SampleCandidate = {
  x: number;
  y: number;
  distanceAlongStroke: number;
  strokeStart?: boolean;
};

const ROOT_EPSILON = 1e-4;
const MOTION_EPSILON = 1;

const clamp01 = (value: number): number => Math.max(0, Math.min(1, value));

const solveQuadratic = (a: number, b: number, c: number): number[] => {
  if (Math.abs(a) < ROOT_EPSILON) {
    if (Math.abs(b) < ROOT_EPSILON) {
      return [];
    }
    return [-c / b];
  }

  const discriminant = b * b - 4 * a * c;
  if (discriminant < -ROOT_EPSILON) {
    return [];
  }
  if (Math.abs(discriminant) < ROOT_EPSILON) {
    return [-b / (2 * a)];
  }

  const sqrtDiscriminant = Math.sqrt(Math.max(0, discriminant));
  return [
    (-b - sqrtDiscriminant) / (2 * a),
    (-b + sqrtDiscriminant) / (2 * a),
  ];
};

const getAxisExtremaTs = (
  p0: number,
  p1: number,
  p2: number,
  p3: number
): number[] => {
  const cubicA = -p0 + 3 * p1 - 3 * p2 + p3;
  const cubicB = 3 * p0 - 6 * p1 + 3 * p2;
  const cubicC = -3 * p0 + 3 * p1;

  return solveQuadratic(3 * cubicA, 2 * cubicB, cubicC)
    .map((t) => clamp01(t))
    .filter((t) => t > ROOT_EPSILON && t < 1 - ROOT_EPSILON);
};

const getCurveExtremaTs = (curve: WritingPath['strokes'][number]['curves'][number]): number[] => {
  const roots = [
    ...getAxisExtremaTs(curve.p0.x, curve.p1.x, curve.p2.x, curve.p3.x),
    ...getAxisExtremaTs(curve.p0.y, curve.p1.y, curve.p2.y, curve.p3.y),
  ];

  const uniqueRoots: number[] = [];
  for (const root of roots) {
    if (!uniqueRoots.some((existing) => Math.abs(existing - root) < ROOT_EPSILON)) {
      uniqueRoots.push(root);
    }
  }

  return uniqueRoots.sort((a, b) => a - b);
};

const approximateCurveLengthToT = (
  curve: WritingPath['strokes'][number]['curves'][number],
  t: number,
  steps = 24
): number => {
  if (t <= 0) {
    return 0;
  }

  let total = 0;
  let previous = curve.getPointAt(0);
  for (let i = 1; i <= steps; i += 1) {
    const point = curve.getPointAt((t * i) / steps);
    total += Math.hypot(point.x - previous.x, point.y - previous.y);
    previous = point;
  }
  return total;
};

const collectCurveExtrema = (
  stroke: WritingPath['strokes'][number]
): SampleCandidate[] => {
  const extrema: SampleCandidate[] = [];
  let strokeOffset = 0;

  for (const curve of stroke.curves) {
    const curveLength = curve.length();
    for (const t of getCurveExtremaTs(curve)) {
      const point = curve.getPointAt(t);
      extrema.push({
        x: point.x,
        y: point.y,
        distanceAlongStroke: strokeOffset + approximateCurveLengthToT(curve, t),
      });
    }
    strokeOffset += curveLength;
  }

  return extrema;
};

const hasOpposingMotion = (before: number, after: number): boolean =>
  Math.abs(before) > MOTION_EPSILON &&
  Math.abs(after) > MOTION_EPSILON &&
  Math.sign(before) !== Math.sign(after);

const isTurningBoundary = (
  stroke: PreparedStroke,
  distanceAlongStroke: number,
  x: number,
  y: number
): boolean => {
  const { samples } = stroke;
  if (samples.length < 3) {
    return false;
  }

  let nextIndex = samples.findIndex(
    (sample) => sample.distanceAlongStroke >= distanceAlongStroke
  );
  if (nextIndex === -1) {
    nextIndex = samples.length - 1;
  }

  const prevIndex = Math.max(0, nextIndex - 1);
  const afterIndex = Math.min(samples.length - 1, nextIndex + 1);
  if (prevIndex === afterIndex) {
    return false;
  }

  const previous = samples[prevIndex];
  const next = samples[afterIndex];
  const dxBefore = x - previous.x;
  const dxAfter = next.x - x;
  const dyBefore = y - previous.y;
  const dyAfter = next.y - y;

  return hasOpposingMotion(dxBefore, dxAfter) || hasOpposingMotion(dyBefore, dyAfter);
};

const collectBoundaryTurns = (
  path: ReturnType<typeof compileTracingPath>
): SampleCandidate[][] => {
  const boundariesByStroke = path.strokes.map(() => [] as SampleCandidate[]);
  let strokeStartDistance = 0;

  for (let strokeIndex = 0; strokeIndex < path.strokes.length; strokeIndex += 1) {
    const stroke = path.strokes[strokeIndex];
    const strokeEndDistance = strokeStartDistance + stroke.totalLength;

    for (const boundary of path.boundaries) {
      if (
        boundary.overallDistance < strokeStartDistance - ROOT_EPSILON ||
        boundary.overallDistance > strokeEndDistance + ROOT_EPSILON
      ) {
        continue;
      }

      const localDistance = boundary.overallDistance - strokeStartDistance;
      if (
        isTurningBoundary(stroke, localDistance, boundary.point.x, boundary.point.y)
      ) {
        boundariesByStroke[strokeIndex].push({
          x: boundary.point.x,
          y: boundary.point.y,
          distanceAlongStroke: localDistance,
        });
      }
    }

    strokeStartDistance = strokeEndDistance;
  }

  return boundariesByStroke;
};

const mergeStrokeCandidates = (
  regularSamples: PreparedStroke['samples'],
  insertedSamples: SampleCandidate[],
  strokeStart: boolean
): LetterPlotPoint[] => {
  const candidates: SampleCandidate[] = [
    ...regularSamples.map((sample, sampleIndex) => ({
      x: sample.x,
      y: sample.y,
      distanceAlongStroke: sample.distanceAlongStroke,
      strokeStart: strokeStart && sampleIndex === 0,
    })),
    ...insertedSamples,
  ];

  candidates.sort((a, b) => a.distanceAlongStroke - b.distanceAlongStroke);

  const merged: LetterPlotPoint[] = [];
  for (const candidate of candidates) {
    const previous = merged[merged.length - 1];
    if (
      previous &&
      Math.abs(previous.x - candidate.x) < 0.5 &&
      Math.abs(previous.y - candidate.y) < 0.5
    ) {
      if (candidate.strokeStart) {
        previous.strokeStart = true;
      }
      continue;
    }

    merged.push({
      order: 0,
      x: candidate.x,
      y: candidate.y,
      strokeStart: candidate.strokeStart,
    });
  }

  if (merged.length > 0 && strokeStart) {
    merged[0].strokeStart = true;
  }

  return merged;
};

export const buildLetterPlot = (
  letter: string,
  style: SupportedLetterStyle
): LetterPlot => {
  const normalizedLetter = letter.toLowerCase();
  const cacheKey = createCacheKey(normalizedLetter, style);
  const cached = plotCache.get(cacheKey);
  if (cached) {
    return {
      name: cached.name,
      points: cached.points.map((point) => ({ ...point })),
    };
  }

  const path = buildHandwritingPath(normalizedLetter, { style });
  const prepared = compileTracingPath(path, {
    sampleRate: tracingSampleRateByStyle[style],
  });
  const turningBoundaries = collectBoundaryTurns(prepared);

  let order = 1;
  const points = prepared.strokes.flatMap((stroke, strokeIndex) => {
    const insertedSamples = [
      ...collectCurveExtrema(path.strokes[strokeIndex]),
      ...turningBoundaries[strokeIndex],
    ];

    return mergeStrokeCandidates(
      stroke.samples,
      insertedSamples,
      strokeIndex > 0
    ).map((point) => ({
      ...point,
      order: order++,
    }));
  });

  const plot = { name: normalizedLetter, points };
  plotCache.set(cacheKey, plot);
  return {
    name: plot.name,
    points: plot.points.map((point) => ({ ...point })),
  };
};
