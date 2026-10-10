import type { LiveView } from '../lib/live';
import { Icon } from './Icons';

export function AnalysisPreview({ view, bounds }: { view: LiveView; bounds: { width: number; height: number } }) {
  const { assessment, url } = view;
  const boxes = assessment.boxes ?? assessment.hazards.filter(h => h.box);
  if (!boxes.length) return null;
  const width = assessment.width, height = assessment.height;
  const scoreSize = width * .09;
  const previewWidth = Math.min(260, bounds.width * .58, bounds.height * .65 * width / height);
  return <figure className="analysis-preview" style={{ width: previewWidth, maxWidth: 'none', maxHeight: 'none' }} aria-label={`Model A analyzed frame, captured ${Math.max(0, Math.round((Date.now() - assessment.capturedAt) / 1000))} seconds ago`}>
    {/* Keep predictions on the exact submitted image, never on the newer live view. */}
    <img src={url} width={width} height={height} alt="Analyzed camera frame" />
    <svg className="analysis-overlay" viewBox={`0 0 ${width} ${height}`} aria-hidden="true">
      {boxes.map(h => {
        const [bx, by, bw, bh] = h.box!;
        const x = bx * width, y = by * height;
        const sx = Math.max(2, Math.min(x, width - scoreSize * 1.7));
        const sy = Math.max(2, y - scoreSize);
        return <g key={h.id}>
          <rect className="hazard-box" x={x} y={y} width={bw * width} height={bh * height} />
          {h.severity !== null && <g transform={`translate(${sx},${sy})`}>
            <rect fill="#ffe14d" width={scoreSize * 1.7} height={scoreSize} rx={scoreSize * .2} />
            <text x={scoreSize * .85} y={scoreSize * .72} textAnchor="middle" fill="#161612" fontSize={scoreSize * .68} fontWeight="650">{h.severity.toFixed(2)}</text>
          </g>}
        </g>;
      })}
    </svg>
    <span className="snapshot-icon" title="Analyzed snapshot"><Icon name="clock" /></span>
  </figure>;
}
