import { assessFlow, MotionGate, unknownMotion, type FlowPoint } from './motion';

// OpenCV's WASM API is loaded only in this worker. Camera pixels never leave it.
type Mat = { data32F: Float32Array; data: Uint8Array; rows: number; cols: number; delete(): void };
type CV = { [name: string]: any };
const scope = self as unknown as { cv: CV; importScripts(url: string): void; onmessage: (event: MessageEvent) => void; postMessage(value: unknown): void };
let cv: CV;
let previous: Mat | undefined;
let oldPoints: Mat | undefined;
let previousTime = 0;
const gate = new MotionGate();

function reset() { previous?.delete(); oldPoints?.delete(); previous = undefined; oldPoints = undefined; previousTime = 0; gate.reset(); }
scope.onmessage = event => {
  if (event.data.type === 'reset') { reset(); return; }
  const { pixels, width, height, timestamp } = event.data;
  if (!cv) { scope.postMessage(unknownMotion(timestamp, 'loading_motion')); return; }
  const owned: Mat[] = [];
  try {
    const rgba = cv.matFromArray(height, width, cv.CV_8UC4, new Uint8Array(pixels)) as Mat; owned.push(rgba);
    const gray = new cv.Mat() as Mat; owned.push(gray);
    cv.cvtColor(rgba, gray, cv.COLOR_RGBA2GRAY);
    let result = unknownMotion(timestamp);
    const dt = (timestamp - previousTime) / 1000;
    if (previous && oldPoints && oldPoints.rows >= 10 && dt > .025 && dt < .5 && previous.rows === height && previous.cols === width) {
      const next = new cv.Mat() as Mat, back = new cv.Mat() as Mat;
      const status = new cv.Mat() as Mat, error = new cv.Mat() as Mat, reverseStatus = new cv.Mat() as Mat, reverseError = new cv.Mat() as Mat;
      owned.push(next, back, status, error, reverseStatus, reverseError);
      const size = new cv.Size(15, 15), criteria = new cv.TermCriteria(cv.TermCriteria_COUNT + cv.TermCriteria_EPS, 20, .03);
      cv.calcOpticalFlowPyrLK(previous, gray, oldPoints, next, status, error, size, 2, criteria);
      cv.calcOpticalFlowPyrLK(gray, previous, next, back, reverseStatus, reverseError, size, 2, criteria);
      const points: FlowPoint[] = [];
      for (let i = 0; i < oldPoints.rows; i++) {
        const x = oldPoints.data32F[i * 2], y = oldPoints.data32F[i * 2 + 1];
        const nx = next.data32F[i * 2], ny = next.data32F[i * 2 + 1];
        const fb = Math.hypot(back.data32F[i * 2] - x, back.data32F[i * 2 + 1] - y);
        if (status.data[i] && reverseStatus.data[i] && error.data32F[i] < 24 && fb < 1.25 && nx >= 0 && nx < width && ny >= 0 && ny < height) {
          points.push({ x: x / width, y: y / height, dx: (nx - x) / width / dt, dy: (ny - y) / height / dt });
        }
      }
      result = assessFlow(points, timestamp, points.length / oldPoints.rows);
    }
    previous?.delete(); oldPoints?.delete(); oldPoints = undefined;
    owned.splice(owned.indexOf(gray), 1);
    previous = gray; previousTime = timestamp;
    oldPoints = new cv.Mat() as Mat;
    const mask = new cv.Mat() as Mat; owned.push(mask);
    cv.goodFeaturesToTrack(gray, oldPoints, 300, .015, 4, mask, 5);
    scope.postMessage(gate.update(result));
  } catch {
    reset(); scope.postMessage(unknownMotion(timestamp, 'motion_unavailable'));
  } finally { for (const mat of owned) mat.delete(); }
};

(async () => {
  try {
    scope.importScripts('/vendor/opencv-4.13.0.js');
    // This OpenCV build has a legacy .then() that resolves to itself. Awaiting
    // the module directly loops forever through Promise thenable assimilation.
    const loaded = scope.cv;
    await new Promise<void>(resolve => { loaded.then(() => resolve()); });
    cv = loaded;
    scope.postMessage({ type: 'ready' });
  } catch { scope.postMessage({ type: 'failed' }); }
})();
