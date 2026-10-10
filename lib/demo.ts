import { detectionsAt, type Track } from './detection';
import { DEMO_2_TRACKS } from './demo-2';

/** Authored ONLY for human data/demo video.MOV, never applied to a camera stream.
 * Time is the decoded video's media time. Coordinates refer to the upright frame.
 * These severity values are deliberately scripted for the demonstration.
 */
export const DEMO = { id: 'building-walk-v1', url: '/media/walk.mp4', duration: 18.278, width: 720, height: 1280 };
export const DEMO_TRACKS: readonly Track[] = [
  {
    id: 'raised-edge', label: 'Raised curb', direction: 'right', start: 1.5, end: 7.2,
    keyframes: [
      { time: 1.5, box: [0.465, 0.445, 0.34, 0.145], severity: 0.38 },
      { time: 2.0, box: [0.490, 0.466, 0.39, 0.155], severity: 0.52 },
      { time: 2.5, box: [0.505, 0.459, 0.47, 0.185], severity: 0.65 },
      { time: 3.0, box: [0.485, 0.465, 0.515, 0.195], severity: 0.71 },
      { time: 3.5, box: [0.455, 0.463, 0.545, 0.243], severity: 0.76 },
      { time: 4.0, box: [0.425, 0.462, 0.575, 0.280], severity: 0.80 },
      { time: 4.5, box: [0.390, 0.460, 0.610, 0.380], severity: 0.82 },
      { time: 5.0, box: [0.365, 0.500, 0.635, 0.448], severity: 0.83 },
      { time: 5.5, box: [0.360, 0.600, 0.640, 0.400], severity: 0.80 },
      { time: 6.0, box: [0.450, 0.680, 0.550, 0.320], severity: 0.70 },
      { time: 6.5, box: [0.505, 0.765, 0.495, 0.235], severity: 0.48 },
      { time: 7.2, box: [0.560, 0.950, 0.440, 0.050], severity: 0.30 }
    ]
  },
  {
    id: 'barrier', label: 'Barrier', direction: 'left', start: 8.0, end: 15.85,
    keyframes: [
      { time: 8.0, box: [0.000, 0.408, 0.315, 0.085], severity: 0.32 },
      { time: 8.5, box: [0.000, 0.402, 0.295, 0.104], severity: 0.38 },
      { time: 9.0, box: [0.000, 0.423, 0.390, 0.115], severity: 0.46 },
      { time: 9.5, box: [0.000, 0.429, 0.445, 0.135], severity: 0.57 },
      { time: 10.0, box: [0.000, 0.433, 0.495, 0.145], severity: 0.66 },
      { time: 10.5, box: [0.000, 0.437, 0.545, 0.143], severity: 0.71 },
      { time: 11.0, box: [0.000, 0.445, 0.540, 0.176], severity: 0.74 },
      { time: 11.5, box: [0.000, 0.463, 0.470, 0.183], severity: 0.77 },
      { time: 12.0, box: [0.000, 0.468, 0.480, 0.188], severity: 0.81 },
      { time: 12.5, box: [0.000, 0.476, 0.530, 0.213], severity: 0.85 },
      { time: 13.0, box: [0.000, 0.488, 0.585, 0.250], severity: 0.87 },
      { time: 13.5, box: [0.000, 0.512, 0.570, 0.271], severity: 0.90 },
      { time: 14.0, box: [0.000, 0.531, 0.545, 0.375], severity: 0.91 },
      { time: 14.5, box: [0.000, 0.571, 0.480, 0.415], severity: 0.88 },
      { time: 15.0, box: [0.000, 0.620, 0.540, 0.380], severity: 0.71 },
      { time: 15.5, box: [0.000, 0.735, 0.490, 0.265], severity: 0.51 },
      { time: 15.85, box: [0.000, 0.940, 0.400, 0.060], severity: 0.30 }
    ]
  }
];
export const DEMOS = [
  { ...DEMO, title: 'Demo video 1', tracks: DEMO_TRACKS },
  { id: 'courtyard-walk-v2', title: 'Demo video 2', url: '/media/walk-2.mp4', duration: 106.974, width: 720, height: 1280, tracks: DEMO_2_TRACKS }
] as const;

export const getDemo = (id: string) => DEMOS.find(demo => demo.id === id);
export const demoDetectionsAt = (time: number, demoId = DEMO.id) => detectionsAt(getDemo(demoId)?.tracks ?? [], time);
