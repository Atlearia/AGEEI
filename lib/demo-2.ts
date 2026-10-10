import type { Track } from './detection';

/** Hand-authored against human data/demo video 2.MOV. All coordinates are
 * normalized to its upright frame; scores describe threat, never confidence.
 * Deliberate clear intervals keep the demo from announcing every visible object.
 */
export const DEMO_2_TRACKS: readonly Track[] = [
  {
    id: 'planter-edge', label: 'Raised planter edge', direction: 'ahead', start: 3, end: 12.5,
    keyframes: [
      { time: 3, box: [.61, .57, .39, .11], severity: .34 },
      { time: 4, box: [.075, .49, .925, .23], severity: .43 },
      { time: 5, box: [.745, .56, .255, .16], severity: .55 },
      { time: 6, box: [.00, .55, 1, .24], severity: .65 },
      { time: 7, box: [.00, .58, 1, .22], severity: .72 },
      { time: 8, box: [.01, .57, .99, .33], severity: .78 },
      { time: 9, box: [.00, .57, 1, .43], severity: .84 },
      { time: 10, box: [.00, .60, 1, .40], severity: .86 },
      { time: 11, box: [.00, .56, 1, .44], severity: .75 },
      { time: 12, box: [.00, .52, .56, .42], severity: .46 },
      { time: 12.5, box: [.00, .53, .42, .32], severity: .30 }
    ]
  },
  {
    id: 'parked-bicycles', label: 'Parked bicycles', direction: 'left', start: 33.5, end: 38.1,
    keyframes: [
      { time: 33.5, box: [.00, .47, .10, .30], severity: .30 },
      { time: 33.7, box: [.00, .47, .25, .30], severity: .65 },
      { time: 34, box: [.00, .48, .36, .31], severity: .69 },
      { time: 34.4, box: [.00, .49, .47, .33], severity: .72 },
      { time: 35, box: [.00, .52, .60, .35], severity: .77 },
      { time: 36, box: [.00, .53, .64, .47], severity: .85 },
      { time: 37, box: [.00, .61, .51, .39], severity: .81 },
      { time: 37.6, box: [.00, .75, .37, .25], severity: .56 },
      { time: 38.1, box: [.00, .88, .12, .12], severity: .30 }
    ]
  },
  {
    id: 'lamp-post', label: 'Lamp post', direction: 'ahead', start: 49, end: 61,
    keyframes: [
      { time: 49, box: [.60, .29, .09, .23], severity: .35 },
      { time: 50, box: [.50, .34, .08, .19], severity: .43 },
      { time: 51, box: [.44, .27, .10, .28], severity: .54 },
      { time: 52, box: [.50, .32, .095, .25], severity: .65 },
      { time: 53, box: [.55, .28, .11, .32], severity: .69 },
      { time: 54, box: [.50, .31, .13, .30], severity: .73 },
      { time: 55, box: [.47, .19, .16, .455], severity: .77 },
      { time: 56, box: [.32, .14, .16, .55], severity: .82 },
      { time: 57, box: [.40, .045, .19, .69], severity: .86 },
      { time: 58, box: [.44, .00, .19, .84], severity: .90 },
      { time: 59, box: [.40, .00, .27, .96], severity: .93 },
      { time: 60, box: [.10, .00, .34, 1], severity: .67 },
      { time: 60.5, box: [.00, .00, .16, 1], severity: .43 },
      { time: 61, box: [.00, .00, .03, 1], severity: .30 }
    ]
  },
  {
    id: 'car-ahead', label: 'Car', direction: 'ahead', start: 58, end: 70.7,
    keyframes: [
      { time: 58, box: [.48, .485, .465, .085], severity: .40 },
      { time: 59, box: [.55, .48, .445, .09], severity: .49 },
      { time: 60, box: [.385, .475, .58, .10], severity: .60 },
      { time: 60.5, box: [.345, .47, .60, .105], severity: .65 },
      { time: 61, box: [.315, .46, .635, .115], severity: .70 },
      { time: 62, box: [.335, .46, .66, .135], severity: .75 },
      { time: 63, box: [.31, .44, .675, .16], severity: .80 },
      { time: 64, box: [.14, .45, .80, .18], severity: .86 },
      { time: 65, box: [.00, .43, .84, .195], severity: .90 },
      { time: 66, box: [.00, .405, .98, .27], severity: .94 },
      { time: 67, box: [.00, .415, 1, .33], severity: .96 },
      { time: 68, box: [.00, .435, 1, .40], severity: .93 },
      { time: 69, box: [.00, .46, 1, .51], severity: .82 },
      { time: 70, box: [.30, .50, .70, .50], severity: .54 },
      { time: 70.7, box: [.90, .65, .10, .35], severity: .30 }
    ]
  },
  {
    id: 'guardrail', label: 'Guardrail', direction: 'right', start: 88.5, end: 93.4,
    keyframes: [
      { time: 88.5, box: [.90, .545, .10, .16], severity: .33 },
      { time: 89, box: [.80, .565, .20, .145], severity: .65 },
      { time: 90, box: [.43, .59, .57, .16], severity: .73 },
      { time: 91, box: [.14, .605, .86, .23], severity: .78 },
      { time: 92, box: [.09, .695, .91, .28], severity: .85 },
      { time: 92.6, box: [.27, .825, .73, .175], severity: .70 },
      { time: 93, box: [.58, .95, .42, .05], severity: .43 },
      { time: 93.4, box: [.86, .985, .14, .015], severity: .30 }
    ]
  }
];
