import type { SVGProps } from 'react';
type Name = 'arrow' | 'camera' | 'video' | 'boxes' | 'sound' | 'muted' | 'stop' | 'close' | 'back' | 'play' | 'alert' | 'retry' | 'clock' | 'motion';
export function Icon({ name, ...props }: SVGProps<SVGSVGElement> & { name: Name }) {
  const paths: Record<Name, React.ReactNode> = {
    arrow: <path d="M6 18 18 6M6 6h12v12" />,
    camera: <><path d="m8 6 1-3h6l1 3" /><rect x="3" y="6" width="18" height="14" rx="3" /><circle cx="12" cy="13" r="4" /></>,
    video: <><rect x="3" y="3" width="18" height="18" rx="5" /><path d="m10 8 6 4-6 4Z" /></>,
    boxes: <><path d="M8 3H3v5m13-5h5v5M3 16v5h5m13-5v5h-5" /><rect x="8" y="8" width="8" height="8" rx="2" /></>,
    sound: <><path d="m11 4-5 4H3v8h3l5 4Z" /><path d="M15 8a6 6 0 0 1 0 8m3-11a10 10 0 0 1 0 14" /></>,
    muted: <><path d="m11 4-5 4H3v8h3l5 4Z" /><path d="m16 9 6 6m0-6-6 6" /></>,
    stop: <rect x="6" y="6" width="12" height="12" rx="3" fill="currentColor" stroke="none" />,
    close: <path d="m6 6 12 12M6 18 18 6" />,
    back: <path d="m14 5-7 7 7 7" />,
    play: <path d="m8 4 12 8-12 8Z" fill="currentColor" stroke="none" />,
    alert: <><path d="m12 3 10 18H2Z" /><path d="M12 9v5m0 3v.1" /></>,
    retry: <path d="M4 10a8 8 0 1 1 1 7M4 4v6h6" />,
    clock: <><circle cx="12" cy="12" r="9" /><path d="M12 7v5l3 2" /></>,
    motion: <><circle cx="12" cy="12" r="9" /><path d="m3 12 5-1 2 5 4-10 2 6h5" /></>
  };
  return <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.7" strokeLinecap="round" strokeLinejoin="round" aria-hidden="true" {...props}>{paths[name]}</svg>;
}
