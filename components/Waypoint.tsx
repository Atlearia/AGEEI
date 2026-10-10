'use client';
import { useCallback, useEffect, useRef, useState } from 'react';
import { DEMO, DEMOS, getDemo, demoDetectionsAt } from '../lib/demo';
import { containRect, type Detection } from '../lib/detection';
import { AlertController } from '../lib/alert-controller';
import { AudioOutput } from '../lib/audio';
import { LiveCamera, type LiveView } from '../lib/live';
import { unknownMotion, type MotionResult } from '../lib/motion';
import { AnalysisPreview } from './AnalysisPreview';
import { Icon } from './Icons';

type Input = 'demo' | 'camera';
type Status = 'home' | 'source' | 'viewer';
type VideoWithCallbacks = HTMLVideoElement & {
  requestVideoFrameCallback?: (callback: (now: number, metadata: { mediaTime: number }) => void) => number;
  cancelVideoFrameCallback?: (id: number) => void;
};

export default function Waypoint() {
  const [screen, setScreen] = useState<Status>('home');
  const [input, setInput] = useState<Input>('demo');
  const [demoId, setDemoId] = useState(DEMO.id);
  const [overlay, setOverlay] = useState(true);
  const [sound, setSound] = useState(true);
  const [audioReady, setAudioReady] = useState(false);
  const [loading, setLoading] = useState(false);
  const [paused, setPaused] = useState(false);
  const [error, setError] = useState('');
  const [showError, setShowError] = useState(false);
  const [detections, setDetections] = useState<Detection[]>([]);
  const [motion, setMotion] = useState<MotionResult>(() => unknownMotion(0));
  const [liveView, setLiveView] = useState<LiveView | null>(null);
  const [liveStatus, setLiveStatus] = useState('warming_up');
  const [frameRect, setFrameRect] = useState({ x: 0, y: 0, width: 0, height: 0 });
  const video = useRef<HTMLVideoElement>(null);
  const playback = useRef<Promise<void> | null>(null);
  const stage = useRef<HTMLDivElement>(null);
  const audio = useRef<AudioOutput | null>(null);
  const alerts = useRef<AlertController | null>(null);
  const live = useRef<LiveCamera | null>(null);
  const stream = useRef<MediaStream | null>(null);
  const session = useRef(0);
  const cancelFrames = useRef<(() => void) | null>(null);
  const soundRef = useRef(sound);
  const wake = useRef<WakeLockSentinel | null>(null);
  const sourceDialog = useRef<HTMLDialogElement>(null);
  const errorDialog = useRef<HTMLDialogElement>(null);

  const reportError = useCallback((message: string) => { setError(message); }, []);
  const stopMedia = useCallback(() => {
    session.current++;
    cancelFrames.current?.(); cancelFrames.current = null;
    alerts.current?.dispose(); alerts.current = null;
    live.current?.dispose(); live.current = null;
    audio.current?.stop();
    stream.current?.getTracks().forEach(track => track.stop()); stream.current = null;
    if (video.current) { video.current.pause(); video.current.srcObject = null; video.current.removeAttribute('src'); video.current.load(); }
    playback.current = null;
    void wake.current?.release(); wake.current = null;
  }, []);
  const stop = useCallback(() => { stopMedia(); setDetections([]); setPaused(false); setLoading(false); setScreen('home'); setShowError(false); }, [stopMedia]);

  const align = useCallback(() => {
    const media = video.current, area = stage.current;
    if (!media || !area) return;
    setFrameRect(containRect(area.clientWidth, area.clientHeight, media.videoWidth, media.videoHeight));
  }, []);
  const unlock = useCallback((confirm = false) => {
    if (!audio.current) audio.current = new AudioOutput(setAudioReady);
    return audio.current.unlock(confirm).then(() => {
      setError(previous => previous === 'Tap the speaker to enable sound.' ? '' : previous);
    }).catch(() => { setAudioReady(false); reportError('Tap the speaker to enable sound.'); });
  }, [reportError]);

  function choose(input: Input, selectedDemoId = DEMO.id) {
    // AudioContext is unlocked within the selecting gesture, before media/API awaits.
    stopMedia();
    void unlock();
    // The same video element stays mounted, so audible playback starts directly
    // in the source-selection gesture on mobile Safari, before React effects.
    const media = video.current;
    const demo = getDemo(selectedDemoId);
    if (media && input === 'demo' && demo) {
      media.muted = false;
      media.volume = 1;
      media.src = demo.url;
      media.load();
      playback.current = media.play();
      // The viewer effect awaits this promise and presents a recoverable error.
      void playback.current.catch(() => {});
    }
    setError(''); setShowError(false); setLoading(true); setPaused(false); setDetections([]);
    setLiveView(null); setMotion(unknownMotion(0)); setLiveStatus('warming_up');
    setDemoId(selectedDemoId); setInput(input); setScreen('viewer');
  }

  useEffect(() => {
    soundRef.current = sound;
    alerts.current?.setMuted(!sound);
    live.current?.setMuted(!sound);
  }, [sound]);
  useEffect(() => {
    if (screen === 'source') sourceDialog.current?.showModal();
    else sourceDialog.current?.close();
  }, [screen]);
  useEffect(() => { if (showError) errorDialog.current?.showModal(); else errorDialog.current?.close(); }, [showError]);

  useEffect(() => {
    if (screen !== 'viewer') return;
    const media = video.current;
    if (!media) return;
    const currentSession = ++session.current;
    let disposed = false;
    const alive = () => !disposed && currentSession === session.current;
    const pause = () => { alerts.current?.setPaused(true); live.current?.setPaused(true); setPaused(true); };
    const playing = () => { if (alive()) { setLoading(false); setPaused(false); alerts.current?.setPaused(false); live.current?.setPaused(false); } };
    const waiting = () => { if (alive()) { alerts.current?.setPaused(true); live.current?.setPaused(true); setLoading(true); } };
    const seek = () => { alerts.current?.reset(); setDetections([]); };
    const failed = () => { if (alive()) { setLoading(false); setPaused(true); reportError('The video could not open. Check your connection and try again.'); } };
    media.addEventListener('loadedmetadata', align);
    media.addEventListener('playing', playing);
    media.addEventListener('pause', pause);
    media.addEventListener('waiting', waiting);
    media.addEventListener('seeking', seek);
    media.addEventListener('error', failed);
    media.addEventListener('ended', stop);
    const observer = new ResizeObserver(align);
    if (stage.current) observer.observe(stage.current);

    async function start() {
      try {
        if (!audio.current) audio.current = new AudioOutput(setAudioReady);
        if (input === 'demo') {
          alerts.current = new AlertController(audio.current, reportError, demoId);
          alerts.current.setMuted(!soundRef.current);
        }
        if (input === 'camera') {
          media!.muted = true;
          if (!navigator.mediaDevices?.getUserMedia) throw new Error('Camera access needs HTTPS and a supported browser.');
          const acquired = await navigator.mediaDevices.getUserMedia({ video: { facingMode: { ideal: 'environment' }, width: { ideal: 1280 }, height: { ideal: 720 } }, audio: false });
          if (!alive()) { acquired.getTracks().forEach(track => track.stop()); return; }
          stream.current = acquired;
          media!.srcObject = acquired;
          live.current = new LiveCamera(media!, audio.current, {
            motion: result => { if (alive()) setMotion(result); },
            result: view => { if (alive()) setLiveView(view); },
            status: status => { if (alive()) setLiveStatus(status); },
            error: message => { if (alive()) reportError(message); }
          });
          live.current.setMuted(!soundRef.current);
          live.current.start();
          acquired.getVideoTracks().forEach(track => track.addEventListener('ended', () => { if (alive()) { reportError('The camera disconnected. Choose an input again.'); setPaused(true); } }));
        } else {
          fetch('/api/status', { cache: 'no-store' }).then(r => r.json()).then(status => {
            if (alive() && !status.ready) reportError('Voice setup is not finished yet.');
          }).catch(() => { if (alive()) reportError('The voice service could not connect.'); });
        }
        let frameId = 0;
        const framed = media as VideoWithCallbacks;
        const update = (time: number) => {
          if (!alive() || media!.paused || media!.seeking || media!.readyState < 2) return;
          const current = input === 'demo' ? demoDetectionsAt(time, demoId) : [];
          setDetections(current);
          if (input === 'demo') alerts.current?.update(current, time);
          else live.current?.frame(time);
        };
        if (typeof framed.requestVideoFrameCallback === 'function') {
          const frame = (_now: number, metadata: { mediaTime: number }) => {
            if (!alive()) return;
            update(metadata.mediaTime);
            frameId = framed.requestVideoFrameCallback!(frame);
          };
          frameId = framed.requestVideoFrameCallback(frame);
          cancelFrames.current = () => framed.cancelVideoFrameCallback?.(frameId);
        } else {
          const frame = () => { if (!alive()) return; update(media!.currentTime); frameId = requestAnimationFrame(frame); };
          frameId = requestAnimationFrame(frame);
          cancelFrames.current = () => cancelAnimationFrame(frameId);
        }
        await (input === 'demo' && playback.current ? playback.current : media!.play());
        if (!alive()) return;
        if (input === 'camera' && soundRef.current) void unlock(true);
        align();
        setLoading(false);
        if ('wakeLock' in navigator) {
          try { const lock = await navigator.wakeLock.request('screen'); if (alive()) wake.current = lock; else void lock.release(); } catch { /* Browser/device may decline; playback still works. */ }
        }
      } catch (e) {
        if (!alive()) return;
        setLoading(false); setPaused(true);
        const name = e instanceof Error ? e.name : '';
        reportError(name === 'NotAllowedError' ? input === 'camera' ? 'Allow camera access, or choose a demo video.' : 'Tap play to start the video with sound.' : e instanceof Error ? e.message : 'The input could not start.');
        setShowError(input === 'camera' || name !== 'NotAllowedError');
      }
    }
    void start();
    const hidden = () => { if (document.hidden && alive()) { media.pause(); alerts.current?.setPaused(true); live.current?.setPaused(true); setPaused(true); } };
    const pageHide = () => { if (alive()) stopMedia(); };
    document.addEventListener('visibilitychange', hidden);
    window.addEventListener('pagehide', pageHide);
    return () => {
      disposed = true;
      observer.disconnect();
      media.removeEventListener('loadedmetadata', align); media.removeEventListener('playing', playing);
      media.removeEventListener('pause', pause); media.removeEventListener('waiting', waiting);
      media.removeEventListener('seeking', seek); media.removeEventListener('error', failed); media.removeEventListener('ended', stop);
      document.removeEventListener('visibilitychange', hidden); window.removeEventListener('pagehide', pageHide);
      stopMedia();
    };
  }, [screen, input, demoId, align, reportError, stop, stopMedia, unlock]);

  useEffect(() => () => { audio.current?.dispose(); }, []);

  function resume() {
    void unlock();
    setError('');
    if (!video.current?.srcObject && !video.current?.getAttribute('src')) { stop(); return; }
    alerts.current?.reset();
    void video.current?.play().catch(() => reportError('Tap again to resume the video.'));
  }

  return <main className={`app ${screen === 'viewer' ? 'viewing' : ''}`}>
    {screen !== 'viewer' && <section className="start-screen" aria-label="Waypoint">
      <div className="brand-mark"><Icon name="arrow" /></div>
      <button className="start-button" onClick={() => { void unlock(); setScreen('source'); }}>Start detecting<Icon name="arrow" /></button>
    </section>}
    <section className="viewer" hidden={screen !== 'viewer'} aria-label="Detection view">
      <div className={`stage ${input === 'camera' && motion.state === 'ALERT' ? 'motion-alert' : ''}`} ref={stage}>
        <video ref={video} muted={input === 'camera'} playsInline disablePictureInPicture disableRemotePlayback controls={false} controlsList="nodownload noremoteplayback noplaybackrate" preload="auto" aria-label="Camera view" />
        <svg className="overlay" aria-hidden="true" width="100%" height="100%" data-count={overlay ? detections.length : 0}>
          {overlay && detections.map(d => {
            const x = frameRect.x + d.box[0] * frameRect.width;
            const y = frameRect.y + d.box[1] * frameRect.height;
            const scoreX = Math.min(Math.max(x, frameRect.x + 2), frameRect.x + frameRect.width - 49);
            const scoreY = Math.min(Math.max(y - 27, frameRect.y + 2), frameRect.y + frameRect.height - 27);
            return <g key={d.id} data-track={d.id}>
              <rect className="hazard-box" x={x} y={y} width={d.box[2] * frameRect.width} height={d.box[3] * frameRect.height} rx="3" />
              <g transform={`translate(${scoreX},${scoreY})`}>
                <rect className="score-background" width="47" height="25" rx="6" />
                <text className="threat-score" x="23.5" y="17.5" textAnchor="middle">{d.severity.toFixed(2)}</text>
              </g>
            </g>;
          })}
        </svg>
        {input === 'camera' && <button className={`motion-indicator ${motion.state.toLowerCase()}`} aria-label={`Motion ${motion.state.toLowerCase()}, ${liveStatus.replaceAll('_', ' ')}`} onClick={() => {
          setError(motion.state === 'ALERT' ? 'Rapid approach or movement across your path detected. The beep comes from local motion analysis.' : motion.state === 'UNKNOWN' ? 'Motion is not clear enough to assess. Keep the camera still with a clear view. Model A continues detecting objects.' : 'Keep the camera still. Fast approach or crossing triggers local beeps. All Model A boxes appear on their analyzed snapshot.'); setShowError(true);
        }}><Icon name={motion.state === 'ALERT' ? 'alert' : 'motion'} /></button>}
        {input === 'camera' && overlay && liveView && <AnalysisPreview view={liveView} bounds={frameRect} />}
        {loading && <div className="loading" role="status" aria-label="Opening input"><span /></div>}
        {paused && !loading && <button className="resume-button" aria-label="Resume" onClick={resume}><Icon name="play" /></button>}
        {error && <button className="error-indicator" aria-label="Voice or input needs attention" onClick={() => setShowError(true)}><Icon name="alert" /></button>}
      </div>
      <div className="control-panel" role="toolbar" aria-label="Detection controls">
        <button aria-label={overlay ? 'Hide detection boxes' : 'Show detection boxes'} aria-pressed={overlay} className={`control ${overlay ? 'selected' : ''}`} onClick={() => setOverlay(v => !v)}><Icon name="boxes" /></button>
        <button aria-label={sound && !audioReady ? 'Enable audio playback' : sound ? 'Mute spoken warnings' : 'Enable spoken warnings'} aria-pressed={sound && audioReady} className={`control ${sound && !audioReady ? 'audio-blocked' : ''}`} onClick={() => {
          if (!sound || !audioReady) { soundRef.current = true; setSound(true); alerts.current?.setMuted(false); live.current?.setMuted(false); void unlock(true); }
          else { soundRef.current = false; setSound(false); alerts.current?.setMuted(true); live.current?.setMuted(true); audio.current?.stop(); }
        }}><Icon name={sound ? 'sound' : 'muted'} /></button>
        <button className="control stop-control" aria-label="Stop detecting" onClick={stop}><Icon name="stop" /></button>
      </div>
    </section>
    <dialog className="source-dialog" ref={sourceDialog} onCancel={() => setScreen('home')} onClick={e => { if (e.target === e.currentTarget) setScreen('home'); }}>
      <div className="sheet"><button className="close-button" aria-label="Close input choices" onClick={() => setScreen('home')}><Icon name="close" /></button>
        <h1>Choose input</h1>
        {DEMOS.map(demo => <button key={demo.id} className="source-button" onClick={() => choose('demo', demo.id)}><Icon name="video" /><span>{demo.title}</span><Icon name="arrow" /></button>)}
        <button className="source-button camera-choice" onClick={() => choose('camera')}><Icon name="camera" /><span>Live camera<small>Model A</small></span><Icon name="arrow" /></button>
      </div>
    </dialog>
    <dialog ref={errorDialog} className="error-dialog" onCancel={() => setShowError(false)}>
      <Icon name="alert" /><p>{error || 'Try again.'}</p><div><button onClick={() => { setShowError(false); resume(); }}>Retry</button><button onClick={stop}>Back</button></div>
    </dialog>
  </main>;
}
