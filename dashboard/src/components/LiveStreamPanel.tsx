import { useEffect, useMemo, useRef, useState } from 'react';
import { DEFAULT_MEDIAMTX_HOST, normalizeBaseUrl } from '../config';
import type { FrameDetection, Status } from '../types/contracts';

export type StreamViewerMode = 'whep' | 'iframe';
export type StreamPathOption = 'mystream' | 'phone';
export type WhepClientState = 'connecting' | 'playing' | 'retrying' | 'idle';

export const DEFAULT_WHEP_RETRY_MS = 3500;

export interface LiveStreamPanelProps {
  /** Whether the dashboard WebSocket (/v1/ws/dashboard) is currently connected */
  wsConnected: boolean;
  /** Latest real-time status event from /v1/ws/dashboard */
  status: Status | null;
  /** Base WHEP endpoint URL (e.g. http://192.168.8.111:8889/mystream/whep) */
  defaultWhepUrl?: string;
  /** Optional bounding boxes ([x1, y1, x2, y2] in 0..1) for the overlay */
  detections?: FrameDetection[];
  /** Retry interval in ms when WHEP POST fails (e.g. 404 Not Found) */
  retryIntervalMs?: number;
}

interface OfferSdpData {
  iceUfrag: string;
  icePwd: string;
  medias: string[];
}

/**
 * Extracts ice-ufrag, ice-pwd, and media descriptions from an SDP offer
 * so Trickle ICE PATCH fragments can be sent to MediaMTX WHEP sessions.
 */
function parseOfferSdp(sdp: string): OfferSdpData {
  const ret: OfferSdpData = {
    iceUfrag: '',
    icePwd: '',
    medias: [],
  };

  for (const line of sdp.split(/\r?\n/)) {
    if (line.startsWith('m=')) {
      ret.medias.push(line.slice('m='.length));
    } else if (ret.iceUfrag === '' && line.startsWith('a=ice-ufrag:')) {
      ret.iceUfrag = line.slice('a=ice-ufrag:'.length);
    } else if (ret.icePwd === '' && line.startsWith('a=ice-pwd:')) {
      ret.icePwd = line.slice('a=ice-pwd:'.length);
    }
  }

  return ret;
}

/**
 * Builds an `application/trickle-ice-sdpfrag` body matching MediaMTX's WHEP specification.
 */
function generateSdpFragment(
  offerData: OfferSdpData,
  candidates: RTCIceCandidate[],
): string {
  const candidatesByMedia: Record<number, RTCIceCandidate[]> = {};
  for (const candidate of candidates) {
    const mid = candidate.sdpMLineIndex ?? 0;
    if (candidatesByMedia[mid] === undefined) {
      candidatesByMedia[mid] = [];
    }
    candidatesByMedia[mid].push(candidate);
  }

  let frag = `a=ice-ufrag:${offerData.iceUfrag}\r\na=ice-pwd:${offerData.icePwd}\r\n`;

  let mid = 0;
  for (const media of offerData.medias) {
    const mediaCandidates = candidatesByMedia[mid];
    if (mediaCandidates !== undefined && mediaCandidates.length > 0) {
      frag += `m=${media}\r\na=mid:${mid}\r\n`;
      for (const candidate of mediaCandidates) {
        if (candidate.candidate) {
          frag += `a=${candidate.candidate}\r\n`;
        }
      }
    }
    mid++;
  }

  return frag;
}

/**
 * Extracts the MediaMTX base origin (e.g. http://192.168.8.111:8889) and initial path
 * ('mystream' | 'phone') from a full WHEP URL.
 */
function parseWhepEndpoint(rawWhepUrl: string): {
  baseOrigin: string;
  initialPath: StreamPathOption;
} {
  try {
    const parsed = new URL(rawWhepUrl);
    const segments = parsed.pathname.split('/').filter(Boolean);
    const firstSegment = segments[0];
    const initialPath: StreamPathOption = firstSegment === 'phone' ? 'phone' : 'mystream';
    return {
      baseOrigin: `${parsed.protocol}//${parsed.host}`,
      initialPath,
    };
  } catch {
    return {
      baseOrigin: `http://${DEFAULT_MEDIAMTX_HOST}`,
      initialPath: 'mystream',
    };
  }
}

export function LiveStreamPanel({
  wsConnected,
  status,
  defaultWhepUrl = `http://${DEFAULT_MEDIAMTX_HOST}/mystream/whep`,
  detections = [],
  retryIntervalMs = DEFAULT_WHEP_RETRY_MS,
}: LiveStreamPanelProps) {
  const parsedEndpoint = useMemo(() => parseWhepEndpoint(defaultWhepUrl), [defaultWhepUrl]);

  const [viewerMode, setViewerMode] = useState<StreamViewerMode>('whep');
  const [streamPath, setStreamPath] = useState<StreamPathOption>(parsedEndpoint.initialPath);
  const [whepState, setWhepState] = useState<WhepClientState>('connecting');
  const [retryCount, setRetryCount] = useState<number>(0);
  const [lastError, setLastError] = useState<string | null>(null);
  const [showOverlay, setShowOverlay] = useState<boolean>(true);
  const [manualReconnectTick, setManualReconnectTick] = useState<number>(0);

  const videoRef = useRef<HTMLVideoElement | null>(null);

  const activeWhepUrl = `${normalizeBaseUrl(parsedEndpoint.baseOrigin)}/${streamPath}/whep`;
  const activeIframeUrl = `${normalizeBaseUrl(parsedEndpoint.baseOrigin)}/${streamPath}`;

  // Derive the real-time connection badge from /v1/ws/dashboard status + WHEP playback state
  const isStreamLive = Boolean(status?.stream_connected) || whepState === 'playing';
  const badgeLabel = isStreamLive
    ? 'Live'
    : wsConnected
      ? 'Waiting for Stream'
      : 'Disconnected';

  const badgeClasses = isStreamLive
    ? 'bg-emerald-500/20 text-emerald-300 border-emerald-500/40'
    : wsConnected
      ? 'bg-amber-500/20 text-amber-300 border-amber-500/40'
      : 'bg-rose-500/20 text-rose-300 border-rose-500/40';

  const badgeDotClasses = isStreamLive
    ? 'bg-emerald-400 animate-pulse'
    : wsConnected
      ? 'bg-amber-400 animate-ping'
      : 'bg-rose-400';

  // Trigger an immediate retry if the backend WS reports stream_connected = true while waiting
  const backendStreamConnected = Boolean(status?.stream_connected);
  useEffect(() => {
    if (backendStreamConnected && whepState === 'retrying') {
      setManualReconnectTick((prev) => prev + 1);
    }
  }, [backendStreamConnected, whepState]);

  useEffect(() => {
    if (viewerMode !== 'whep') {
      setWhepState('idle');
      return;
    }

    let cancelled = false;
    let activePc: RTCPeerConnection | null = null;
    let retryTimer: ReturnType<typeof setTimeout> | null = null;
    let abortController: AbortController | null = null;

    const clearRetryTimer = () => {
      if (retryTimer !== null) {
        clearTimeout(retryTimer);
        retryTimer = null;
      }
    };

    const cleanupPeerConnection = () => {
      if (abortController) {
        abortController.abort();
        abortController = null;
      }
      if (activePc) {
        activePc.ontrack = null;
        activePc.onicecandidate = null;
        activePc.onconnectionstatechange = null;
        activePc.oniceconnectionstatechange = null;
        try {
          activePc.close();
        } catch {
          // Ignore close errors
        }
        activePc = null;
      }
      const videoEl = videoRef.current;
      if (videoEl && videoEl.srcObject) {
        videoEl.srcObject = null;
      }
    };

    const scheduleRetry = (reason: string) => {
      if (cancelled) return;
      cleanupPeerConnection();
      clearRetryTimer();
      setWhepState('retrying');
      setLastError(reason);
      setRetryCount((prev) => prev + 1);

      retryTimer = setTimeout(() => {
        if (!cancelled) {
          void startWhepSession();
        }
      }, retryIntervalMs);
    };

    const startWhepSession = async () => {
      if (cancelled) return;
      clearRetryTimer();
      cleanupPeerConnection();
      setWhepState('connecting');

      if (typeof RTCPeerConnection === 'undefined') {
        scheduleRetry('WebRTC RTCPeerConnection is not supported in this environment.');
        return;
      }

      const currentPc = new RTCPeerConnection();
      activePc = currentPc;

      let offerData: OfferSdpData | null = null;
      let sessionUrl: string | null = null;
      let remoteDescSet = false;
      const queuedCandidates: RTCIceCandidate[] = [];

      const sendTrickleCandidates = (candidates: RTCIceCandidate[]) => {
        if (cancelled || activePc !== currentPc || !sessionUrl || !offerData) {
          return;
        }
        const valid = candidates.filter((c) => Boolean(c.candidate));
        if (valid.length === 0) {
          return;
        }
        const fragment = generateSdpFragment(offerData, valid);
        void globalThis
          .fetch(sessionUrl, {
            method: 'PATCH',
            headers: {
              'Content-Type': 'application/trickle-ice-sdpfrag',
              'If-Match': '*',
            },
            body: fragment,
          })
          .catch(() => {
            // Ignore non-fatal trickle ICE errors if candidates were already accepted
          });
      };

      try {
        // 1. Add video transceiver set to recvonly
        currentPc.addTransceiver('video', { direction: 'recvonly' });

        // 2. Collect and trickle local ICE candidates to MediaMTX WHEP session URL
        currentPc.onicecandidate = (event: RTCPeerConnectionIceEvent) => {
          if (cancelled || activePc !== currentPc) return;
          if (event.candidate && event.candidate.candidate) {
            if (!remoteDescSet || !sessionUrl) {
              queuedCandidates.push(event.candidate);
            } else {
              sendTrickleCandidates([event.candidate]);
            }
          }
        };

        // 3. Attach resulting MediaStream from ontrack to the <video> element's srcObject
        currentPc.ontrack = (event: RTCTrackEvent) => {
          if (cancelled || activePc !== currentPc) return;
          const stream =
            event.streams && event.streams[0]
              ? event.streams[0]
              : new MediaStream([event.track]);
          const videoEl = videoRef.current;
          if (videoEl && videoEl.srcObject !== stream) {
            videoEl.srcObject = stream;
          }
          setWhepState('playing');
          setRetryCount(0);
          setLastError(null);
        };

        const handleStateChange = () => {
          if (cancelled || activePc !== currentPc) return;
          const connState = currentPc.connectionState;
          const iceState = currentPc.iceConnectionState;

          if (connState === 'connected' || iceState === 'connected' || iceState === 'completed') {
            setWhepState('playing');
            setRetryCount(0);
            setLastError(null);
          } else if (
            connState === 'failed' ||
            connState === 'closed' ||
            iceState === 'failed' ||
            iceState === 'closed'
          ) {
            scheduleRetry('WebRTC stream disconnected — waiting to reconnect…');
          }
        };

        currentPc.onconnectionstatechange = handleStateChange;
        currentPc.oniceconnectionstatechange = handleStateChange;

        // 4. Create SDP offer and set as local description
        const offer = await currentPc.createOffer();
        if (cancelled || activePc !== currentPc) return;

        const offerSdp = offer.sdp ?? '';
        if (!offerSdp) {
          scheduleRetry('Failed to generate WebRTC SDP offer.');
          return;
        }
        offerData = parseOfferSdp(offerSdp);

        await currentPc.setLocalDescription(offer);
        if (cancelled || activePc !== currentPc) return;

        // 5. POST SDP offer to WHEP endpoint with Content-Type: application/sdp
        abortController = new AbortController();
        const response = await globalThis.fetch(activeWhepUrl, {
          method: 'POST',
          headers: {
            'Content-Type': 'application/sdp',
          },
          body: offerSdp,
          signal: abortController.signal,
        });

        if (cancelled || activePc !== currentPc) return;

        if (!response.ok) {
          scheduleRetry(
            response.status === 404
              ? `Stream not published yet at /${streamPath} (HTTP 404). Retrying…`
              : `WHEP handshake returned HTTP ${response.status}. Retrying…`,
          );
          return;
        }

        const locationHeader =
          response.headers?.get?.('Location') ?? response.headers?.get?.('location') ?? null;
        if (locationHeader) {
          try {
            sessionUrl = new URL(locationHeader, activeWhepUrl).toString();
          } catch {
            sessionUrl = null;
          }
        }

        // 6. Take SDP answer from HTTP 201 (or 200) response and pass to setRemoteDescription
        const answerSdp = await response.text();
        if (cancelled || activePc !== currentPc) return;

        await currentPc.setRemoteDescription({
          type: 'answer',
          sdp: answerSdp,
        });
        if (cancelled || activePc !== currentPc) return;

        remoteDescSet = true;
        if (queuedCandidates.length > 0 && sessionUrl) {
          sendTrickleCandidates(queuedCandidates.splice(0, queuedCandidates.length));
        }
      } catch (err) {
        if (cancelled || activePc !== currentPc) return;
        const msg = err instanceof Error ? err.message : 'Unable to reach MediaMTX WHEP server';
        scheduleRetry(`Waiting for stream (${msg}). Retrying…`);
      }
    };

    void startWhepSession();

    return () => {
      cancelled = true;
      clearRetryTimer();
      cleanupPeerConnection();
    };
  }, [
    viewerMode,
    activeWhepUrl,
    streamPath,
    retryIntervalMs,
    manualReconnectTick,
  ]);

  return (
    <section
      aria-labelledby="panel-live-title"
      data-testid="panel-live-status"
      className="rounded-xl border border-slate-800 bg-slate-900/70 p-5 flex flex-col gap-4"
    >
      {/* Header Row: Title, Stream Connection Badge, Path Selector, and Fallback Toggle */}
      <div className="flex flex-wrap items-center justify-between gap-3 border-b border-slate-800 pb-3">
        <div>
          <h2 id="panel-live-title" className="text-base font-semibold text-white">
            1. Live Camera &amp; System Status
          </h2>
          <p className="text-xs text-slate-400">
            WebRTC (WHEP) stream viewer &amp; 2s telemetry from{' '}
            <code className="font-mono">/v1/ws/dashboard</code>
          </p>
        </div>

        <div className="flex flex-wrap items-center gap-2">
          {/* Stream Connection Badge driven by /v1/ws/dashboard + WHEP state */}
          <div
            data-testid="stream-connection-badge"
            data-status={badgeLabel}
            className={`inline-flex items-center gap-1.5 rounded-full border px-2.5 py-0.5 text-xs font-semibold ${badgeClasses}`}
          >
            <span className={`h-2 w-2 rounded-full ${badgeDotClasses}`} />
            <span>{badgeLabel}</span>
          </div>

          {/* Stream Path Selector (/mystream vs /phone) */}
          <select
            aria-label="Stream path"
            data-testid="stream-path-select"
            value={streamPath}
            onChange={(e) => setStreamPath(e.target.value as StreamPathOption)}
            className="rounded-lg border border-slate-700 bg-slate-800 px-2 py-1 text-xs font-mono text-slate-200 focus:border-indigo-500 focus:outline-none"
          >
            <option value="mystream">/mystream</option>
            <option value="phone">/phone</option>
          </select>

          {/* Bounding Box Overlay Toggle */}
          {viewerMode === 'whep' && (
            <button
              type="button"
              data-testid="toggle-bbox-overlay"
              onClick={() => setShowOverlay((prev) => !prev)}
              className={`rounded-lg border px-2.5 py-1 text-xs font-medium transition ${
                showOverlay
                  ? 'border-indigo-500/40 bg-indigo-500/15 text-indigo-300'
                  : 'border-slate-700 bg-slate-800 text-slate-400'
              }`}
            >
              Boxes: {showOverlay ? 'On' : 'Off'}
            </button>
          )}

          {/* Fallback Toggle: WHEP <video> vs <iframe /> */}
          <button
            type="button"
            data-testid="toggle-stream-mode"
            onClick={() =>
              setViewerMode((prev) => (prev === 'whep' ? 'iframe' : 'whep'))
            }
            className="rounded-lg border border-slate-700 bg-slate-800 hover:bg-slate-700 px-2.5 py-1 text-xs font-medium text-slate-200 transition"
          >
            {viewerMode === 'whep' ? 'Use Iframe Fallback' : 'Use WHEP Player'}
          </button>
        </div>
      </div>

      {/* Video / Iframe Stage + Bounding Box Overlay */}
      <div className="relative aspect-video w-full overflow-hidden rounded-lg border border-slate-800 bg-slate-950">
        {viewerMode === 'whep' ? (
          <>
            <video
              ref={videoRef}
              data-testid="whep-video"
              autoPlay
              playsInline
              muted
              onLoadedMetadata={(e) => {
                const playPromise = e.currentTarget.play();
                if (playPromise && typeof playPromise.catch === 'function') {
                  playPromise.catch(() => {});
                }
              }}
              className="h-full w-full object-contain bg-black"
            />

            {/* Transparent Bounding Box Overlay Scaffold (§6.7) */}
            <div
              data-testid="bbox-overlay"
              aria-label="Object detection bounding box overlay"
              className="pointer-events-none absolute inset-0 overflow-hidden"
            >
              {showOverlay &&
                detections.map((det, idx) => {
                  const [x1, y1, x2, y2] = det.bbox;
                  return (
                    <div
                      key={`${det.label}-${idx}`}
                      data-testid="bbox-item"
                      style={{
                        left: `${x1 * 100}%`,
                        top: `${y1 * 100}%`,
                        width: `${Math.max(0, x2 - x1) * 100}%`,
                        height: `${Math.max(0, y2 - y1) * 100}%`,
                      }}
                      className="absolute border-2 border-emerald-400 bg-emerald-400/10 rounded-sm"
                    >
                      <span
                        dir="auto"
                        className="inline-block bg-emerald-500 px-1.5 py-0.5 text-[10px] font-bold text-slate-950"
                      >
                        {det.label} ({Math.round(det.confidence * 100)}%)
                      </span>
                    </div>
                  );
                })}
            </div>

            {/* Status / Retry Overlay when WHEP is not yet playing */}
            {whepState !== 'playing' && (
              <div
                data-testid="whep-status-overlay"
                className="absolute inset-0 flex flex-col items-center justify-center bg-slate-950/85 p-4 text-center"
              >
                <p className="text-sm font-medium text-slate-200" dir="auto">
                  {whepState === 'connecting'
                    ? 'Connecting to WebRTC (WHEP) stream…'
                    : 'Waiting for phone camera stream…'}
                </p>
                <p className="mt-1 text-xs font-mono text-slate-400">{activeWhepUrl}</p>
                {lastError && (
                  <p className="mt-2 max-w-md text-xs text-amber-300/90" dir="auto">
                    {lastError} {retryCount > 0 ? `(attempt #${retryCount})` : ''}
                  </p>
                )}
                <button
                  type="button"
                  onClick={() => setManualReconnectTick((prev) => prev + 1)}
                  className="mt-3 rounded-lg border border-slate-700 bg-slate-800 hover:bg-slate-700 px-3 py-1 text-xs font-medium text-slate-200 transition"
                >
                  Retry now
                </button>
              </div>
            )}
          </>
        ) : (
          <iframe
            data-testid="iframe-fallback"
            src={activeIframeUrl}
            title="MediaMTX Live Stream Fallback"
            className="h-full w-full border-0"
            allow="autoplay; fullscreen"
          />
        )}
      </div>

      {/* Live Status Telemetry Grid */}
      <div className="grid grid-cols-2 sm:grid-cols-4 gap-2.5 text-xs">
        <div className="rounded-lg bg-slate-950/80 border border-slate-800 p-2.5">
          <span className="text-slate-400 block">Input / Analysed FPS</span>
          <span className="font-mono text-sm font-semibold text-white">
            {status ? `${status.input_fps} / ${status.analysed_fps}` : '0 / 0'}
          </span>
        </div>
        <div className="rounded-lg bg-slate-950/80 border border-slate-800 p-2.5">
          <span className="text-slate-400 block">Frames in DB</span>
          <span className="font-mono text-sm font-semibold text-white">
            {status?.frames_in_db ?? 0} ({status?.saved_per_min ?? 0}/min)
          </span>
        </div>
        <div className="rounded-lg bg-slate-950/80 border border-slate-800 p-2.5">
          <span className="text-slate-400 block">Retention / GPU</span>
          <span className="font-mono text-sm font-semibold text-white">
            {status?.retention_min ?? 30}m /{' '}
            {status?.gpu_mem_mb != null ? `${status.gpu_mem_mb} MB` : 'N/A'}
          </span>
        </div>
        <div className="rounded-lg bg-slate-950/80 border border-slate-800 p-2.5">
          <span className="text-slate-400 block">Cloud Status</span>
          <div className="flex items-center gap-2 mt-0.5">
            <span
              className={
                status?.groq_ok ? 'text-emerald-400 font-semibold' : 'text-slate-500'
              }
            >
              Groq
            </span>
            <span className="text-slate-600">·</span>
            <span
              className={
                status?.gemini_ok ? 'text-emerald-400 font-semibold' : 'text-slate-500'
              }
            >
              Gemini
            </span>
          </div>
        </div>
      </div>
    </section>
  );
}

export default LiveStreamPanel;
