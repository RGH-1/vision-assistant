import { useEffect, useMemo, useState } from 'react';
import { ApiError, VisionAssistantApiClient } from './api/client';
import { resolveDashboardConfig } from './config';
import { useDashboardSocket } from './hooks/useDashboardSocket';
import type { Lang } from './types/contracts';

export function App() {
  const config = useMemo(() => resolveDashboardConfig(), []);
  const apiClient = useMemo(
    () => new VisionAssistantApiClient(config.serverUrl),
    [config.serverUrl],
  );

  const {
    connectionState,
    isConnected,
    reconnectAttempt,
    lastStatus,
    recentFrames,
    recentQuestions,
    recentAlerts,
    setInitialStatus,
  } = useDashboardSocket({
    wsUrl: config.wsDashboardUrl,
  });

  const [typedQuestion, setTypedQuestion] = useState<string>('');
  const [questionLang, setQuestionLang] = useState<Lang>('en');
  const [searchQuery, setSearchQuery] = useState<string>('');
  const [controlFeedback, setControlFeedback] = useState<string | null>(null);
  const [isSubmittingAsk, setIsSubmittingAsk] = useState<boolean>(false);

  // Seed initial status from GET /v1/status on mount
  useEffect(() => {
    let cancelled = false;
    apiClient
      .getStatus()
      .then((status) => {
        if (!cancelled) {
          setInitialStatus(status);
        }
      })
      .catch(() => {
        // WebSocket status stream will populate status when backend is reachable
      });
    return () => {
      cancelled = true;
    };
  }, [apiClient, setInitialStatus]);

  const activeSessionId = lastStatus?.session_id ?? null;

  const handleAskByTyping = async (e: React.FormEvent) => {
    e.preventDefault();
    const trimmed = typedQuestion.trim();
    if (!trimmed) return;
    if (!activeSessionId) {
      setControlFeedback('No active session yet. Start the mobile app or create a session first.');
      return;
    }

    setIsSubmittingAsk(true);
    setControlFeedback(null);
    try {
      const res = await apiClient.askText({
        session_id: activeSessionId,
        text: trimmed,
        lang: questionLang,
      });
      setTypedQuestion('');
      setControlFeedback(`Answered (#${res.question_id}): ${res.answer}`);
    } catch (err) {
      if (err instanceof ApiError) {
        setControlFeedback(`Error (${err.code}): ${err.message}`);
      } else {
        const detail = err instanceof Error ? err.message : String(err);
        setControlFeedback(
          `Failed to send question to ${config.serverUrl} (${detail}). Make sure the Python server window is running and open the dashboard at http://localhost:5173.`,
        );
      }
    } finally {
      setIsSubmittingAsk(false);
    }
  };

  const handleEndSession = async () => {
    if (!activeSessionId) {
      setControlFeedback('No active session to end.');
      return;
    }
    try {
      await apiClient.endSession(activeSessionId);
      setControlFeedback(`Ended session ${activeSessionId}.`);
    } catch (err) {
      setControlFeedback(
        err instanceof ApiError ? `Error (${err.code}): ${err.message}` : 'Could not end session.',
      );
    }
  };

  const handleDeleteAllData = async () => {
    if (!activeSessionId) {
      setControlFeedback('No active session to delete.');
      return;
    }
    try {
      await apiClient.deleteSession(activeSessionId);
      setControlFeedback(`Deleted session ${activeSessionId} and all stored data.`);
    } catch (err) {
      setControlFeedback(
        err instanceof ApiError
          ? `Error (${err.code}): ${err.message}`
          : 'Could not delete session data.',
      );
    }
  };

  const badgeStyle = isConnected
    ? 'bg-emerald-500/15 text-emerald-300 border-emerald-500/40'
    : connectionState === 'reconnecting'
      ? 'bg-amber-500/15 text-amber-300 border-amber-500/40'
      : 'bg-rose-500/15 text-rose-300 border-rose-500/40';

  const badgeDotStyle = isConnected
    ? 'bg-emerald-400 animate-pulse'
    : connectionState === 'reconnecting'
      ? 'bg-amber-400 animate-ping'
      : 'bg-rose-400';

  const badgeText = isConnected
    ? 'Connected'
    : connectionState === 'reconnecting'
      ? `Reconnecting (retry #${reconnectAttempt})`
      : connectionState === 'connecting'
        ? 'Connecting...'
        : 'Disconnected';

  return (
    <div className="min-h-screen bg-slate-950 text-slate-100 flex flex-col">
      {/* Header & Controls Bar */}
      <header className="border-b border-slate-800 bg-slate-900/90 backdrop-blur px-6 py-4">
        <div className="max-w-7xl mx-auto flex flex-col gap-4 lg:flex-row lg:items-center lg:justify-between">
          <div className="flex flex-wrap items-center gap-3">
            <h1 className="text-xl font-bold tracking-tight text-white">
              Vision Assistant Dashboard
            </h1>

            {/* Live WebSocket Connection Badge */}
            <div
              data-testid="ws-connection-badge"
              data-state={connectionState}
              className={`inline-flex items-center gap-2 rounded-full border px-3 py-1 text-xs font-semibold ${badgeStyle}`}
            >
              <span className={`h-2 w-2 rounded-full ${badgeDotStyle}`} />
              <span>{badgeText}</span>
            </div>

            <span className="text-xs font-mono text-slate-400 bg-slate-800/80 px-2.5 py-1 rounded border border-slate-700">
              API: {config.serverUrl}
            </span>

            <span className="text-xs font-mono text-slate-400 bg-slate-800/80 px-2.5 py-1 rounded border border-slate-700">
              Session: {activeSessionId ? `${activeSessionId.slice(0, 8)}…` : 'None'}
            </span>
          </div>

          {/* Controls Bar (§6.7: Ask by typing, End session, Delete all data) */}
          <div className="flex flex-wrap items-center gap-2" aria-label="Dashboard controls">
            <form onSubmit={handleAskByTyping} className="flex items-center gap-2">
              <select
                aria-label="Question language"
                value={questionLang}
                onChange={(e) => setQuestionLang(e.target.value as Lang)}
                className="rounded-lg border border-slate-700 bg-slate-800 px-2.5 py-1.5 text-xs font-medium text-slate-200 focus:border-indigo-500 focus:outline-none"
              >
                <option value="en">EN</option>
                <option value="ar">AR (العربية)</option>
                <option value="fr">FR</option>
              </select>
              <input
                type="text"
                dir="auto"
                value={typedQuestion}
                onChange={(e) => setTypedQuestion(e.target.value)}
                placeholder="Ask by typing (EN / العربية / FR)…"
                className="w-56 sm:w-64 rounded-lg border border-slate-700 bg-slate-900 px-3 py-1.5 text-xs text-slate-100 placeholder-slate-500 focus:border-indigo-500 focus:outline-none"
              />
              <button
                type="submit"
                disabled={isSubmittingAsk}
                className="rounded-lg bg-indigo-600 hover:bg-indigo-500 disabled:opacity-50 px-3 py-1.5 text-xs font-semibold text-white transition"
              >
                {isSubmittingAsk ? 'Sending…' : 'Ask'}
              </button>
            </form>

            <button
              type="button"
              onClick={handleEndSession}
              className="rounded-lg border border-slate-700 bg-slate-800 hover:bg-slate-700 px-3 py-1.5 text-xs font-medium text-slate-200 transition"
            >
              End session
            </button>

            <button
              type="button"
              onClick={handleDeleteAllData}
              className="rounded-lg border border-rose-500/40 bg-rose-500/15 hover:bg-rose-500/25 px-3 py-1.5 text-xs font-semibold text-rose-300 transition"
            >
              Delete all data
            </button>
          </div>
        </div>

        {controlFeedback && (
          <div
            dir="auto"
            role="status"
            className="max-w-7xl mx-auto mt-3 rounded-lg border border-indigo-500/30 bg-indigo-950/40 px-3 py-2 text-xs text-indigo-200"
          >
            {controlFeedback}
          </div>
        )}
      </header>

      {/* Main 4-Panel Grid Layout (§6.7) */}
      <main className="flex-1 max-w-7xl w-full mx-auto p-6 grid grid-cols-1 lg:grid-cols-2 gap-6">
        {/* Panel 1: Live Stream & System Status */}
        <section
          aria-labelledby="panel-live-title"
          data-testid="panel-live-status"
          className="rounded-xl border border-slate-800 bg-slate-900/70 p-5 flex flex-col gap-4"
        >
          <div className="flex items-center justify-between border-b border-slate-800 pb-3">
            <div>
              <h2 id="panel-live-title" className="text-base font-semibold text-white">
                1. Live Camera &amp; System Status
              </h2>
              <p className="text-xs text-slate-400">
                WebRTC (WHEP) stream preview &amp; 2s telemetry from <code className="font-mono">/v1/ws/dashboard</code>
              </p>
            </div>
            <span
              className={`rounded-full px-2.5 py-0.5 text-xs font-medium ${
                lastStatus?.stream_connected
                  ? 'bg-emerald-500/20 text-emerald-300'
                  : 'bg-slate-800 text-slate-400'
              }`}
            >
              {lastStatus?.stream_connected ? 'Stream Live' : 'Stream Idle'}
            </span>
          </div>

          {/* Live Video / WHEP Placeholder Box */}
          <div className="aspect-video w-full rounded-lg border border-slate-800 bg-slate-950 flex flex-col items-center justify-center p-4 text-center">
            <p className="text-sm font-medium text-slate-300" dir="auto">
              Live WebRTC (WHEP) Stream Panel Placeholder
            </p>
            <p className="mt-1 text-xs font-mono text-slate-500">{config.whepUrl}</p>
          </div>

          {/* Live Status Telemetry Grid */}
          <div className="grid grid-cols-2 sm:grid-cols-4 gap-2.5 text-xs">
            <div className="rounded-lg bg-slate-950/80 border border-slate-800 p-2.5">
              <span className="text-slate-400 block">Input / Analysed FPS</span>
              <span className="font-mono text-sm font-semibold text-white">
                {lastStatus ? `${lastStatus.input_fps} / ${lastStatus.analysed_fps}` : '0 / 0'}
              </span>
            </div>
            <div className="rounded-lg bg-slate-950/80 border border-slate-800 p-2.5">
              <span className="text-slate-400 block">Frames in DB</span>
              <span className="font-mono text-sm font-semibold text-white">
                {lastStatus?.frames_in_db ?? 0} ({lastStatus?.saved_per_min ?? 0}/min)
              </span>
            </div>
            <div className="rounded-lg bg-slate-950/80 border border-slate-800 p-2.5">
              <span className="text-slate-400 block">Retention / GPU</span>
              <span className="font-mono text-sm font-semibold text-white">
                {lastStatus?.retention_min ?? 30}m /{' '}
                {lastStatus?.gpu_mem_mb != null ? `${lastStatus.gpu_mem_mb} MB` : 'N/A'}
              </span>
            </div>
            <div className="rounded-lg bg-slate-950/80 border border-slate-800 p-2.5">
              <span className="text-slate-400 block">Cloud Status</span>
              <div className="flex items-center gap-2 mt-0.5">
                <span
                  className={
                    lastStatus?.groq_ok ? 'text-emerald-400 font-semibold' : 'text-slate-500'
                  }
                >
                  Groq
                </span>
                <span className="text-slate-600">·</span>
                <span
                  className={
                    lastStatus?.gemini_ok ? 'text-emerald-400 font-semibold' : 'text-slate-500'
                  }
                >
                  Gemini
                </span>
              </div>
            </div>
          </div>
        </section>

        {/* Panel 2: Frame Timeline & Hybrid Search */}
        <section
          aria-labelledby="panel-timeline-title"
          data-testid="panel-timeline"
          className="rounded-xl border border-slate-800 bg-slate-900/70 p-5 flex flex-col gap-4"
        >
          <div className="flex items-center justify-between border-b border-slate-800 pb-3">
            <div>
              <h2 id="panel-timeline-title" className="text-base font-semibold text-white">
                2. Frame Timeline &amp; Search by Meaning
              </h2>
              <p className="text-xs text-slate-400">
                Hybrid CLIP + OCR + Object search (<code className="font-mono">GET /v1/frames</code> &amp;{' '}
                <code className="font-mono">frame_stored</code>)
              </p>
            </div>
            <span className="rounded-full bg-slate-800 px-2.5 py-0.5 text-xs font-mono text-slate-300">
              {recentFrames.length} live events
            </span>
          </div>

          {/* Search-by-meaning placeholder input with dir="auto" */}
          <div className="flex items-center gap-2">
            <input
              type="search"
              dir="auto"
              value={searchQuery}
              onChange={(e) => setSearchQuery(e.target.value)}
              placeholder="Search by meaning, OCR text, or object (e.g. staircase / مخرج)…"
              className="flex-1 rounded-lg border border-slate-700 bg-slate-950 px-3 py-2 text-xs text-slate-100 placeholder-slate-500 focus:border-indigo-500 focus:outline-none"
            />
          </div>

          <div className="flex-1 rounded-lg border border-dashed border-slate-800 bg-slate-950/60 p-4 flex flex-col justify-center">
            {recentFrames.length === 0 ? (
              <p className="text-xs text-slate-400 text-center" dir="auto">
                Timeline placeholder — saved frames (<code className="font-mono">frame_stored</code>)
                with thumbnails, YOLOE boxes, OCR text, and before/after enhancement will render here.
              </p>
            ) : (
              <ul className="space-y-2 max-h-56 overflow-y-auto">
                {recentFrames.map((f) => (
                  <li
                    key={f.frame_id}
                    className="rounded border border-slate-800 bg-slate-900 px-3 py-2 text-xs flex items-center justify-between gap-2"
                  >
                    <span className="font-mono text-indigo-300">#{f.frame_id}</span>
                    <span className="text-slate-300 truncate flex-1" dir="auto">
                      {f.text_preview || f.labels.join(', ') || f.keep_reason}
                    </span>
                    <span className="rounded bg-slate-800 px-1.5 py-0.5 text-[10px] text-slate-400">
                      {f.keep_reason}
                    </span>
                  </li>
                ))}
              </ul>
            )}
          </div>
        </section>

        {/* Panel 3: Q&A History & Evidence */}
        <section
          aria-labelledby="panel-qa-title"
          data-testid="panel-qa"
          className="rounded-xl border border-slate-800 bg-slate-900/70 p-5 flex flex-col gap-4"
        >
          <div className="flex items-center justify-between border-b border-slate-800 pb-3">
            <div>
              <h2 id="panel-qa-title" className="text-base font-semibold text-white">
                3. Q&amp;A History &amp; Grounded Evidence
              </h2>
              <p className="text-xs text-slate-400">
                Questions, intents, answers, score parts, and step timings (
                <code className="font-mono">query_answered</code>)
              </p>
            </div>
            <span className="rounded-full bg-slate-800 px-2.5 py-0.5 text-xs font-mono text-slate-300">
              {recentQuestions.length} live
            </span>
          </div>

          <div className="flex-1 rounded-lg border border-dashed border-slate-800 bg-slate-950/60 p-4 flex flex-col justify-center">
            {recentQuestions.length === 0 ? (
              <p className="text-xs text-slate-400 text-center" dir="auto">
                Q&amp;A panel placeholder — spoken and typed questions (EN / العربية / FR) with
                intent badges, confidence, and evidence score parts will appear here in real time.
              </p>
            ) : (
              <ul className="space-y-3 max-h-56 overflow-y-auto">
                {recentQuestions.map((q) => (
                  <li
                    key={q.question_id}
                    className="rounded-lg border border-slate-800 bg-slate-900 p-3 text-xs space-y-1.5"
                  >
                    <div className="flex items-center justify-between gap-2 text-[11px] text-slate-400">
                      <span className="font-mono text-indigo-300">
                        #{q.question_id} · {q.intent} · {q.lang.toUpperCase()}
                      </span>
                      <span>{q.confidence}</span>
                    </div>
                    <p className="font-medium text-slate-200" dir="auto">
                      Q: {q.transcript}
                    </p>
                    <p className="text-emerald-300" dir="auto">
                      A: {q.answer}
                    </p>
                  </li>
                ))}
              </ul>
            )}
          </div>
        </section>

        {/* Panel 4: Proactive Alerts */}
        <section
          aria-labelledby="panel-alerts-title"
          data-testid="panel-alerts"
          className="rounded-xl border border-slate-800 bg-slate-900/70 p-5 flex flex-col gap-4"
        >
          <div className="flex items-center justify-between border-b border-slate-800 pb-3">
            <div>
              <h2 id="panel-alerts-title" className="text-base font-semibold text-white">
                4. Proactive Obstacle &amp; Navigation Alerts
              </h2>
              <p className="text-xs text-slate-400">
                Real-time warnings pushed over <code className="font-mono">/v1/ws/dashboard</code> (
                <code className="font-mono">alert</code> event)
              </p>
            </div>
            <span className="rounded-full bg-slate-800 px-2.5 py-0.5 text-xs font-mono text-slate-300">
              {recentAlerts.length} alerts
            </span>
          </div>

          <div className="flex-1 rounded-lg border border-dashed border-slate-800 bg-slate-950/60 p-4 flex flex-col justify-center">
            {recentAlerts.length === 0 ? (
              <p className="text-xs text-slate-400 text-center" dir="auto">
                Alerts panel placeholder — proactive hazard notifications (
                <code className="font-mono">door_ahead</code>,{' '}
                <code className="font-mono">stairs_ahead</code>,{' '}
                <code className="font-mono">obstacle_ahead</code>,{' '}
                <code className="font-mono">vehicle_near</code>) with thumbnails will display here.
              </p>
            ) : (
              <ul className="space-y-2 max-h-56 overflow-y-auto">
                {recentAlerts.map((a) => (
                  <li
                    key={a.alert_id}
                    className="rounded-lg border border-amber-500/30 bg-amber-950/30 px-3 py-2 text-xs flex items-center justify-between gap-2"
                  >
                    <span className="font-mono text-amber-300">{a.kind}</span>
                    <span className="text-slate-100 flex-1" dir="auto">
                      {a.message}
                    </span>
                    <span className="text-[10px] text-slate-400 uppercase">{a.lang}</span>
                  </li>
                ))}
              </ul>
            )}
          </div>
        </section>
      </main>
    </div>
  );
}

export default App;

