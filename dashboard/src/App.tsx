import { useEffect, useMemo, useState } from 'react';
import { ApiError, VisionAssistantApiClient } from './api/client';
import { LiveStreamPanel } from './components/LiveStreamPanel';
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
    ? 'bg-emerald-500/15 text-emerald-300 border-emerald-500/40 shadow-[0_0_12px_rgba(16,185,129,0.18)]'
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
    <div className="min-h-screen bg-[#060913] bg-[radial-gradient(ellipse_80%_50%_at_50%_-15%,rgba(56,189,248,0.13),rgba(99,102,241,0.05),transparent)] text-slate-100 flex flex-col selection:bg-sky-500/30">
      {/* Header & Controls Bar */}
      <header className="sticky top-0 z-30 border-b border-slate-800/80 bg-[#090D1A]/85 backdrop-blur-xl px-6 py-4 shadow-lg shadow-black/30">
        <div className="max-w-7xl mx-auto flex flex-col gap-4 lg:flex-row lg:items-center lg:justify-between">
          <div className="flex flex-wrap items-center gap-3">
            <div className="flex items-center gap-2.5">
              <div className="h-9 w-9 rounded-xl bg-gradient-to-br from-sky-500/25 to-indigo-500/25 border border-sky-400/40 flex items-center justify-center shadow-inner">
                <span className="h-2.5 w-2.5 rounded-full bg-sky-400 shadow-[0_0_10px_#38bdf8]" />
              </div>
              <div>
                <h1 className="text-lg sm:text-xl font-extrabold tracking-tight bg-gradient-to-r from-white via-sky-100 to-indigo-200 bg-clip-text text-transparent">
                  Vision Assistant Dashboard
                </h1>
              </div>
            </div>

            {/* Live WebSocket Connection Badge */}
            <div
              data-testid="ws-connection-badge"
              data-state={connectionState}
              className={`inline-flex items-center gap-2 rounded-full border px-3 py-1 text-xs font-semibold ${badgeStyle}`}
            >
              <span className={`h-2 w-2 rounded-full ${badgeDotStyle}`} />
              <span>{badgeText}</span>
            </div>

            <span className="text-xs font-mono text-slate-300 bg-slate-900/90 px-2.5 py-1 rounded-lg border border-slate-800 shadow-inner">
              API: {config.serverUrl}
            </span>

            <span className="text-xs font-mono text-sky-300/90 bg-sky-950/30 px-2.5 py-1 rounded-lg border border-sky-500/25">
              Session: {activeSessionId ? `${activeSessionId.slice(0, 8)}…` : 'None'}
            </span>
          </div>

          {/* Controls Bar (§6.7: Ask by typing, End session, Delete all data) */}
          <div className="flex flex-wrap items-center gap-2.5" aria-label="Dashboard controls">
            <form
              onSubmit={handleAskByTyping}
              className="flex items-center gap-1.5 rounded-xl bg-slate-900/90 p-1 border border-slate-800"
            >
              <select
                aria-label="Question language"
                value={questionLang}
                onChange={(e) => setQuestionLang(e.target.value as Lang)}
                className="rounded-lg border border-slate-700/80 bg-slate-800/90 px-2.5 py-1.5 text-xs font-semibold text-slate-200 focus:border-sky-500 focus:outline-none"
              >
                <option value="en">EN</option>
                <option value="ar">AR (العربية)</option>
              </select>
              <input
                type="text"
                dir="auto"
                value={typedQuestion}
                onChange={(e) => setTypedQuestion(e.target.value)}
                placeholder="Ask by typing (EN / العربية)…"
                className="w-56 sm:w-64 rounded-lg bg-slate-950/80 border border-slate-800 px-3 py-1.5 text-xs text-slate-100 placeholder-slate-500 focus:border-sky-500 focus:outline-none"
              />
              <button
                type="submit"
                disabled={isSubmittingAsk}
                className="rounded-lg bg-gradient-to-r from-sky-500 to-indigo-600 hover:from-sky-400 hover:to-indigo-500 disabled:opacity-50 px-3.5 py-1.5 text-xs font-bold text-white shadow-sm shadow-indigo-500/30 transition"
              >
                {isSubmittingAsk ? 'Sending…' : 'Ask'}
              </button>
            </form>

            <button
              type="button"
              onClick={handleEndSession}
              className="rounded-xl border border-slate-700/80 bg-slate-900/90 hover:bg-slate-800 px-3.5 py-2 text-xs font-semibold text-slate-200 transition"
            >
              End session
            </button>

            <button
              type="button"
              onClick={handleDeleteAllData}
              className="rounded-xl border border-rose-500/40 bg-rose-500/15 hover:bg-rose-500/25 px-3.5 py-2 text-xs font-semibold text-rose-300 transition"
            >
              Delete all data
            </button>
          </div>
        </div>

        {controlFeedback && (
          <div
            dir="auto"
            role="status"
            className="max-w-7xl mx-auto mt-3 rounded-xl border border-sky-500/30 bg-sky-950/35 px-3.5 py-2 text-xs font-medium text-sky-200 shadow-sm"
          >
            {controlFeedback}
          </div>
        )}
      </header>

      {/* Main 4-Panel Grid Layout (§6.7) */}
      <main className="flex-1 max-w-7xl w-full mx-auto p-6 grid grid-cols-1 lg:grid-cols-2 gap-6">
        {/* Panel 1: Live Stream Viewer (WHEP + Iframe Fallback + Bounding Box Overlay + Telemetry) */}
        <LiveStreamPanel
          wsConnected={isConnected}
          status={lastStatus}
          defaultWhepUrl={config.whepUrl}
        />

        {/* Panel 2: Frame Timeline & Hybrid Search */}
        <section
          aria-labelledby="panel-timeline-title"
          data-testid="panel-timeline"
          className="rounded-2xl border border-slate-800/90 bg-gradient-to-b from-slate-900/85 to-[#0A0F1D]/90 p-5 flex flex-col gap-4 shadow-xl shadow-black/40"
        >
          <div className="flex items-center justify-between border-b border-slate-800/80 pb-3.5">
            <div className="flex items-center gap-3">
              <span className="flex h-7 w-7 items-center justify-center rounded-lg bg-sky-500/15 border border-sky-400/30 text-xs font-bold text-sky-300">
                2
              </span>
              <div>
                <h2 id="panel-timeline-title" className="text-base font-bold text-white">
                  2. Frame Timeline &amp; Search by Meaning
                </h2>
                <p className="text-xs text-slate-400">
                  Hybrid CLIP + OCR + Object search (<code className="font-mono text-sky-300/90">GET /v1/frames</code> &amp;{' '}
                  <code className="font-mono text-sky-300/90">frame_stored</code>)
                </p>
              </div>
            </div>
            <span className="rounded-full bg-sky-500/15 border border-sky-500/30 px-3 py-0.5 text-xs font-mono font-semibold text-sky-300">
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
              className="flex-1 rounded-xl border border-slate-800 bg-[#060913]/90 px-3.5 py-2.5 text-xs text-slate-100 placeholder-slate-500 focus:border-sky-500 focus:outline-none shadow-inner"
            />
          </div>

          <div className="flex-1 rounded-xl border border-slate-800/80 bg-[#060913]/70 p-4 flex flex-col justify-center">
            {recentFrames.length === 0 ? (
              <p className="text-xs text-slate-400 text-center leading-relaxed" dir="auto">
                Timeline placeholder — saved frames (<code className="font-mono text-sky-300">frame_stored</code>)
                with thumbnails, YOLOE boxes, OCR text, and before/after enhancement will render here.
              </p>
            ) : (
              <ul className="space-y-2.5 max-h-64 overflow-y-auto pr-1">
                {recentFrames.map((f) => (
                  <li
                    key={f.frame_id}
                    className="rounded-xl border border-slate-800/90 bg-slate-900/90 hover:border-sky-500/40 px-3 py-2.5 text-xs flex items-center justify-between gap-3 transition"
                  >
                    <div className="flex items-center gap-3 min-w-0 flex-1">
                      {f.thumb_url && (
                        <img
                          src={`${config.serverUrl}${f.thumb_url}`}
                          alt={`Frame #${f.frame_id}`}
                          className="h-10 w-14 rounded-lg object-cover border border-slate-700 shrink-0 bg-slate-950"
                        />
                      )}
                      <span className="font-mono font-bold text-sky-300 shrink-0">
                        #{f.frame_id}
                      </span>
                      <span className="text-slate-200 truncate flex-1" dir="auto">
                        {f.text_preview || f.labels.join(', ') || f.keep_reason}
                      </span>
                    </div>
                    <span className="rounded-md bg-slate-800/90 border border-slate-700 px-2 py-0.5 text-[10px] font-mono text-slate-300 shrink-0">
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
          className="rounded-2xl border border-slate-800/90 bg-gradient-to-b from-slate-900/85 to-[#0A0F1D]/90 p-5 flex flex-col gap-4 shadow-xl shadow-black/40"
        >
          <div className="flex items-center justify-between border-b border-slate-800/80 pb-3.5">
            <div className="flex items-center gap-3">
              <span className="flex h-7 w-7 items-center justify-center rounded-lg bg-indigo-500/15 border border-indigo-400/30 text-xs font-bold text-indigo-300">
                3
              </span>
              <div>
                <h2 id="panel-qa-title" className="text-base font-bold text-white">
                  3. Q&amp;A History &amp; Grounded Evidence
                </h2>
                <p className="text-xs text-slate-400">
                  Questions, intents, answers, score parts, and step timings (
                  <code className="font-mono text-indigo-300/90">query_answered</code>)
                </p>
              </div>
            </div>
            <span className="rounded-full bg-indigo-500/15 border border-indigo-500/30 px-3 py-0.5 text-xs font-mono font-semibold text-indigo-300">
              {recentQuestions.length} live
            </span>
          </div>

          <div className="flex-1 rounded-xl border border-slate-800/80 bg-[#060913]/70 p-4 flex flex-col justify-center">
            {recentQuestions.length === 0 ? (
              <p className="text-xs text-slate-400 text-center leading-relaxed" dir="auto">
                Q&amp;A panel placeholder — spoken and typed questions (EN / العربية) with
                intent badges, confidence, and evidence score parts will appear here in real time.
              </p>
            ) : (
              <ul className="space-y-3 max-h-64 overflow-y-auto pr-1">
                {recentQuestions.map((q) => (
                  <li
                    key={q.question_id}
                    className="rounded-xl border border-slate-800/90 bg-slate-900/90 hover:border-indigo-500/40 p-3.5 text-xs space-y-2 transition"
                  >
                    <div className="flex items-center justify-between gap-2 text-[11px] text-slate-400">
                      <span className="font-mono font-semibold text-indigo-300">
                        #{q.question_id} · {q.intent} · {q.lang.toUpperCase()}
                      </span>
                      <span className="rounded-full bg-slate-800 border border-slate-700 px-2 py-0.5 text-[10px] font-semibold uppercase tracking-wider text-slate-300">
                        {q.confidence}
                      </span>
                    </div>
                    <p className="font-semibold text-slate-100" dir="auto">
                      Q: {q.transcript}
                    </p>
                    <p
                      className="text-emerald-300 bg-emerald-950/30 border border-emerald-500/20 rounded-lg px-2.5 py-1.5 font-medium"
                      dir="auto"
                    >
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
          className="rounded-2xl border border-slate-800/90 bg-gradient-to-b from-slate-900/85 to-[#0A0F1D]/90 p-5 flex flex-col gap-4 shadow-xl shadow-black/40"
        >
          <div className="flex items-center justify-between border-b border-slate-800/80 pb-3.5">
            <div className="flex items-center gap-3">
              <span className="flex h-7 w-7 items-center justify-center rounded-lg bg-amber-500/15 border border-amber-400/30 text-xs font-bold text-amber-300">
                4
              </span>
              <div>
                <h2 id="panel-alerts-title" className="text-base font-bold text-white">
                  4. Proactive Obstacle &amp; Navigation Alerts
                </h2>
                <p className="text-xs text-slate-400">
                  Real-time warnings pushed over <code className="font-mono text-amber-300/90">/v1/ws/dashboard</code> (
                  <code className="font-mono text-amber-300/90">alert</code> event)
                </p>
              </div>
            </div>
            <span className="rounded-full bg-amber-500/15 border border-amber-500/30 px-3 py-0.5 text-xs font-mono font-semibold text-amber-300">
              {recentAlerts.length} alerts
            </span>
          </div>

          <div className="flex-1 rounded-xl border border-slate-800/80 bg-[#060913]/70 p-4 flex flex-col justify-center">
            {recentAlerts.length === 0 ? (
              <p className="text-xs text-slate-400 text-center leading-relaxed" dir="auto">
                Alerts panel placeholder — proactive hazard notifications (
                <code className="font-mono text-amber-300">door_ahead</code>,{' '}
                <code className="font-mono text-amber-300">stairs_ahead</code>,{' '}
                <code className="font-mono text-amber-300">obstacle_ahead</code>,{' '}
                <code className="font-mono text-amber-300">vehicle_near</code>) with thumbnails will display here.
              </p>
            ) : (
              <ul className="space-y-2.5 max-h-64 overflow-y-auto pr-1">
                {recentAlerts.map((a) => (
                  <li
                    key={a.alert_id}
                    className="rounded-xl border border-amber-500/35 bg-amber-950/30 px-3.5 py-2.5 text-xs flex items-center justify-between gap-2.5"
                  >
                    <span className="font-mono font-bold text-amber-300">{a.kind}</span>
                    <span className="text-slate-100 flex-1 font-medium" dir="auto">
                      {a.message}
                    </span>
                    <span className="rounded bg-amber-500/20 px-1.5 py-0.5 text-[10px] font-bold text-amber-200 uppercase">
                      {a.lang}
                    </span>
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
