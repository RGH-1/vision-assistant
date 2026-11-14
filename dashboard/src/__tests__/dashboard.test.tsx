import { act, render, screen } from '@testing-library/react';
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';
import App from '../App';
import { ApiError, VisionAssistantApiClient } from '../api/client';
import { resolveDashboardConfig } from '../config';
import { RECONNECT_DELAYS_MS } from '../hooks/useDashboardSocket';

class MockWebSocket {
  public static instances: MockWebSocket[] = [];
  public static readonly OPEN = 1;
  public static readonly CLOSED = 3;

  public readonly url: string;
  public readyState: number = 0;
  public sentMessages: string[] = [];
  public onopen: (() => void) | null = null;
  public onmessage: ((ev: { data: string }) => void) | null = null;
  public onerror: (() => void) | null = null;
  public onclose: (() => void) | null = null;

  constructor(url: string) {
    this.url = url;
    MockWebSocket.instances.push(this);
  }

  public send(data: string): void {
    this.sentMessages.push(data);
  }

  public close(): void {
    this.readyState = MockWebSocket.CLOSED;
  }

  public triggerOpen(): void {
    this.readyState = MockWebSocket.OPEN;
    this.onopen?.();
  }

  public triggerMessage(payload: unknown): void {
    this.onmessage?.({ data: JSON.stringify(payload) });
  }

  public triggerClose(): void {
    this.readyState = MockWebSocket.CLOSED;
    this.onclose?.();
  }
}

describe('Dashboard scaffold (VA-29)', () => {
  beforeEach(() => {
    MockWebSocket.instances = [];
    vi.stubGlobal('WebSocket', MockWebSocket);
    vi.stubGlobal(
      'fetch',
      vi.fn().mockResolvedValue({
        ok: true,
        json: async () => ({
          session_id: '3f2b8c1e-7a4d-4e6f-9b1a-2c5d8e7f6a01',
          stream_connected: true,
          input_fps: 15,
          analysed_fps: 5,
          saved_per_min: 12,
          frames_in_db: 48,
          retention_min: 30,
          gpu_mem_mb: 2100,
          last_answer_at: null,
          groq_ok: true,
          gemini_ok: true,
        }),
      }),
    );
  });

  afterEach(() => {
    vi.useRealTimers();
    vi.unstubAllGlobals();
  });

  it('resolves server URL from query override or VITE_SERVER_URL env', () => {
    const fromQuery = resolveDashboardConfig(
      '?server=http://192.168.8.107:8000/',
      'http://localhost:8000',
    );
    expect(fromQuery.serverUrl).toBe('http://192.168.8.107:8000');
    expect(fromQuery.wsDashboardUrl).toBe('ws://192.168.8.107:8000/v1/ws/dashboard');

    const fromEnv = resolveDashboardConfig('', 'http://10.0.0.5:8000');
    expect(fromEnv.serverUrl).toBe('http://10.0.0.5:8000');
    expect(fromEnv.wsDashboardUrl).toBe('ws://10.0.0.5:8000/v1/ws/dashboard');
  });

  it('renders 4 panels + controls and transitions badge to Connected when /v1/ws/dashboard opens', async () => {
    await act(async () => {
      render(<App />);
    });

    expect(screen.getByTestId('panel-live-status')).toBeInTheDocument();
    expect(screen.getByTestId('panel-timeline')).toBeInTheDocument();
    expect(screen.getByTestId('panel-qa')).toBeInTheDocument();
    expect(screen.getByTestId('panel-alerts')).toBeInTheDocument();

    const badge = screen.getByTestId('ws-connection-badge');
    expect(badge).toHaveTextContent(/Connecting/i);

    const ws = MockWebSocket.instances[0];
    expect(ws).toBeDefined();
    expect(ws?.url).toContain('/v1/ws/dashboard');

    act(() => {
      ws?.triggerOpen();
    });

    expect(badge).toHaveTextContent('Connected');

    // Verify ping -> pong reply and Arabic query_answered rendering with dir="auto"
    act(() => {
      ws?.triggerMessage({ type: 'ping' });
      ws?.triggerMessage({
        type: 'query_answered',
        question_id: 7,
        session_id: '3f2b8c1e-7a4d-4e6f-9b1a-2c5d8e7f6a01',
        asked_at: '2026-11-14T00:00:00Z',
        input_mode: 'voice',
        stt_engine: 'groq',
        transcript: 'ماذا يوجد أمامي؟',
        lang: 'ar',
        intent: 'LIVE_SCENE',
        t_from: null,
        t_to: null,
        answer: 'أمامك باب خشبي.',
        confidence: 'high',
        abstained: false,
        safety_rewrite: false,
        model: 'gemini-3.6-flash',
        timings_ms: { total: 950 },
        evidence: [],
      });
    });

    expect(ws?.sentMessages).toContain(JSON.stringify({ type: 'pong' }));
    const answerEl = screen.getByText(/أمامك باب خشبي/);
    expect(answerEl).toHaveAttribute('dir', 'auto');
  });

  it('reconnects using the 1, 2, 4, 8 s backoff schedule when WebSocket closes', async () => {
    vi.useFakeTimers();
    await act(async () => {
      render(<App />);
    });

    expect(MockWebSocket.instances).toHaveLength(1);
    act(() => {
      MockWebSocket.instances[0]?.triggerOpen();
    });
    expect(screen.getByTestId('ws-connection-badge')).toHaveTextContent('Connected');

    act(() => {
      MockWebSocket.instances[0]?.triggerClose();
    });
    expect(screen.getByTestId('ws-connection-badge')).toHaveTextContent(/Reconnecting/i);

    act(() => {
      vi.advanceTimersByTime(RECONNECT_DELAYS_MS[0]);
    });
    expect(MockWebSocket.instances).toHaveLength(2);
  });

  it('typed API client parses contract responses and Error envelopes accurately', async () => {
    const mockFetch = vi.fn().mockResolvedValueOnce({
      ok: false,
      status: 503,
      statusText: 'Service Unavailable',
      json: async () => ({
        error: {
          code: 'NO_STREAM',
          message: 'The camera stream is not connected.',
          speak: "I can't see right now. The camera is not connected.",
        },
      }),
    });

    const client = new VisionAssistantApiClient('http://localhost:8000', mockFetch);
    await expect(
      client.askText({
        session_id: '3f2b8c1e-7a4d-4e6f-9b1a-2c5d8e7f6a01',
        text: 'What is ahead?',
        lang: 'en',
      }),
    ).rejects.toThrowError(ApiError);
  });
});
