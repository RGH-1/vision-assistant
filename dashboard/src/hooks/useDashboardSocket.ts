import { useEffect, useRef, useState } from 'react';
import type {
  AlertMessage,
  DashboardWsMessage,
  FrameStoredMessage,
  QueryAnsweredMessage,
  Status,
  StatusMessage,
} from '../types/contracts';

export type WsConnectionState = 'connecting' | 'connected' | 'reconnecting' | 'disconnected';

/** Reconnect backoff schedule in seconds per contracts/events.schema.json & §8.2: 1, 2, 4, 8 s */
export const RECONNECT_DELAYS_MS = [1000, 2000, 4000, 8000] as const;

export interface UseDashboardSocketOptions {
  wsUrl: string;
  enabled?: boolean;
  onEvent?: (event: DashboardWsMessage) => void;
  onStatus?: (status: StatusMessage) => void;
  onFrameStored?: (frame: FrameStoredMessage) => void;
  onQueryAnswered?: (question: QueryAnsweredMessage) => void;
  onAlert?: (alert: AlertMessage) => void;
}

export interface UseDashboardSocketResult {
  connectionState: WsConnectionState;
  isConnected: boolean;
  reconnectAttempt: number;
  lastStatus: Status | null;
  lastEvent: DashboardWsMessage | null;
  recentFrames: FrameStoredMessage[];
  recentQuestions: QueryAnsweredMessage[];
  recentAlerts: AlertMessage[];
  setInitialStatus: (status: Status) => void;
}

function isDashboardWsMessage(value: unknown): value is DashboardWsMessage {
  return (
    typeof value === 'object' &&
    value !== null &&
    'type' in value &&
    typeof (value as { type: unknown }).type === 'string'
  );
}

export function useDashboardSocket({
  wsUrl,
  enabled = true,
  onEvent,
  onStatus,
  onFrameStored,
  onQueryAnswered,
  onAlert,
}: UseDashboardSocketOptions): UseDashboardSocketResult {
  const [connectionState, setConnectionState] = useState<WsConnectionState>(
    enabled ? 'connecting' : 'disconnected',
  );
  const [reconnectAttempt, setReconnectAttempt] = useState<number>(0);
  const [lastStatus, setLastStatus] = useState<Status | null>(null);
  const [lastEvent, setLastEvent] = useState<DashboardWsMessage | null>(null);
  const [recentFrames, setRecentFrames] = useState<FrameStoredMessage[]>([]);
  const [recentQuestions, setRecentQuestions] = useState<QueryAnsweredMessage[]>([]);
  const [recentAlerts, setRecentAlerts] = useState<AlertMessage[]>([]);

  const callbacksRef = useRef({
    onEvent,
    onStatus,
    onFrameStored,
    onQueryAnswered,
    onAlert,
  });
  callbacksRef.current = {
    onEvent,
    onStatus,
    onFrameStored,
    onQueryAnswered,
    onAlert,
  };

  useEffect(() => {
    if (!enabled) {
      setConnectionState('disconnected');
      return;
    }

    let unmounted = false;
    let socket: WebSocket | null = null;
    let reconnectTimer: ReturnType<typeof setTimeout> | null = null;
    let attemptCount = 0;

    const clearReconnectTimer = () => {
      if (reconnectTimer !== null) {
        clearTimeout(reconnectTimer);
        reconnectTimer = null;
      }
    };

    const scheduleReconnect = () => {
      if (unmounted) return;
      clearReconnectTimer();
      const delayIndex = Math.min(attemptCount, RECONNECT_DELAYS_MS.length - 1);
      const delayMs = RECONNECT_DELAYS_MS[delayIndex] ?? 8000;
      attemptCount += 1;
      setReconnectAttempt(attemptCount);
      setConnectionState('reconnecting');

      reconnectTimer = setTimeout(() => {
        if (!unmounted) {
          connect();
        }
      }, delayMs);
    };

    const connect = () => {
      if (unmounted) return;
      clearReconnectTimer();
      setConnectionState(attemptCount === 0 ? 'connecting' : 'reconnecting');

      try {
        socket = new WebSocket(wsUrl);
      } catch {
        scheduleReconnect();
        return;
      }

      socket.onopen = () => {
        if (unmounted) return;
        attemptCount = 0;
        setReconnectAttempt(0);
        setConnectionState('connected');
      };

      socket.onmessage = (msgEvent: MessageEvent) => {
        if (unmounted || typeof msgEvent.data !== 'string') return;
        let parsed: unknown;
        try {
          parsed = JSON.parse(msgEvent.data);
        } catch {
          return;
        }
        if (!isDashboardWsMessage(parsed)) {
          return;
        }

        // Reply pong to server ping immediately (§8.2)
        if (parsed.type === 'ping') {
          if (socket && socket.readyState === WebSocket.OPEN) {
            socket.send(JSON.stringify({ type: 'pong' }));
          }
          setLastEvent(parsed);
          callbacksRef.current.onEvent?.(parsed);
          return;
        }

        setLastEvent(parsed);
        callbacksRef.current.onEvent?.(parsed);

        switch (parsed.type) {
          case 'status': {
            const { type: _type, ...statusFields } = parsed;
            void _type;
            setLastStatus(statusFields);
            callbacksRef.current.onStatus?.(parsed);
            break;
          }
          case 'frame_stored': {
            setRecentFrames((prev) => [parsed, ...prev].slice(0, 50));
            callbacksRef.current.onFrameStored?.(parsed);
            break;
          }
          case 'query_answered': {
            setRecentQuestions((prev) => [parsed, ...prev].slice(0, 50));
            callbacksRef.current.onQueryAnswered?.(parsed);
            break;
          }
          case 'alert': {
            setRecentAlerts((prev) => [parsed, ...prev].slice(0, 50));
            callbacksRef.current.onAlert?.(parsed);
            break;
          }
          default:
            break;
        }
      };

      socket.onerror = () => {
        // Browser fires onclose immediately after onerror, which schedules reconnect.
      };

      socket.onclose = () => {
        if (unmounted) return;
        scheduleReconnect();
      };
    };

    connect();

    return () => {
      unmounted = true;
      clearReconnectTimer();
      if (socket) {
        socket.onopen = null;
        socket.onmessage = null;
        socket.onerror = null;
        socket.onclose = null;
        socket.close();
      }
    };
  }, [wsUrl, enabled]);

  return {
    connectionState,
    isConnected: connectionState === 'connected',
    reconnectAttempt,
    lastStatus,
    lastEvent,
    recentFrames,
    recentQuestions,
    recentAlerts,
    setInitialStatus: setLastStatus,
  };
}

