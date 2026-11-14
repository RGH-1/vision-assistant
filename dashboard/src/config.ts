/**
 * Resolves the backend REST, WebSocket, and MediaMTX stream URLs.
 * Priority:
 * 1. URL query parameter (?server=http://... or ?serverUrl=http://...)
 * 2. Vite environment variable (VITE_SERVER_URL)
 * 3. Current origin if served directly from FastAPI (:8000), otherwise http://localhost:8000
 */

export interface DashboardConfig {
  serverUrl: string;
  wsDashboardUrl: string;
  whepUrl: string;
  streamIframeUrl: string;
}

export function normalizeBaseUrl(rawUrl: string): string {
  return rawUrl.trim().replace(/\/+$/, '');
}

export function toWebSocketUrl(httpBaseUrl: string, path: string): string {
  const normalized = normalizeBaseUrl(httpBaseUrl);
  const wsBase = normalized.replace(/^http(s?):\/\//i, 'ws$1://');
  const cleanPath = path.startsWith('/') ? path : `/${path}`;
  return `${wsBase}${cleanPath}`;
}

export function resolveDashboardConfig(
  search: string = typeof window !== 'undefined' ? window.location.search : '',
  envServerUrl: string | undefined = import.meta.env.VITE_SERVER_URL,
  envWhepUrl: string | undefined = import.meta.env.VITE_WHEP_URL,
): DashboardConfig {
  const params = new URLSearchParams(search);
  const queryServer = params.get('server') ?? params.get('serverUrl');
  const queryWhep = params.get('whep') ?? params.get('whepUrl');

  let defaultServer = 'http://localhost:8000';
  if (
    typeof window !== 'undefined' &&
    window.location.port === '8000' &&
    window.location.origin.startsWith('http')
  ) {
    defaultServer = window.location.origin;
  }

  const serverUrl = normalizeBaseUrl(
    (queryServer && queryServer.trim().length > 0 ? queryServer : undefined) ??
      (envServerUrl && envServerUrl.trim().length > 0 ? envServerUrl : undefined) ??
      defaultServer,
  );

  let hostForStream = 'localhost';
  try {
    hostForStream = new URL(serverUrl).hostname || 'localhost';
  } catch {
    hostForStream = 'localhost';
  }

  const defaultWhepUrl = `http://${hostForStream}:8889/mystream/whep`;
  const whepUrl = normalizeBaseUrl(
    (queryWhep && queryWhep.trim().length > 0 ? queryWhep : undefined) ??
      (envWhepUrl && envWhepUrl.trim().length > 0 ? envWhepUrl : undefined) ??
      defaultWhepUrl,
  );

  const streamIframeUrl = whepUrl.replace(/\/whep$/i, '');

  return {
    serverUrl,
    wsDashboardUrl: toWebSocketUrl(serverUrl, '/v1/ws/dashboard'),
    whepUrl,
    streamIframeUrl,
  };
}

