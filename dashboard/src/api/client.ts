import { normalizeBaseUrl } from '../config';
import type {
  Alert,
  ApiErrorEnvelope,
  AskResponse,
  AskTextRequest,
  ErrorCode,
  FrameDetail,
  FrameSummary,
  Lang,
  ListAlertsParams,
  ListFramesParams,
  ListQuestionsParams,
  MediaKind,
  QuestionRecord,
  SessionCreate,
  SessionCreated,
  Status,
} from '../types/contracts';

export class ApiError extends Error {
  public readonly status: number;
  public readonly code: ErrorCode;
  public readonly speak: string;

  constructor(status: number, code: ErrorCode, message: string, speak: string) {
    super(message);
    this.name = 'ApiError';
    this.status = status;
    this.code = code;
    this.speak = speak;
  }
}

function isApiErrorEnvelope(value: unknown): value is ApiErrorEnvelope {
  if (typeof value !== 'object' || value === null || !('error' in value)) {
    return false;
  }
  const err = (value as { error: unknown }).error;
  return (
    typeof err === 'object' &&
    err !== null &&
    'code' in err &&
    'message' in err &&
    'speak' in err
  );
}

export class VisionAssistantApiClient {
  public readonly baseUrl: string;
  private readonly fetchImpl: typeof fetch;

  constructor(
    baseUrl: string,
    fetchImpl: typeof fetch = (input, init) => globalThis.fetch(input, init),
  ) {
    this.baseUrl = normalizeBaseUrl(baseUrl);
    this.fetchImpl = fetchImpl;
  }

  public getMediaUrl(kind: MediaKind, file: string): string {
    const cleanFile = file.replace(/^\/+/, '');
    return `${this.baseUrl}/media/${kind}/${encodeURIComponent(cleanFile)}`;
  }

  public resolveThumbUrl(thumbPathOrUrl: string | null): string | null {
    if (!thumbPathOrUrl) {
      return null;
    }
    if (/^https?:\/\//i.test(thumbPathOrUrl)) {
      return thumbPathOrUrl;
    }
    const cleanPath = thumbPathOrUrl.startsWith('/') ? thumbPathOrUrl : `/${thumbPathOrUrl}`;
    return `${this.baseUrl}${cleanPath}`;
  }

  private async requestJson<T>(path: string, init?: RequestInit): Promise<T> {
    const response = await this.fetchImpl(`${this.baseUrl}${path}`, init);
    if (!response.ok) {
      await this.throwApiError(response);
    }
    return (await response.json()) as T;
  }

  private async requestNoContent(path: string, init?: RequestInit): Promise<void> {
    const response = await this.fetchImpl(`${this.baseUrl}${path}`, init);
    if (!response.ok) {
      await this.throwApiError(response);
    }
  }

  private async throwApiError(response: Response): Promise<never> {
    try {
      const data: unknown = await response.json();
      if (isApiErrorEnvelope(data)) {
        throw new ApiError(
          response.status,
          data.error.code,
          data.error.message,
          data.error.speak,
        );
      }
    } catch (err) {
      if (err instanceof ApiError) {
        throw err;
      }
    }
    throw new ApiError(
      response.status,
      'INTERNAL',
      `HTTP ${response.status}: ${response.statusText || 'Request failed'}`,
      'Sorry, something went wrong. Please try again.',
    );
  }

  /** POST /v1/sessions (operationId: createSession) */
  public async createSession(body: SessionCreate): Promise<SessionCreated> {
    return this.requestJson<SessionCreated>('/v1/sessions', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(body),
    });
  }

  /** POST /v1/sessions/{id}/end (operationId: endSession) */
  public async endSession(id: string): Promise<void> {
    return this.requestNoContent(`/v1/sessions/${encodeURIComponent(id)}/end`, {
      method: 'POST',
    });
  }

  /** DELETE /v1/sessions/{id} (operationId: deleteSession) */
  public async deleteSession(id: string): Promise<void> {
    return this.requestNoContent(`/v1/sessions/${encodeURIComponent(id)}`, {
      method: 'DELETE',
    });
  }

  /** POST /v1/ask (operationId: ask) */
  public async ask(params: {
    audio: Blob;
    session_id: string;
    lang: Lang;
    filename?: string;
  }): Promise<AskResponse> {
    const formData = new FormData();
    formData.append('audio', params.audio, params.filename ?? 'question.wav');
    formData.append('session_id', params.session_id);
    formData.append('lang', params.lang);

    return this.requestJson<AskResponse>('/v1/ask', {
      method: 'POST',
      body: formData,
    });
  }

  /** POST /v1/ask_text (operationId: askText) */
  public async askText(body: AskTextRequest): Promise<AskResponse> {
    return this.requestJson<AskResponse>('/v1/ask_text', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(body),
    });
  }

  /** GET /v1/status (operationId: getStatus) */
  public async getStatus(): Promise<Status> {
    return this.requestJson<Status>('/v1/status');
  }

  /** GET /v1/frames (operationId: listFrames) */
  public async listFrames(params: ListFramesParams): Promise<FrameSummary[]> {
    const query = new URLSearchParams();
    query.set('session_id', params.session_id);
    if (params.from) query.set('from', params.from);
    if (params.to) query.set('to', params.to);
    if (params.label) query.set('label', params.label);
    if (params.text) query.set('text', params.text);
    if (params.q) query.set('q', params.q);
    if (params.limit !== undefined) query.set('limit', String(params.limit));

    return this.requestJson<FrameSummary[]>(`/v1/frames?${query.toString()}`);
  }

  /** GET /v1/frames/{id} (operationId: getFrame) */
  public async getFrame(id: number): Promise<FrameDetail> {
    return this.requestJson<FrameDetail>(`/v1/frames/${encodeURIComponent(String(id))}`);
  }

  /** GET /v1/questions (operationId: listQuestions) */
  public async listQuestions(params: ListQuestionsParams): Promise<QuestionRecord[]> {
    const query = new URLSearchParams();
    query.set('session_id', params.session_id);
    if (params.limit !== undefined) query.set('limit', String(params.limit));

    return this.requestJson<QuestionRecord[]>(`/v1/questions?${query.toString()}`);
  }

  /** GET /v1/alerts (operationId: listAlerts) */
  public async listAlerts(params: ListAlertsParams): Promise<Alert[]> {
    const query = new URLSearchParams();
    query.set('session_id', params.session_id);
    if (params.limit !== undefined) query.set('limit', String(params.limit));

    return this.requestJson<Alert[]>(`/v1/alerts?${query.toString()}`);
  }
}

