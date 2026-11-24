/**
 * Exact TypeScript definitions matching contracts/openapi.yaml and contracts/events.schema.json (§8).
 */

export type Lang = 'en' | 'ar' | 'fr';

export type Intent =
  | 'LIVE_SCENE'
  | 'READ_TEXT_NOW'
  | 'READ_TEXT_PAST'
  | 'MEMORY'
  | 'FIND_OBJECT'
  | 'IDENTITY'
  | 'OUT_OF_SCOPE';

export type Confidence = 'high' | 'medium' | 'low' | 'none';

export type KeepReason = 'novel' | 'new_objects' | 'text' | 'heartbeat';

export type ErrorCode =
  | 'STT_FAILED'
  | 'GEMINI_FAILED'
  | 'NO_STREAM'
  | 'BAD_AUDIO'
  | 'BAD_REQUEST'
  | 'NOT_FOUND'
  | 'INTERNAL';

export type AlertKind = 'door_ahead' | 'stairs_ahead' | 'obstacle_ahead' | 'vehicle_near';

export type MediaKind = 'frames' | 'thumbs' | 'originals';

export type Position = 'left' | 'center' | 'right';

export type Bbox = [number, number, number, number];

export type TimingsMs = Record<string, number>;

export interface ErrorDetail {
  code: ErrorCode;
  message: string;
  speak: string;
}

export interface ApiErrorEnvelope {
  error: ErrorDetail;
}

export interface SessionCreate {
  app_lang: Lang;
  consent_at: string;
  consent_ver: string;
}

export interface SessionCreated {
  session_id: string;
  whip_url: string;
  ws_url: string;
}

export interface AskTextRequest {
  session_id: string;
  text: string;
  lang: Lang;
}

export interface AskEvidence {
  frame_id: number | null;
  seconds_ago: number;
  thumb_url: string | null;
  score: number;
}

export interface AskResponse {
  question_id: number;
  transcript: string;
  lang: Lang;
  intent: Intent;
  answer: string;
  confidence: Confidence;
  abstained: boolean;
  evidence: AskEvidence[];
  timings_ms: TimingsMs;
}

export interface FrameSummary {
  frame_id: number;
  captured_at: string;
  thumb_url: string;
  labels: string[];
  text_preview: string;
  keep_reason: KeepReason;
  score: number | null;
}

export interface FrameDetection {
  label: string;
  confidence: number;
  bbox: Bbox;
  area_ratio: number;
  position: Position;
}

export interface FrameOcrItem {
  text: string;
  confidence: number;
  bbox: Bbox;
  from_crop: boolean;
}

export interface FrameDetail {
  frame_id: number;
  session_id: string;
  captured_at: string;
  image_url: string;
  thumb_url: string;
  original_url: string | null;
  width: number;
  height: number;
  sharpness: number;
  brightness: number;
  contrast: number;
  edge_density: number;
  enhanced: boolean;
  enhance_ops: string[];
  keep_reason: KeepReason;
  faces_blurred: number;
  scene_text: string;
  timings_ms: TimingsMs;
  detections: FrameDetection[];
  ocr: FrameOcrItem[];
}

export interface Alert {
  alert_id: number;
  kind: AlertKind;
  message: string;
  lang: Lang;
  frame_id: number | null;
  thumb_url: string | null;
  created_at: string;
}

export interface Status {
  session_id: string | null;
  stream_connected: boolean;
  input_fps: number;
  analysed_fps: number;
  saved_per_min: number;
  frames_in_db: number;
  retention_min: number;
  gpu_mem_mb: number | null;
  last_answer_at: string | null;
  groq_ok: boolean;
  gemini_ok: boolean;
}

export interface ScoreParts {
  clip: number;
  text: number;
  obj: number;
  rec: number;
  q: number;
}

export interface QuestionEvidence {
  frame_id: number;
  rank: number;
  score: number;
  parts: ScoreParts;
  thumb_url: string;
}

export interface QuestionRecord {
  question_id: number;
  session_id: string;
  asked_at: string;
  input_mode: 'voice' | 'text';
  stt_engine: 'groq' | 'local' | null;
  transcript: string;
  lang: Lang;
  intent: Intent;
  t_from: string | null;
  t_to: string | null;
  answer: string;
  confidence: Confidence;
  abstained: boolean;
  safety_rewrite: boolean;
  model: string | null;
  timings_ms: TimingsMs;
  evidence: QuestionEvidence[];
}

export interface ListFramesParams {
  session_id: string;
  from?: string;
  to?: string;
  label?: string;
  text?: string;
  q?: string;
  limit?: number;
}

export interface ListQuestionsParams {
  session_id: string;
  limit?: number;
}

export interface ListAlertsParams {
  session_id: string;
  limit?: number;
}

/**
 * WebSocket messages (contracts/events.schema.json)
 */
export interface HelloMessage {
  type: 'hello';
  session_id: string;
  app_lang: Lang;
}

export interface AlertMessage extends Alert {
  type: 'alert';
}

export interface StatusMessage extends Status {
  type: 'status';
}

export interface FrameStoredMessage {
  type: 'frame_stored';
  frame_id: number;
  captured_at: string;
  thumb_url: string;
  labels: string[];
  text_preview: string;
  keep_reason: KeepReason;
}

export interface QueryAnsweredMessage extends QuestionRecord {
  type: 'query_answered';
}

export interface PingMessage {
  type: 'ping';
}

export interface PongMessage {
  type: 'pong';
}

export type DashboardWsMessage =
  | HelloMessage
  | AlertMessage
  | StatusMessage
  | FrameStoredMessage
  | QueryAnsweredMessage
  | PingMessage
  | PongMessage;

