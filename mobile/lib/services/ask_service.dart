import 'dart:async';
import 'dart:convert';
import 'dart:io';

import 'package:flutter/foundation.dart';
import 'package:http/http.dart' as http;
import 'package:record/record.dart';

import '../models/ask_models.dart';
import 'session_websocket_service.dart';

abstract class AudioRecorderAdapter {
  Future<bool> hasPermission();
  Future<void> start(RecordConfig config, {required String path});
  Future<Uint8List?> stopAndRead({required bool readBytes});
  Future<void> dispose();
}

class RecordAudioRecorderAdapter implements AudioRecorderAdapter {
  final AudioRecorder _recorder = AudioRecorder();
  String? _lastStartedPath;

  @override
  Future<bool> hasPermission() => _recorder.hasPermission();

  @override
  Future<void> start(RecordConfig config, {required String path}) async {
    _lastStartedPath = path;
    await _recorder.start(config, path: path);
  }

  @override
  Future<Uint8List?> stopAndRead({required bool readBytes}) async {
    String? recordedPath;
    try {
      recordedPath = await _recorder.stop();
    } catch (_) {
      recordedPath = _lastStartedPath;
    }
    final filePath = recordedPath ?? _lastStartedPath;
    _lastStartedPath = null;
    if (filePath == null) return null;

    final file = File(filePath);
    Uint8List? bytes;
    try {
      if (await file.exists()) {
        if (readBytes) {
          bytes = await file.readAsBytes();
        }
        await file.delete();
      }
    } catch (_) {
      // Best-effort cleanup so audio never lingers on disk (§6.5.1).
    }
    return bytes;
  }

  @override
  Future<void> dispose() => _recorder.dispose();
}

enum StopOutcomeKind { ignoredShortTap, uploaded, notRecording }

class StopRecordingOutcome {
  final StopOutcomeKind kind;
  final Duration duration;
  final AskResult? result;

  const StopRecordingOutcome({
    required this.kind,
    required this.duration,
    this.result,
  });
}

class AskService {
  static const Duration minDuration = Duration(milliseconds: 400);
  static const Duration maxDuration = Duration(seconds: 10);
  static const int sampleRate = 16000;
  static const int numChannels = 1;
  static const int bytesPerSample = 2;
  static const int maxPcmBytes =
      10 * sampleRate * numChannels * bytesPerSample; // 320,000 bytes

  final String baseUrl;
  final String Function()? baseUrlProvider;
  final AudioRecorderAdapter _recorder;
  final http.Client _httpClient;
  final DateTime Function() _now;

  DateTime? _recordingStartedAt;
  DateTime? _holdStartedAt;
  Duration? _lastHoldDuration;
  Timer? _maxDurationTimer;
  bool _isRecording = false;
  bool _isStopping = false;

  AskService({
    this.baseUrl = SessionWebSocketService.baseUrl,
    this.baseUrlProvider,
    AudioRecorderAdapter? recorder,
    http.Client? httpClient,
    DateTime Function()? now,
  }) : _recorder = recorder ?? RecordAudioRecorderAdapter(),
       _httpClient = httpClient ?? http.Client(),
       _now = now ?? DateTime.now;

  bool get isRecording => _isRecording;

  void markHoldStart() {
    _holdStartedAt = _now();
    _lastHoldDuration = null;
  }

  Duration? markHoldEnd() {
    if (_holdStartedAt != null) {
      _lastHoldDuration = _now().difference(_holdStartedAt!);
    }
    return _lastHoldDuration;
  }

  static String normalizeLang(String rawLang) {
    final lower = rawLang.trim().toLowerCase();
    if (lower.startsWith('ar')) return 'ar';
    if (lower.startsWith('fr')) return 'fr';
    return 'en';
  }

  static String fallbackSpeakForLang(String lang) {
    switch (normalizeLang(lang)) {
      case 'ar':
        return 'عذرًا، حدث خطأ ما. من فضلك حاول مرة أخرى.';
      case 'fr':
        return 'Désolé, une erreur s\'est produite. Veuillez réessayer.';
      default:
        return 'Sorry, something went wrong. Please try again.';
    }
  }

  /// Ensures a 16 kHz mono 16-bit PCM WAV never exceeds 10.0 seconds (320,000 data bytes).
  static Uint8List clampWavToMaxDuration(
    Uint8List wavBytes, {
    int maxDataBytes = maxPcmBytes,
  }) {
    if (wavBytes.length <= 44) return wavBytes;
    final view = ByteData.sublistView(wavBytes);
    // Standard RIFF/WAVE header check ('RIFF' at 0, 'WAVE' at 8)
    if (view.getUint32(0, Endian.big) != 0x52494646 ||
        view.getUint32(8, Endian.big) != 0x57415645) {
      return wavBytes;
    }

    int offset = 12;
    while (offset + 8 <= wavBytes.length) {
      final chunkId = view.getUint32(offset, Endian.big);
      final chunkSize = view.getUint32(offset + 4, Endian.little);
      final dataStart = offset + 8;
      // 'data' chunk (0x64617461)
      if (chunkId == 0x64617461) {
        final availableData = wavBytes.length - dataStart;
        final actualData = chunkSize < availableData ? chunkSize : availableData;
        if (actualData <= maxDataBytes) {
          return wavBytes;
        }
        final clampedData = maxDataBytes - (maxDataBytes % bytesPerSample);
        final totalLen = dataStart + clampedData;
        final out = Uint8List.fromList(wavBytes.sublist(0, totalLen));
        final outView = ByteData.sublistView(out);
        outView.setUint32(4, totalLen - 8, Endian.little);
        outView.setUint32(offset + 4, clampedData, Endian.little);
        return out;
      }
      offset = dataStart + chunkSize + (chunkSize.isOdd ? 1 : 0);
    }
    return wavBytes;
  }

  static Future<String> _resolveTempWavPath(DateTime timestamp) async {
    final fileName = 'va_ask_${timestamp.millisecondsSinceEpoch}.wav';
    if (Platform.isAndroid) {
      for (final candidate in const [
        '/data/user/0/edu.visionassistant.mobile/cache',
        '/data/data/edu.visionassistant.mobile/cache',
      ]) {
        final dir = Directory(candidate);
        if (await dir.exists()) {
          return '$candidate/$fileName';
        }
      }
    }
    return '${Directory.systemTemp.path}${Platform.pathSeparator}$fileName';
  }

  Future<bool> startRecording({
    required void Function() onMaxDurationReached,
  }) async {
    if (_isRecording || _isStopping) return false;

    try {
      final permitted = await _recorder.hasPermission();
      if (!permitted) {
        if (kDebugMode) {
          debugPrint('ASK: Microphone permission not granted.');
        }
        return false;
      }

      final now = _now();
      _recordingStartedAt = now;
      final tempPath = await _resolveTempWavPath(now);

      const config = RecordConfig(
        encoder: AudioEncoder.wav,
        sampleRate: sampleRate,
        numChannels: numChannels,
        bitRate: sampleRate * numChannels * bytesPerSample * 8,
      );

      await _recorder.start(config, path: tempPath);
      _isRecording = true;

      if (kDebugMode) {
        debugPrint('ASK: Started recording WAV (16kHz mono) -> $tempPath');
      }

      _maxDurationTimer?.cancel();
      _maxDurationTimer = Timer(maxDuration, () {
        if (_isRecording && !_isStopping) {
          onMaxDurationReached();
        }
      });

      return true;
    } catch (e) {
      if (kDebugMode) {
        debugPrint('ASK: Failed to start recording: $e');
      }
      _isRecording = false;
      return false;
    }
  }

  Future<StopRecordingOutcome> stopAndAsk({
    required String? sessionId,
    required String lang,
    Duration? holdDuration,
  }) async {
    if (!_isRecording || _isStopping) {
      return const StopRecordingOutcome(
        kind: StopOutcomeKind.notRecording,
        duration: Duration.zero,
      );
    }

    _isStopping = true;
    _maxDurationTimer?.cancel();
    _maxDurationTimer = null;

    final startedAt = _holdStartedAt ?? _recordingStartedAt ?? _now();
    final elapsed = holdDuration ?? _lastHoldDuration ?? _now().difference(startedAt);
    final shouldUpload = elapsed >= minDuration;

    _isRecording = false;
    _recordingStartedAt = null;
    _holdStartedAt = null;
    _lastHoldDuration = null;

    Uint8List? audioBytes;
    try {
      audioBytes = await _recorder.stopAndRead(readBytes: shouldUpload);
    } finally {
      _isStopping = false;
    }

    if (!shouldUpload) {
      if (kDebugMode) {
        debugPrint(
          'ASK: Ignored short tap (${elapsed.inMilliseconds} ms < 400 ms)',
        );
      }
      return StopRecordingOutcome(
        kind: StopOutcomeKind.ignoredShortTap,
        duration: elapsed,
      );
    }

    final normalizedLang = normalizeLang(lang);
    if (sessionId == null || sessionId.isEmpty) {
      return StopRecordingOutcome(
        kind: StopOutcomeKind.uploaded,
        duration: elapsed,
        result: AskResult.failure(
          AskError(
            code: 'BAD_REQUEST',
            message: 'No active session.',
            speak: fallbackSpeakForLang(normalizedLang),
          ),
        ),
      );
    }

    if (audioBytes == null || audioBytes.isEmpty) {
      return StopRecordingOutcome(
        kind: StopOutcomeKind.uploaded,
        duration: elapsed,
        result: AskResult.failure(
          AskError(
            code: 'BAD_AUDIO',
            message: 'Recorded audio file was empty.',
            speak: fallbackSpeakForLang(normalizedLang),
          ),
        ),
      );
    }

    final clampedAudio = clampWavToMaxDuration(audioBytes);
    final result = await uploadAudio(
      audioBytes: clampedAudio,
      sessionId: sessionId,
      lang: normalizedLang,
    );

    return StopRecordingOutcome(
      kind: StopOutcomeKind.uploaded,
      duration: elapsed,
      result: result,
    );
  }

  Future<AskResult> uploadAudio({
    required Uint8List audioBytes,
    required String sessionId,
    required String lang,
  }) async {
    final normalizedLang = normalizeLang(lang);
    final effectiveBaseUrl = baseUrlProvider != null
        ? baseUrlProvider!()
        : baseUrl;
    final uri = Uri.parse('$effectiveBaseUrl/v1/ask');

    try {
      final request = http.MultipartRequest('POST', uri)
        ..fields['session_id'] = sessionId
        ..fields['lang'] = normalizedLang
        ..files.add(
          http.MultipartFile.fromBytes(
            'audio',
            audioBytes,
            filename: 'question.wav',
          ),
        );

      final streamedResponse = await _httpClient
          .send(request)
          .timeout(const Duration(seconds: 45));
      final response = await http.Response.fromStream(streamedResponse);

      final decoded = jsonDecode(utf8.decode(response.bodyBytes));
      if (kDebugMode) {
        debugPrint('ASK RESPONSE (${response.statusCode}): $decoded');
      }
      if (response.statusCode == 200 && decoded is Map<String, dynamic>) {
        return AskResult.success(AskResponse.fromJson(decoded));
      }

      if (decoded is Map<String, dynamic>) {
        return AskResult.failure(AskError.fromJson(decoded));
      }

      return AskResult.failure(
        AskError(
          code: 'INTERNAL',
          message: 'Unexpected server response (${response.statusCode}).',
          speak: fallbackSpeakForLang(normalizedLang),
        ),
      );
    } catch (e) {
      return AskResult.failure(
        AskError(
          code: 'INTERNAL',
          message: e.toString(),
          speak: fallbackSpeakForLang(normalizedLang),
        ),
      );
    }
  }

  Future<void> dispose() async {
    _maxDurationTimer?.cancel();
    await _recorder.dispose();
    _httpClient.close();
  }
}
