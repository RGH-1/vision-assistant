import 'dart:async';
import 'dart:convert';

import 'package:audioplayers/audioplayers.dart';
import 'package:flutter/foundation.dart';
import 'package:flutter_tts/flutter_tts.dart';
import 'package:http/http.dart' as http;

import '../models/settings_models.dart';
import 'session_websocket_service.dart';

abstract class TtsEngineAdapter {
  Future<dynamic> setLanguage(String language);
  Future<dynamic> setSpeechRate(double rate);
  Future<dynamic> speak(String text);
  Future<dynamic> stop();
  void setCompletionHandler(VoidCallback callback);
  void setCancelHandler(VoidCallback callback);
  void setErrorHandler(void Function(dynamic message) callback);
}

class FlutterTtsAdapter implements TtsEngineAdapter {
  final FlutterTts _tts = FlutterTts();
  bool _triedGoogleEngine = false;

  @override
  Future<dynamic> setLanguage(String language) async {
    var result = await _tts.setLanguage(language);
    if (result == 0 &&
        !kIsWeb &&
        defaultTargetPlatform == TargetPlatform.android &&
        !_triedGoogleEngine) {
      _triedGoogleEngine = true;
      try {
        final engines = await _tts.getEngines;
        if (engines is List && engines.contains('com.google.android.tts')) {
          if (kDebugMode) {
            debugPrint(
              'TTS: Default engine missing "$language", switching to com.google.android.tts',
            );
          }
          await _tts.setEngine('com.google.android.tts');
          result = await _tts.setLanguage(language);
        }
      } catch (e) {
        if (kDebugMode) {
          debugPrint('TTS: Failed switching engine: $e');
        }
      }
    }
    return result;
  }

  @override
  Future<dynamic> setSpeechRate(double rate) => _tts.setSpeechRate(rate);

  @override
  Future<dynamic> speak(String text) => _tts.speak(text);

  @override
  Future<dynamic> stop() => _tts.stop();

  @override
  void setCompletionHandler(VoidCallback callback) {
    _tts.setCompletionHandler(callback);
  }

  @override
  void setCancelHandler(VoidCallback callback) {
    _tts.setCancelHandler(callback);
  }

  @override
  void setErrorHandler(void Function(dynamic message) callback) {
    _tts.setErrorHandler(callback);
  }
}

/// ElevenLabs TTS adapter with automatic fallback to [FlutterTtsAdapter].
///
/// Works seamlessly with plain `flutter run` by calling the backend's `/v1/tts`
/// proxy (which reads `ELEVENLABS_API_KEY` from `.env`), or directly if
/// `--dart-define=ELEVENLABS_API_KEY=...` is passed.
class ElevenLabsTtsAdapter implements TtsEngineAdapter {
  static const String _envApiKey = String.fromEnvironment('ELEVENLABS_API_KEY');
  static const String _envVoiceId = String.fromEnvironment(
    'ELEVENLABS_VOICE_ID',
    defaultValue: 'EXAVITQu4vr4xnSDxMaL', // Sarah (free-tier default natural multilingual voice)
  );
  static const String _envModelId = String.fromEnvironment(
    'ELEVENLABS_MODEL_ID',
    defaultValue: 'eleven_flash_v2_5', // Low-latency multilingual (EN, AR, FR)
  );

  final String apiKey;
  final String voiceId;
  final String modelId;
  final String Function()? baseUrlProvider;
  final http.Client _httpClient;
  final TtsEngineAdapter _fallback;
  AudioPlayer? _audioPlayer;
  StreamSubscription<void>? _playerCompleteSub;

  String _languageCode = 'en';
  double _speechRate = 0.5;
  int _activeRequestId = 0;
  VoidCallback? _completionHandler;
  VoidCallback? _cancelHandler;

  ElevenLabsTtsAdapter({
    String? apiKey,
    String? voiceId,
    String? modelId,
    this.baseUrlProvider,
    http.Client? httpClient,
    TtsEngineAdapter? fallback,
    AudioPlayer? audioPlayer,
  }) : apiKey = (apiKey ?? _envApiKey).trim(),
       voiceId = (voiceId ?? _envVoiceId).trim(),
       modelId = (modelId ?? _envModelId).trim(),
       _httpClient = httpClient ?? http.Client(),
       _fallback = fallback ?? FlutterTtsAdapter() {
    _audioPlayer = audioPlayer;
  }

  AudioPlayer _ensureAudioPlayer() {
    var player = _audioPlayer;
    if (player == null) {
      player = AudioPlayer();
      _audioPlayer = player;
      _playerCompleteSub = player.onPlayerComplete.listen((_) {
        _completionHandler?.call();
      });
    }
    return player;
  }

  static String _toIsoLang(String locale) {
    final lower = locale.trim().toLowerCase();
    if (lower.startsWith('ar')) return 'ar';
    if (lower.startsWith('fr')) return 'fr';
    return 'en';
  }

  /// Maps Flutter's [0.2 .. 1.0] speechRate slider (default 0.5 -> 1.0x)
  /// to ElevenLabs' allowed [0.7 .. 1.2] voice speed range.
  static double mapRateToElevenLabsSpeed(double flutterRate) {
    final scaled = 0.7 + ((flutterRate.clamp(0.2, 1.0) - 0.2) / 0.8) * 0.5;
    return double.parse(scaled.clamp(0.7, 1.2).toStringAsFixed(2));
  }

  @override
  Future<dynamic> setLanguage(String language) async {
    _languageCode = _toIsoLang(language);
    return _fallback.setLanguage(language);
  }

  @override
  Future<dynamic> setSpeechRate(double rate) async {
    _speechRate = rate;
    return _fallback.setSpeechRate(rate);
  }

  @override
  Future<dynamic> speak(String text) async {
    final int requestId = ++_activeRequestId;
    final double speed = mapRateToElevenLabsSpeed(_speechRate);

    try {
      final http.Response response;
      if (apiKey.isNotEmpty) {
        final uri = Uri.parse(
          'https://api.elevenlabs.io/v1/text-to-speech/$voiceId?output_format=mp3_44100_128',
        );
        response = await _httpClient
            .post(
              uri,
              headers: {
                'xi-api-key': apiKey,
                'Content-Type': 'application/json',
                'Accept': 'audio/mpeg',
              },
              body: jsonEncode({
                'text': text,
                'model_id': modelId,
                'language_code': _languageCode,
                'voice_settings': {
                  'stability': 0.5,
                  'similarity_boost': 0.75,
                  'speed': speed,
                },
              }),
            )
            .timeout(const Duration(seconds: 8));
      } else {
        final baseUrl = baseUrlProvider != null
            ? baseUrlProvider!()
            : SessionWebSocketService.baseUrl;
        final uri = Uri.parse('$baseUrl/v1/tts');
        response = await _httpClient
            .post(
              uri,
              headers: {
                'Content-Type': 'application/json',
                'Accept': 'audio/mpeg',
              },
              body: jsonEncode({
                'text': text,
                'lang': _languageCode,
                'speed': speed,
              }),
            )
            .timeout(const Duration(seconds: 8));
      }

      if (requestId != _activeRequestId) {
        return 1;
      }

      if (response.statusCode == 200 && response.bodyBytes.isNotEmpty) {
        final player = _ensureAudioPlayer();
        await player.stop();
        if (requestId != _activeRequestId) {
          return 1;
        }
        await player.play(BytesSource(response.bodyBytes));
        return 1;
      }

      if (kDebugMode) {
        debugPrint(
          'TTS: ElevenLabs returned HTTP ${response.statusCode}, falling back to flutter_tts.',
        );
      }
    } catch (e) {
      if (kDebugMode) {
        debugPrint('TTS: ElevenLabs failed ($e), falling back to flutter_tts.');
      }
    }

    if (requestId != _activeRequestId) {
      return 1;
    }
    return _fallback.speak(text);
  }

  @override
  Future<dynamic> stop() async {
    ++_activeRequestId;
    try {
      await _audioPlayer?.stop();
    } catch (_) {}
    await _fallback.stop();
    _cancelHandler?.call();
    return 1;
  }

  @override
  void setCompletionHandler(VoidCallback callback) {
    _completionHandler = callback;
    _fallback.setCompletionHandler(callback);
  }

  @override
  void setCancelHandler(VoidCallback callback) {
    _cancelHandler = callback;
    _fallback.setCancelHandler(callback);
  }

  @override
  void setErrorHandler(void Function(dynamic message) callback) {
    _fallback.setErrorHandler(callback);
  }

  Future<void> dispose() async {
    await _playerCompleteSub?.cancel();
    await _audioPlayer?.dispose();
  }
}

class NoOpTtsEngineAdapter implements TtsEngineAdapter {
  @override
  Future<dynamic> setLanguage(String language) async => 1;

  @override
  Future<dynamic> setSpeechRate(double rate) async => 1;

  @override
  Future<dynamic> speak(String text) async => 1;

  @override
  Future<dynamic> stop() async => 1;

  @override
  void setCompletionHandler(VoidCallback callback) {}

  @override
  void setCancelHandler(VoidCallback callback) {}

  @override
  void setErrorHandler(void Function(dynamic message) callback) {}
}

class TtsService {
  final TtsEngineAdapter _engine;
  final SettingsModel Function() settingsProvider;

  bool _isSpeakingAnswer = false;
  bool _isSpeakingAlert = false;
  bool _suppressCancelCallback = false;
  int _speechToken = 0;

  TtsService({
    TtsEngineAdapter? engine,
    String Function()? baseUrlProvider,
    required this.settingsProvider,
  }) : _engine =
           engine ?? ElevenLabsTtsAdapter(baseUrlProvider: baseUrlProvider) {
    _engine.setCompletionHandler(_handleSpeechFinished);
    _engine.setCancelHandler(() {
      if (_suppressCancelCallback) return;
      _handleSpeechFinished();
    });
    _engine.setErrorHandler((dynamic msg) {
      if (kDebugMode) {
        debugPrint('TTS ERROR: $msg');
      }
      _handleSpeechFinished();
    });
  }

  bool get isSpeakingAnswer => _isSpeakingAnswer;
  bool get isSpeakingAlert => _isSpeakingAlert;
  bool get isSpeaking => _isSpeakingAnswer || _isSpeakingAlert;

  void _handleSpeechFinished() {
    _isSpeakingAnswer = false;
    _isSpeakingAlert = false;
  }

  /// Maps contract/settings language codes ('en', 'ar', 'fr') to exact flutter_tts locales:
  /// - 'en' -> 'en-US'
  /// - 'ar' -> 'ar'
  /// - 'fr' -> 'fr-FR'
  static String mapLangToTtsLocale(String rawLang) {
    final lower = rawLang.trim().toLowerCase();
    if (lower.startsWith('ar')) return 'ar';
    if (lower.startsWith('fr')) return 'fr-FR';
    return 'en-US';
  }

  /// Speaks an assistant answer. Answers always take priority over alerts:
  /// any ongoing alert or previous speech is immediately stopped.
  Future<void> speakAnswer(String text, {required String lang}) async {
    final trimmed = text.trim();
    if (trimmed.isEmpty) return;

    final int token = ++_speechToken;
    final locale = mapLangToTtsLocale(lang);
    final rate = settingsProvider().speechRate.clamp(0.1, 1.0);

    _isSpeakingAlert = false;
    _isSpeakingAnswer = true;

    try {
      _suppressCancelCallback = true;
      await _engine.stop();
    } catch (_) {
      // Ignore stop errors in headless/uninitialized environments.
    } finally {
      _suppressCancelCallback = false;
    }

    if (token != _speechToken) return;

    try {
      final langResult = await _engine.setLanguage(locale);
      if (kDebugMode) {
        debugPrint(
          'TTS [answer]: setLanguage($locale) -> $langResult, '
          'setSpeechRate($rate), text="$trimmed"',
        );
      }
      if (token != _speechToken) return;

      await _engine.setSpeechRate(rate);
      if (token != _speechToken) return;

      await _engine.speak(trimmed);
    } catch (e) {
      if (kDebugMode) {
        debugPrint('TTS [answer] exception: $e');
      }
      if (token == _speechToken) {
        _isSpeakingAnswer = false;
      }
    }
  }

  /// Speaks an incoming WebSocket alert only if alerts are enabled and no answer is currently being spoken.
  Future<bool> speakAlert(String text, {required String lang}) async {
    final trimmed = text.trim();
    if (trimmed.isEmpty) return false;

    final settings = settingsProvider();
    if (!settings.alertsEnabled) {
      if (kDebugMode) {
        debugPrint('TTS [alert]: skipped (alerts disabled in settings).');
      }
      return false;
    }

    // Priority rule: Answers take priority over alerts (§6.1.2).
    if (_isSpeakingAnswer) {
      if (kDebugMode) {
        debugPrint(
          'TTS [alert]: suppressed because an answer is currently speaking.',
        );
      }
      return false;
    }

    final int token = ++_speechToken;
    final locale = mapLangToTtsLocale(lang);
    final rate = settings.speechRate.clamp(0.1, 1.0);

    _isSpeakingAlert = true;

    try {
      _suppressCancelCallback = true;
      await _engine.stop();
    } catch (_) {
    } finally {
      _suppressCancelCallback = false;
    }

    if (token != _speechToken || _isSpeakingAnswer) {
      _isSpeakingAlert = false;
      return false;
    }

    try {
      final langResult = await _engine.setLanguage(locale);
      if (kDebugMode) {
        debugPrint(
          'TTS [alert]: setLanguage($locale) -> $langResult, '
          'setSpeechRate($rate), text="$trimmed"',
        );
      }
      if (token != _speechToken || _isSpeakingAnswer) {
        _isSpeakingAlert = false;
        return false;
      }

      await _engine.setSpeechRate(rate);
      if (token != _speechToken || _isSpeakingAnswer) {
        _isSpeakingAlert = false;
        return false;
      }

      await _engine.speak(trimmed);
      return true;
    } catch (e) {
      if (kDebugMode) {
        debugPrint('TTS [alert] exception: $e');
      }
      if (token == _speechToken) {
        _isSpeakingAlert = false;
      }
      return false;
    }
  }

  /// Barge-in: immediately stops any current speech (answer or alert).
  Future<void> stop() async {
    ++_speechToken;
    _isSpeakingAnswer = false;
    _isSpeakingAlert = false;
    try {
      _suppressCancelCallback = true;
      await _engine.stop();
      if (kDebugMode) {
        debugPrint('TTS: stopped (barge-in)');
      }
    } catch (_) {
      // Ignore TTS channel errors in headless environments.
    } finally {
      _suppressCancelCallback = false;
    }
  }
}

