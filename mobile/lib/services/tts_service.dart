import 'package:flutter/foundation.dart';
import 'package:flutter_tts/flutter_tts.dart';

import '../models/settings_models.dart';

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
    required this.settingsProvider,
  }) : _engine = engine ?? FlutterTtsAdapter() {
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

