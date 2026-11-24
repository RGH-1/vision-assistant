import 'dart:convert';
import 'dart:typed_data';

import 'package:flutter/material.dart';
import 'package:flutter_test/flutter_test.dart';
import 'package:http/http.dart' as http;
import 'package:http/testing.dart';
import 'package:mobile/main.dart';
import 'package:mobile/models/ask_models.dart';
import 'package:mobile/models/settings_models.dart';
import 'package:mobile/services/ask_service.dart';
import 'package:mobile/services/session_websocket_service.dart';
import 'package:mobile/services/tts_service.dart';
import 'package:record/record.dart';

Uint8List buildTestWav({int numSamples = 16000}) {
  final dataSize = numSamples * 2;
  final totalSize = 44 + dataSize;
  final bytes = Uint8List(totalSize);
  final view = ByteData.sublistView(bytes);

  // "RIFF"
  view.setUint32(0, 0x52494646, Endian.big);
  view.setUint32(4, totalSize - 8, Endian.little);
  // "WAVE"
  view.setUint32(8, 0x57415645, Endian.big);
  // "fmt "
  view.setUint32(12, 0x666d7420, Endian.big);
  view.setUint32(16, 16, Endian.little);
  view.setUint16(20, 1, Endian.little); // PCM
  view.setUint16(22, 1, Endian.little); // mono
  view.setUint32(24, 16000, Endian.little); // 16 kHz
  view.setUint32(28, 32000, Endian.little); // byte rate
  view.setUint16(32, 2, Endian.little); // block align
  view.setUint16(34, 16, Endian.little); // 16-bit
  // "data"
  view.setUint32(36, 0x64617461, Endian.big);
  view.setUint32(40, dataSize, Endian.little);
  return bytes;
}

class FakeAudioRecorder implements AudioRecorderAdapter {
  RecordConfig? lastConfig;
  String? lastPath;
  int startCalls = 0;
  int stopCalls = 0;
  Uint8List wavPayload = buildTestWav(numSamples: 16000);

  @override
  Future<bool> hasPermission() async => true;

  @override
  Future<void> start(RecordConfig config, {required String path}) async {
    startCalls++;
    lastConfig = config;
    lastPath = path;
  }

  @override
  Future<Uint8List?> stopAndRead({required bool readBytes}) async {
    stopCalls++;
    return readBytes ? wavPayload : null;
  }

  @override
  Future<void> dispose() async {}
}

class FakeTtsEngine implements TtsEngineAdapter {
  final List<String> languagesSet = [];
  final List<double> speechRatesSet = [];
  final List<String> spokenTexts = [];
  int stopCalls = 0;

  VoidCallback? _completionHandler;
  VoidCallback? _cancelHandler;

  @override
  Future<dynamic> setLanguage(String language) async {
    languagesSet.add(language);
    return 1;
  }

  @override
  Future<dynamic> setSpeechRate(double rate) async {
    speechRatesSet.add(rate);
    return 1;
  }

  @override
  Future<dynamic> speak(String text) async {
    spokenTexts.add(text);
    return 1;
  }

  @override
  Future<dynamic> stop() async {
    stopCalls++;
    _cancelHandler?.call();
    return 1;
  }

  void simulateCompletion() {
    _completionHandler?.call();
  }

  @override
  void setCompletionHandler(VoidCallback callback) {
    _completionHandler = callback;
  }

  @override
  void setCancelHandler(VoidCallback callback) {
    _cancelHandler = callback;
  }

  @override
  void setErrorHandler(void Function(dynamic message) callback) {}
}

void main() {
  testWidgets(
    'Short tap (< 0.4 s) is ignored and does not upload to /v1/ask',
    (WidgetTester tester) async {
      DateTime fakeNow = DateTime.utc(2026, 11, 11, 12, 0, 0);
      final fakeRecorder = FakeAudioRecorder();
      int uploadCalls = 0;

      final mockClient = MockClient((request) async {
        uploadCalls++;
        return http.Response('{}', 200);
      });

      final wsService = SessionWebSocketService()
        ..setSessionIdForTesting('3f2b8c1e-7a4d-4e6f-9b1a-2c5d8e7f6a01');
      final askService = AskService(
        recorder: fakeRecorder,
        httpClient: mockClient,
        now: () => fakeNow,
      );

      await tester.pumpWidget(
        MyApp(
          askService: askService,
          wsService: wsService,
          enablePlatformInit: false,
        ),
      );

      final gesture = await tester.startGesture(
        tester.getCenter(find.byKey(const Key('hold_to_talk_surface'))),
      );
      await tester.pump();
      expect(fakeRecorder.startCalls, 1);
      expect(fakeRecorder.lastConfig?.encoder, AudioEncoder.wav);
      expect(fakeRecorder.lastConfig?.sampleRate, 16000);
      expect(fakeRecorder.lastConfig?.numChannels, 1);

      // Advance only 250 ms (< 400 ms threshold)
      fakeNow = fakeNow.add(const Duration(milliseconds: 250));
      await gesture.up();
      await tester.pumpAndSettle();

      expect(fakeRecorder.stopCalls, 1);
      expect(uploadCalls, 0);
      expect(
        find.text('Hold anywhere on the screen to ask a question'),
        findsOneWidget,
      );
    },
  );

  testWidgets(
    'Hold >= 0.4 s uploads multipart POST /v1/ask and displays AskResponse',
    (WidgetTester tester) async {
      DateTime fakeNow = DateTime.utc(2026, 11, 11, 12, 0, 0);
      final fakeRecorder = FakeAudioRecorder();
      String? capturedSessionId;
      String? capturedLang;
      int? capturedAudioLength;

      final mockClient = MockClient.streaming((request, bodyStream) async {
        expect(request.method, 'POST');
        expect(request.url.path, '/v1/ask');
        final contentType = request.headers['content-type'] ?? '';
        expect(contentType, contains('multipart/form-data'));

        final bodyBytes = await bodyStream.toBytes();
        final bodyText = latin1.decode(bodyBytes);
        if (bodyText.contains('3f2b8c1e-7a4d-4e6f-9b1a-2c5d8e7f6a01')) {
          capturedSessionId = '3f2b8c1e-7a4d-4e6f-9b1a-2c5d8e7f6a01';
        }
        if (bodyText.contains('name="lang"\r\n\r\nen')) {
          capturedLang = 'en';
        }
        if (bodyText.contains('filename="question.wav"')) {
          capturedAudioLength = bodyBytes.length;
        }

        final payload = jsonEncode({
          'question_id': 42,
          'transcript': 'What is in front of me?',
          'lang': 'en',
          'intent': 'LIVE_SCENE',
          'answer': 'There is a wooden door directly ahead.',
          'confidence': 'high',
          'abstained': false,
          'evidence': [
            {
              'frame_id': null,
              'seconds_ago': 0.2,
              'thumb_url': null,
              'score': 1.0,
            },
          ],
          'timings_ms': {'stt': 310.0, 'gemini': 1200.0, 'total': 1510.0},
        });

        return http.StreamedResponse(
          Stream.value(utf8.encode(payload)),
          200,
          headers: {'content-type': 'application/json'},
        );
      });

      final wsService = SessionWebSocketService()
        ..setSessionIdForTesting('3f2b8c1e-7a4d-4e6f-9b1a-2c5d8e7f6a01');
      final askService = AskService(
        recorder: fakeRecorder,
        httpClient: mockClient,
        now: () => fakeNow,
      );

      await tester.pumpWidget(
        MyApp(
          askService: askService,
          wsService: wsService,
          enablePlatformInit: false,
        ),
      );

      final gesture = await tester.startGesture(
        tester.getCenter(find.byKey(const Key('hold_to_talk_surface'))),
      );
      await tester.pump();
      expect(find.text('Recording... Release to send'), findsOneWidget);

      // Hold for 1.2 s (>= 0.4 s)
      fakeNow = fakeNow.add(const Duration(milliseconds: 1200));
      await gesture.up();
      await tester.pumpAndSettle();

      expect(capturedSessionId, '3f2b8c1e-7a4d-4e6f-9b1a-2c5d8e7f6a01');
      expect(capturedLang, 'en');
      expect(capturedAudioLength, isNotNull);
      expect(find.text('“What is in front of me?”'), findsOneWidget);
      expect(
        find.text('There is a wooden door directly ahead.'),
        findsOneWidget,
      );
    },
  );

  testWidgets(
    'Hold auto-stops at 10 s and surfaces Error speak field on server error',
    (WidgetTester tester) async {
      DateTime fakeNow = DateTime.utc(2026, 11, 11, 12, 0, 0);
      final fakeRecorder = FakeAudioRecorder();
      int uploadCalls = 0;

      final mockClient = MockClient.streaming((request, bodyStream) async {
        uploadCalls++;
        await bodyStream.toBytes();
        final errorPayload = jsonEncode({
          'error': {
            'code': 'NO_STREAM',
            'message': 'The camera stream is not connected.',
            'speak': "I can't see right now. The camera is not connected.",
          },
        });
        return http.StreamedResponse(
          Stream.value(utf8.encode(errorPayload)),
          503,
          headers: {'content-type': 'application/json'},
        );
      });

      final wsService = SessionWebSocketService()
        ..setSessionIdForTesting('3f2b8c1e-7a4d-4e6f-9b1a-2c5d8e7f6a01');
      final askService = AskService(
        recorder: fakeRecorder,
        httpClient: mockClient,
        now: () => fakeNow,
      );

      await tester.pumpWidget(
        MyApp(
          askService: askService,
          wsService: wsService,
          enablePlatformInit: false,
        ),
      );

      final gesture = await tester.startGesture(
        tester.getCenter(find.byKey(const Key('hold_to_talk_surface'))),
      );
      await tester.pump();
      expect(fakeRecorder.startCalls, 1);

      // Advance clock and timer by 10 seconds without releasing finger
      fakeNow = fakeNow.add(const Duration(seconds: 10));
      await tester.pump(const Duration(seconds: 10));
      await tester.pumpAndSettle();

      expect(fakeRecorder.stopCalls, 1);
      expect(uploadCalls, 1);
      expect(
        find.text("I can't see right now. The camera is not connected."),
        findsOneWidget,
      );

      // Releasing finger afterwards is a no-op
      await gesture.up();
      await tester.pumpAndSettle();
      expect(uploadCalls, 1);
    },
  );

  test('clampWavToMaxDuration caps WAV data chunk at 10 s (320,000 bytes)', () {
    // 11 seconds = 176,000 samples = 352,000 bytes of PCM data
    final elevenSecWav = buildTestWav(numSamples: 176000);
    expect(elevenSecWav.length, 44 + 352000);

    final clamped = AskService.clampWavToMaxDuration(elevenSecWav);
    expect(clamped.length, 44 + 320000);

    final view = ByteData.sublistView(clamped);
    expect(view.getUint32(4, Endian.little), 44 + 320000 - 8);
    expect(view.getUint32(40, Endian.little), 320000);
  });

  test('AskResponse and AskError parse contract JSON accurately', () {
    final askResponse = AskResponse.fromJson({
      'question_id': 123,
      'transcript': 'What did that sign say?',
      'lang': 'en',
      'intent': 'READ_TEXT_PAST',
      'answer': 'The sign on your left said "Room 204 – Computer Lab".',
      'confidence': 'high',
      'abstained': false,
      'evidence': [
        {
          'frame_id': 812,
          'seconds_ago': 14,
          'thumb_url': '/media/thumbs/812.jpg',
          'score': 0.91,
        },
      ],
      'timings_ms': {
        'stt': 520,
        'understand': 2,
        'retrieve': 40,
        'gemini': 3100,
        'total': 3700,
      },
    });
    expect(askResponse.questionId, 123);
    expect(askResponse.evidence.single.frameId, 812);

    final askError = AskError.fromJson({
      'error': {
        'code': 'STT_FAILED',
        'message': 'Groq and local Whisper both failed.',
        'speak': 'Sorry, something went wrong. Please try again.',
      },
    });
    expect(askError.code, 'STT_FAILED');
    expect(askError.speak, 'Sorry, something went wrong. Please try again.');
  });

  test('TtsService maps en/ar/fr to en-US, ar, fr-FR and uses settings speechRate', () async {
    final fakeEngine = FakeTtsEngine();
    final settings = SettingsModel(language: 'en-US', speechRate: 0.65);
    final ttsService = TtsService(
      engine: fakeEngine,
      settingsProvider: () => settings,
    );

    await ttsService.speakAnswer('Door ahead.', lang: 'en');
    expect(fakeEngine.languagesSet.last, 'en-US');
    expect(fakeEngine.speechRatesSet.last, 0.65);
    expect(fakeEngine.spokenTexts.last, 'Door ahead.');

    settings.speechRate = 0.45;
    await ttsService.speakAnswer('أمامك باب خشبي.', lang: 'ar');
    expect(fakeEngine.languagesSet.last, 'ar');
    expect(fakeEngine.speechRatesSet.last, 0.45);
    expect(fakeEngine.spokenTexts.last, 'أمامك باب خشبي.');

    await ttsService.speakAnswer('Il y a une porte devant vous.', lang: 'fr');
    expect(fakeEngine.languagesSet.last, 'fr-FR');
    expect(fakeEngine.spokenTexts.last, 'Il y a une porte devant vous.');
  });

  test('Priority: answers suppress incoming alerts and preempt ongoing alerts', () async {
    final fakeEngine = FakeTtsEngine();
    final settings = SettingsModel(
      language: 'en-US',
      speechRate: 0.5,
      alertsEnabled: true,
    );
    final ttsService = TtsService(
      engine: fakeEngine,
      settingsProvider: () => settings,
    );

    // 1. Start speaking an alert
    final alertSpoken = await ttsService.speakAlert('Stairs ahead.', lang: 'en');
    expect(alertSpoken, isTrue);
    expect(ttsService.isSpeakingAlert, isTrue);
    expect(fakeEngine.spokenTexts.last, 'Stairs ahead.');

    // 2. An answer arrives while alert is still speaking -> preempts the alert
    await ttsService.speakAnswer('أمامك باب.', lang: 'ar');
    expect(ttsService.isSpeakingAnswer, isTrue);
    expect(ttsService.isSpeakingAlert, isFalse);
    expect(fakeEngine.languagesSet.last, 'ar');
    expect(fakeEngine.spokenTexts.last, 'أمامك باب.');

    // 3. Another alert arrives while the answer is still speaking -> suppressed!
    final secondAlertSpoken = await ttsService.speakAlert(
      'Obstacle ahead.',
      lang: 'en',
    );
    expect(secondAlertSpoken, isFalse);
    expect(fakeEngine.spokenTexts.last, 'أمامك باب.');

    // 4. Once answer finishes speaking, alerts can speak again
    fakeEngine.simulateCompletion();
    expect(ttsService.isSpeakingAnswer, isFalse);

    final thirdAlertSpoken = await ttsService.speakAlert(
      'Door ahead.',
      lang: 'en',
    );
    expect(thirdAlertSpoken, isTrue);
    expect(fakeEngine.spokenTexts.last, 'Door ahead.');
  });

  testWidgets(
    'Arabic answer is spoken with ar locale and new press triggers barge-in stop',
    (WidgetTester tester) async {
      DateTime fakeNow = DateTime.utc(2026, 11, 13, 12, 0, 0);
      final fakeRecorder = FakeAudioRecorder();
      final fakeTts = FakeTtsEngine();
      final settings = SettingsModel(language: 'ar', speechRate: 0.55);
      final ttsService = TtsService(
        engine: fakeTts,
        settingsProvider: () => settings,
      );

      final mockClient = MockClient.streaming((request, bodyStream) async {
        await bodyStream.toBytes();
        final payload = jsonEncode({
          'question_id': 99,
          'transcript': 'ماذا يوجد أمامي؟',
          'lang': 'ar',
          'intent': 'LIVE_SCENE',
          'answer': 'أمامك باب أبيض وحقيبة سوداء.',
          'confidence': 'high',
          'abstained': false,
          'evidence': [
            {
              'frame_id': null,
              'seconds_ago': 0.5,
              'thumb_url': null,
              'score': 1.0,
            },
          ],
          'timings_ms': {'stt': 300.0, 'gemini': 1100.0, 'total': 1400.0},
        });
        return http.StreamedResponse(
          Stream.value(utf8.encode(payload)),
          200,
          headers: {'content-type': 'application/json'},
        );
      });

      final wsService = SessionWebSocketService()
        ..setSessionIdForTesting('3f2b8c1e-7a4d-4e6f-9b1a-2c5d8e7f6a01');
      final askService = AskService(
        recorder: fakeRecorder,
        httpClient: mockClient,
        now: () => fakeNow,
      );

      await tester.pumpWidget(
        MyApp(
          askService: askService,
          wsService: wsService,
          ttsService: ttsService,
          settings: settings,
          enablePlatformInit: false,
        ),
      );

      // Hold for 1.0 s and release -> receives Arabic answer and speaks it
      final gesture1 = await tester.startGesture(
        tester.getCenter(find.byKey(const Key('hold_to_talk_surface'))),
      );
      await tester.pump();
      fakeNow = fakeNow.add(const Duration(seconds: 1));
      await gesture1.up();
      await tester.pumpAndSettle();

      expect(fakeTts.languagesSet.last, 'ar');
      expect(fakeTts.speechRatesSet.last, 0.55);
      expect(fakeTts.spokenTexts.last, 'أمامك باب أبيض وحقيبة سوداء.');
      expect(ttsService.isSpeakingAnswer, isTrue);

      // Incoming WS alert while answer is still speaking is suppressed
      wsService.onEventReceived?.call({
        'type': 'alert',
        'alert_id': 7,
        'kind': 'door_ahead',
        'message': 'Door ahead.',
        'lang': 'en',
      });
      await tester.pump();
      expect(fakeTts.spokenTexts.last, 'أمامك باب أبيض وحقيبة سوداء.');

      // Barge-in: pressing the screen while answer is still speaking immediately stops TTS
      final stopCountBeforePress = fakeTts.stopCalls;
      final gesture2 = await tester.startGesture(
        tester.getCenter(find.byKey(const Key('hold_to_talk_surface'))),
      );
      await tester.pump();
      expect(fakeTts.stopCalls, greaterThan(stopCountBeforePress));
      expect(ttsService.isSpeakingAnswer, isFalse);

      await gesture2.up();
      await tester.pumpAndSettle();
    },
  );
}
