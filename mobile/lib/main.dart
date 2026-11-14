import 'package:flutter/material.dart';
import 'package:flutter/services.dart';
import 'package:flutter_tts/flutter_tts.dart';
import 'package:flutter_webrtc/flutter_webrtc.dart';
import 'package:http/http.dart' as http;
import 'package:permission_handler/permission_handler.dart';
import 'package:wakelock_plus/wakelock_plus.dart';

import 'models/ask_models.dart';
import 'services/ask_service.dart';
import 'services/session_websocket_service.dart';

void main() async {
  WidgetsFlutterBinding.ensureInitialized();
  await SystemChrome.setPreferredOrientations([
    DeviceOrientation.portraitUp,
    DeviceOrientation.portraitDown,
  ]);
  runApp(const MyApp());
}

class MyApp extends StatelessWidget {
  const MyApp({
    super.key,
    this.askService,
    this.wsService,
    this.enablePlatformInit = true,
  });

  final AskService? askService;
  final SessionWebSocketService? wsService;
  final bool enablePlatformInit;

  @override
  Widget build(BuildContext context) {
    return MaterialApp(
      title: 'Vision Assistant WHIP & WS',
      theme: ThemeData(
        colorScheme: ColorScheme.fromSeed(seedColor: Colors.deepPurple),
        useMaterial3: true,
      ),
      home: MyHomePage(
        title: 'Vision Assistant',
        askService: askService,
        wsService: wsService,
        enablePlatformInit: enablePlatformInit,
      ),
    );
  }
}

class MyHomePage extends StatefulWidget {
  const MyHomePage({
    super.key,
    required this.title,
    this.askService,
    this.wsService,
    this.enablePlatformInit = true,
  });

  final String title;
  final AskService? askService;
  final SessionWebSocketService? wsService;
  final bool enablePlatformInit;

  @override
  State<MyHomePage> createState() => _MyHomePageState();
}

class _MyHomePageState extends State<MyHomePage> {
  // Routes through FastAPI proxy / MediaMTX on port 8889
  final String whipEndpoint = "http://localhost:8889/mystream/whip";

  RTCPeerConnection? _peerConnection;
  MediaStream? _localStream;
  final RTCVideoRenderer _localRenderer = RTCVideoRenderer();
  bool _isStreaming = false;
  String _statusMessage = "Ready to stream";

  // VA-18: Session & WebSocket Service instance
  late final SessionWebSocketService _wsService;
  String _wsStatusText = "Initializing session...";
  final List<String> _eventLogs = [];

  // Hold-to-talk recording + /v1/ask upload
  late final AskService _askService;
  FlutterTts? _tts;
  final String _appLang = "en";
  bool _isHolding = false;
  bool _isRecording = false;
  bool _isUploading = false;
  int? _activePointerId;
  String _askStatusText = "Hold anywhere on the screen to ask a question";
  String _lastTranscript = "";
  String _spokenResponseText = "";

  @override
  void initState() {
    super.initState();
    _wsService = widget.wsService ?? SessionWebSocketService();
    _askService =
        widget.askService ??
        AskService(baseUrlProvider: () => _wsService.activeBaseUrl);
    if (widget.enablePlatformInit) {
      WakelockPlus.enable();
      _tts = FlutterTts();
      _initializeCamera();
      _bootstrapAppSession();
    }
  }

  Future<void> _initializeCamera() async {
    try {
      await _localRenderer.initialize();

      Map<Permission, PermissionStatus> statuses = await [
        Permission.camera,
        Permission.microphone,
      ].request();

      if (statuses[Permission.camera] != PermissionStatus.granted ||
          statuses[Permission.microphone] != PermissionStatus.granted) {
        if (!mounted) return;
        setState(() {
          _statusMessage = "Camera or Microphone permission denied.";
        });
        return;
      }

      // Design §6.1.4: Request video only in WebRTC (audio: false).
      // The microphone is used by `record` for questions.
      final Map<String, dynamic> mediaConstraints = {
        'audio': false,
        'video': {'facingMode': 'environment'},
      };

      _localStream = await navigator.mediaDevices.getUserMedia(
        mediaConstraints,
      );
      _localRenderer.srcObject = _localStream;
      if (!mounted) return;
      setState(() {
        _statusMessage = "Camera loaded successfully!";
      });
    } catch (e) {
      if (!mounted) return;
      setState(() {
        _statusMessage = "Camera error: $e";
      });
    }
  }

  // VA-18 Step 1 & 2: POST session matching exact backend schema parameters
  Future<void> _bootstrapAppSession() async {
    while (mounted && _wsService.sessionId == null) {
      bool success = await _wsService.initializeSession(
        appLang: _appLang,
        consentAt: DateTime.now().toUtc().toIso8601String(),
        consentVer: "1.0",
      );

      if (!mounted) return;
      if (success) {
        setState(() {
          _wsStatusText = "Session established. Connecting WS...";
        });

        _wsService.onEventReceived = (eventData) {
          if (!mounted) return;
          setState(() {
            _eventLogs.insert(
              0,
              "[${eventData['type'] ?? 'event'}] ${eventData.toString()}",
            );
          });
        };

        _wsService.connectWebSocket();

        setState(() {
          _wsStatusText = "WebSocket Connected & Listening";
        });
        return;
      } else {
        setState(() {
          _wsStatusText = "Waiting for backend (retrying)...";
        });
        await Future.delayed(const Duration(seconds: 2));
      }
    }
  }

  Future<void> _speak(String text, {String? lang}) async {
    if (text.isEmpty || _tts == null) return;
    try {
      await _tts!.stop();
      await _tts!.setLanguage(lang ?? _appLang);
      await _tts!.speak(text);
    } catch (_) {
      // Ignore TTS channel errors in headless environments.
    }
  }

  Future<void> _stopSpeech() async {
    if (_tts == null) return;
    try {
      await _tts!.stop();
    } catch (_) {}
  }

  Future<void> _onHoldStart() async {
    if (_isHolding || _isRecording || _isUploading) return;
    _isHolding = true;
    _askService.markHoldStart();

    // Barge-in: new press stops current speech (§6.1.2)
    await _stopSpeech();
    HapticFeedback.mediumImpact();
    SystemSound.play(SystemSoundType.click);

    if (!_isHolding) {
      return;
    }

    final started = await _askService.startRecording(
      onMaxDurationReached: _onMaxDurationReached,
    );

    if (!mounted) return;
    if (!started) {
      _isHolding = false;
      setState(() {
        _askStatusText = "Microphone unavailable. Check permission.";
      });
      return;
    }

    setState(() {
      _isRecording = true;
      _askStatusText = "Recording... Release to send";
    });

    // If the user already released their finger while native AudioRecord was initializing,
    // immediately finish and discard the short tap.
    if (!_isHolding) {
      await _finishRecordingAndUpload();
    }
  }

  Future<void> _onMaxDurationReached() async {
    if (!_isRecording) return;
    _isHolding = false;
    _activePointerId = null;
    await _finishRecordingAndUpload(holdDuration: AskService.maxDuration);
  }

  Future<void> _onHoldEnd() async {
    if (!_isHolding && !_isRecording) return;
    final holdDuration = _askService.markHoldEnd();
    _isHolding = false;
    _activePointerId = null;
    if (!_isRecording) return;
    await _finishRecordingAndUpload(holdDuration: holdDuration);
  }

  Future<void> _finishRecordingAndUpload({Duration? holdDuration}) async {
    HapticFeedback.lightImpact();
    SystemSound.play(SystemSoundType.click);

    if (mounted) {
      setState(() {
        _isRecording = false;
        _isUploading = true;
        _askStatusText = "Uploading question...";
      });
    }

    final outcome = await _askService.stopAndAsk(
      sessionId: _wsService.sessionId,
      lang: _appLang,
      holdDuration: holdDuration,
    );

    if (!mounted) return;

    if (outcome.kind == StopOutcomeKind.ignoredShortTap ||
        outcome.kind == StopOutcomeKind.notRecording) {
      setState(() {
        _isUploading = false;
        _askStatusText = "Hold anywhere on the screen to ask a question";
      });
      return;
    }

    final AskResult? result = outcome.result;
    if (result == null) {
      setState(() {
        _isUploading = false;
        _askStatusText = "Hold anywhere on the screen to ask a question";
      });
      return;
    }

    if (result.isSuccess && result.response != null) {
      final resp = result.response!;
      setState(() {
        _isUploading = false;
        _askStatusText = "Hold anywhere on the screen to ask a question";
        _lastTranscript = resp.transcript;
        _spokenResponseText = resp.answer;
      });
      await _speak(resp.answer, lang: resp.lang);
    } else {
      final speakText = result.spokenText;
      setState(() {
        _isUploading = false;
        _askStatusText = "Hold anywhere on the screen to ask a question";
        _spokenResponseText = speakText;
      });
      await _speak(speakText, lang: _appLang);
    }
  }

  @override
  void dispose() {
    if (widget.enablePlatformInit) {
      _localRenderer.dispose();
    }
    _localStream?.dispose();
    _peerConnection?.dispose();
    _wsService.disconnect();
    _askService.dispose();
    super.dispose();
  }

  Future<void> _startStreaming() async {
    if (_localStream == null) {
      setState(() {
        _statusMessage = "Error: Camera stream not ready yet.";
      });
      return;
    }

    setState(() {
      _statusMessage = "Connecting to MediaMTX...";
    });

    try {
      _peerConnection = await createPeerConnection({'iceServers': []});

      for (var track in _localStream!.getTracks()) {
        _peerConnection!.addTrack(track, _localStream!);
      }

      RTCSessionDescription offer = await _peerConnection!.createOffer();
      await _peerConnection!.setLocalDescription(offer);

      await Future.delayed(const Duration(milliseconds: 500));

      RTCSessionDescription? localDesc = await _peerConnection!
          .getLocalDescription();

      final response = await http
          .post(
            Uri.parse(whipEndpoint),
            headers: {'Content-Type': 'application/sdp'},
            body: localDesc?.sdp ?? '',
          )
          .timeout(const Duration(seconds: 15));

      if (response.statusCode == 201 || response.statusCode == 200) {
        RTCSessionDescription answer = RTCSessionDescription(
          response.body,
          'answer',
        );
        await _peerConnection!.setRemoteDescription(answer);

        setState(() {
          _isStreaming = true;
          _statusMessage = "Streaming Live Successfully!";
        });
      } else {
        setState(() {
          _statusMessage = "Failed (${response.statusCode}): ${response.body}";
        });
      }
    } catch (e) {
      setState(() {
        _statusMessage = "Connection failed: $e";
      });
    }
  }

  @override
  Widget build(BuildContext context) {
    final backgroundColor = _isRecording
        ? const Color(0xFF7F1D1D)
        : _isUploading
        ? const Color(0xFF1E3A8A)
        : const Color(0xFF111827);

    return Scaffold(
      backgroundColor: backgroundColor,
      appBar: AppBar(
        backgroundColor: Colors.black,
        foregroundColor: Colors.white,
        title: Semantics(
          header: true,
          label: widget.title,
          child: Text(widget.title),
        ),
      ),
      body: SafeArea(
        child: Semantics(
          key: const Key('hold_to_talk_surface'),
          button: true,
          label: 'Hold anywhere on the screen to ask a question',
          hint:
              'Press and hold to record your question, then release to send it',
          child: Listener(
            behavior: HitTestBehavior.opaque,
            onPointerDown: (event) {
              if (_activePointerId == null) {
                _activePointerId = event.pointer;
                _onHoldStart();
              }
            },
            onPointerUp: (event) {
              if (_activePointerId == event.pointer) {
                _onHoldEnd();
              }
            },
            onPointerCancel: (event) {
              if (_activePointerId == event.pointer) {
                _onHoldEnd();
              }
            },
            child: GestureDetector(
              behavior: HitTestBehavior.opaque,
              onLongPressStart: (_) => _onHoldStart(),
              onLongPressEnd: (_) => _onHoldEnd(),
              child: SizedBox.expand(
                child: LayoutBuilder(
                  builder: (context, viewportConstraints) {
                    return SingleChildScrollView(
                      padding: const EdgeInsets.all(12.0),
                      child: ConstrainedBox(
                        constraints: BoxConstraints(
                          minHeight: (viewportConstraints.maxHeight - 24.0)
                              .clamp(0.0, double.infinity),
                        ),
                        child: Column(
                          mainAxisAlignment: MainAxisAlignment.spaceBetween,
                          crossAxisAlignment: CrossAxisAlignment.stretch,
                          children: [
                            // Status & Answer section (high contrast, >= 20 sp per §6.1.3)
                            Column(
                              children: [
                                Semantics(
                                  liveRegion: true,
                                  label: 'Recording status: $_askStatusText',
                                  child: Text(
                                    _askStatusText,
                                    key: const Key('ask_status_text'),
                                    textAlign: TextAlign.center,
                                    style: const TextStyle(
                                      fontSize: 20,
                                      fontWeight: FontWeight.bold,
                                      color: Colors.white,
                                    ),
                                  ),
                                ),
                                if (_lastTranscript.isNotEmpty) ...[
                                  const SizedBox(height: 6),
                                  Semantics(
                                    label: 'Heard question: $_lastTranscript',
                                    child: Text(
                                      '“$_lastTranscript”',
                                      key: const Key('transcript_text'),
                                      textAlign: TextAlign.center,
                                      style: const TextStyle(
                                        fontSize: 20,
                                        fontStyle: FontStyle.italic,
                                        color: Colors.amberAccent,
                                      ),
                                    ),
                                  ),
                                ],
                                if (_spokenResponseText.isNotEmpty) ...[
                                  const SizedBox(height: 8),
                                  Semantics(
                                    liveRegion: true,
                                    label:
                                        'Assistant response: $_spokenResponseText',
                                    child: Container(
                                      padding: const EdgeInsets.all(10),
                                      decoration: BoxDecoration(
                                        color: Colors.black54,
                                        borderRadius: BorderRadius.circular(10),
                                        border: Border.all(
                                          color: Colors.white54,
                                        ),
                                      ),
                                      child: Text(
                                        _spokenResponseText,
                                        key: const Key('spoken_response_text'),
                                        textAlign: TextAlign.center,
                                        style: const TextStyle(
                                          fontSize: 20,
                                          fontWeight: FontWeight.w600,
                                          color: Colors.white,
                                        ),
                                      ),
                                    ),
                                  ),
                                ],
                              ],
                            ),

                            const SizedBox(height: 8),

                            // Camera & Stream helper view for sighted helpers (§6.1.1)
                            Column(
                              children: [
                                if (widget.enablePlatformInit)
                                  Semantics(
                                    label: 'Rear camera preview',
                                    child: SizedBox(
                                      width: 130,
                                      height: 140,
                                      child: ClipRRect(
                                        borderRadius: BorderRadius.circular(12),
                                        child: RTCVideoView(
                                          _localRenderer,
                                          mirror: false,
                                        ),
                                      ),
                                    ),
                                  ),
                                const SizedBox(height: 6),
                                Semantics(
                                  label:
                                      'Camera stream status: $_statusMessage',
                                  child: Text(
                                    _statusMessage,
                                    textAlign: TextAlign.center,
                                    style: TextStyle(
                                      fontSize: 13,
                                      fontWeight: FontWeight.w500,
                                      color: _isStreaming
                                          ? Colors.greenAccent
                                          : Colors.white70,
                                    ),
                                  ),
                                ),
                                const SizedBox(height: 6),
                                Semantics(
                                  button: true,
                                  label: _isStreaming
                                      ? 'Streaming live'
                                      : 'Start WHIP video stream',
                                  child: ElevatedButton(
                                    onPressed: _isStreaming
                                        ? null
                                        : _startStreaming,
                                    child: Text(
                                      _isStreaming
                                          ? 'Streaming Live...'
                                          : 'Start WHIP Stream',
                                    ),
                                  ),
                                ),
                              ],
                            ),

                            const SizedBox(height: 8),

                            // WebSocket status & logs
                            Column(
                              crossAxisAlignment: CrossAxisAlignment.start,
                              children: [
                                Semantics(
                                  label:
                                      'WebSocket connection status: $_wsStatusText',
                                  child: Text(
                                    "WS Status: $_wsStatusText",
                                    style: const TextStyle(
                                      fontWeight: FontWeight.bold,
                                      fontSize: 12,
                                      color: Colors.white70,
                                    ),
                                  ),
                                ),
                                const SizedBox(height: 4),
                                Semantics(
                                  label: 'Live event logs',
                                  child: Container(
                                    height: 60,
                                    decoration: BoxDecoration(
                                      color: Colors.black38,
                                      border: Border.all(color: Colors.white24),
                                      borderRadius: BorderRadius.circular(8),
                                    ),
                                    child: ListView.builder(
                                      itemCount: _eventLogs.length,
                                      itemBuilder: (context, index) {
                                        return Padding(
                                          padding: const EdgeInsets.symmetric(
                                            horizontal: 8,
                                            vertical: 2,
                                          ),
                                          child: Text(
                                            _eventLogs[index],
                                            style: const TextStyle(
                                              fontFamily: 'monospace',
                                              fontSize: 11,
                                              color: Colors.white70,
                                            ),
                                          ),
                                        );
                                      },
                                    ),
                                  ),
                                ),
                              ],
                            ),
                          ],
                        ),
                      ),
                    );
                  },
                ),
              ),
            ),
          ),
        ),
      ),
    );
  }
}
