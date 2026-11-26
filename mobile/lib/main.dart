import 'package:flutter/material.dart';
import 'package:flutter/services.dart';
import 'package:flutter_webrtc/flutter_webrtc.dart';
import 'package:http/http.dart' as http;
import 'package:permission_handler/permission_handler.dart';
import 'package:wakelock_plus/wakelock_plus.dart';

import 'models/ask_models.dart';
import 'models/settings_models.dart';
import 'services/ask_service.dart';
import 'services/session_websocket_service.dart';
import 'services/tts_service.dart';

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
    this.ttsService,
    this.settings,
    this.enablePlatformInit = true,
  });

  final AskService? askService;
  final SessionWebSocketService? wsService;
  final TtsService? ttsService;
  final SettingsModel? settings;
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
        ttsService: ttsService,
        settings: settings,
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
    this.ttsService,
    this.settings,
    this.enablePlatformInit = true,
  });

  final String title;
  final AskService? askService;
  final SessionWebSocketService? wsService;
  final TtsService? ttsService;
  final SettingsModel? settings;
  final bool enablePlatformInit;

  @override
  State<MyHomePage> createState() => _MyHomePageState();
}

class _MyHomePageState extends State<MyHomePage> {
  // Routes through FastAPI proxy / MediaMTX on port 8889 using single source of truth IP
  String get whipEndpoint =>
      "http://${_settings.serverIp}:8889/mystream/whip";

  RTCPeerConnection? _peerConnection;
  MediaStream? _localStream;
  final RTCVideoRenderer _localRenderer = RTCVideoRenderer();
  bool _isStreaming = false;
  bool _isConnectingStream = false;
  String _statusMessage = "Ready to stream";

  // VA-18: Session & WebSocket Service instance
  late final SessionWebSocketService _wsService;
  String _wsStatusText = "Initializing session...";
  final List<String> _eventLogs = [];

  // Hold-to-talk recording + /v1/ask upload + TTS (EN/AR/FR)
  late final AskService _askService;
  late final TtsService _ttsService;
  late SettingsModel _settings;
  bool _isHolding = false;
  bool _isRecording = false;
  bool _isUploading = false;
  int? _activePointerId;
  String _askStatusText = "Hold anywhere on the screen to ask a question";
  String _lastTranscript = "";
  String _spokenResponseText = "";

  String get _appLang => AskService.normalizeLang(_settings.language);

  @override
  void initState() {
    super.initState();
    _settings = widget.settings ?? SettingsModel();
    _wsService = widget.wsService ?? SessionWebSocketService();
    _wsService.onEventReceived = _handleWsEvent;
    _askService =
        widget.askService ??
        AskService(baseUrlProvider: () => _wsService.activeBaseUrl);
    _ttsService =
        widget.ttsService ??
        TtsService(
          engine: widget.enablePlatformInit ? null : NoOpTtsEngineAdapter(),
          baseUrlProvider: () => _wsService.activeBaseUrl,
          settingsProvider: () => _settings,
        );
    if (widget.enablePlatformInit) {
      WakelockPlus.enable();
      if (widget.settings == null) {
        _loadSavedSettings();
      }
      _initializeCamera();
      _bootstrapAppSession();
    }
  }

  Future<void> _loadSavedSettings() async {
    try {
      final loaded = await SettingsModel.load();
      if (!mounted) return;
      setState(() {
        // Always prefer the latest defaultServerIp synced by start.ps1 if it changed networks
        _settings = loaded..serverIp = SettingsModel.defaultServerIp;
      });
      _wsService.updateServerIp(_settings.serverIp);
    } catch (_) {}
  }

  Future<void> _selectLanguage(String localeCode) async {
    setState(() {
      _settings.language = localeCode;
    });
    if (widget.enablePlatformInit) {
      try {
        await _settings.save();
      } catch (_) {}
    }
  }

  Future<void> _updateSpeechRate(double rate) async {
    setState(() {
      _settings.speechRate = double.parse(rate.toStringAsFixed(2));
    });
    if (widget.enablePlatformInit) {
      try {
        await _settings.save();
      } catch (_) {}
    }
  }

  Future<void> _handleWsEvent(Map<String, dynamic> eventData) async {
    if (!mounted) return;
    setState(() {
      _eventLogs.insert(
        0,
        "[${eventData['type'] ?? 'event'}] ${eventData.toString()}",
      );
    });

    // Design §6.1.2 & §6.6: Speak alerts unless recording, disabled, or an answer is speaking.
    if (eventData['type'] == 'alert') {
      if (_isHolding || _isRecording || !_settings.alertsEnabled) {
        return;
      }
      final message = eventData['message']?.toString() ?? '';
      final alertLang = eventData['lang']?.toString() ?? _appLang;
      if (message.isEmpty) return;

      if (!_ttsService.isSpeakingAnswer) {
        HapticFeedback.heavyImpact();
      }
      await _ttsService.speakAlert(message, lang: alertLang);
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

        _wsService.onEventReceived = _handleWsEvent;
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

  Future<void> _speakAnswer(String text, {String? lang}) async {
    await _ttsService.speakAnswer(text, lang: lang ?? _appLang);
  }

  Future<void> _stopSpeech() async {
    await _ttsService.stop();
  }

  Future<void> _onHoldStart() async {
    // Barge-in: a new press immediately stops current speech (§6.1.2)
    if (_isHolding || _isRecording || _isUploading) {
      await _stopSpeech();
      return;
    }
    _isHolding = true;
    _askService.markHoldStart();
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
      await _speakAnswer(resp.answer, lang: resp.lang);
    } else {
      final speakText = result.spokenText;
      setState(() {
        _isUploading = false;
        _askStatusText = "Hold anywhere on the screen to ask a question";
        _spokenResponseText = speakText;
      });
      await _speakAnswer(speakText, lang: _appLang);
    }
  }

  void _showSettingsSheet() {
    showModalBottomSheet<void>(
      context: context,
      backgroundColor: const Color(0xFF0F172A),
      shape: const RoundedRectangleBorder(
        borderRadius: BorderRadius.vertical(top: Radius.circular(24)),
      ),
      builder: (sheetContext) {
        return StatefulBuilder(
          builder: (context, setSheetState) {
            return Padding(
              padding: const EdgeInsets.fromLTRB(20, 12, 20, 24),
              child: Column(
                mainAxisSize: MainAxisSize.min,
                crossAxisAlignment: CrossAxisAlignment.stretch,
                children: [
                  Center(
                    child: Container(
                      width: 40,
                      height: 4,
                      decoration: BoxDecoration(
                        color: Colors.white24,
                        borderRadius: BorderRadius.circular(2),
                      ),
                    ),
                  ),
                  const SizedBox(height: 16),
                  Semantics(
                    header: true,
                    label: 'Speech and alert settings',
                    child: const Text(
                      'Settings',
                      style: TextStyle(
                        fontSize: 20,
                        fontWeight: FontWeight.w700,
                        color: Colors.white,
                        letterSpacing: 0.3,
                      ),
                    ),
                  ),
                  const SizedBox(height: 16),
                  Container(
                    padding: const EdgeInsets.symmetric(
                      horizontal: 14,
                      vertical: 10,
                    ),
                    decoration: BoxDecoration(
                      color: const Color(0xFF1E293B),
                      borderRadius: BorderRadius.circular(14),
                      border: Border.all(color: Colors.white12),
                    ),
                    child: Column(
                      crossAxisAlignment: CrossAxisAlignment.stretch,
                      children: [
                        Semantics(
                          label:
                              'Speech rate: ${_settings.speechRate.toStringAsFixed(2)}',
                          child: Text(
                            'Speech Rate: ${_settings.speechRate.toStringAsFixed(2)}',
                            style: const TextStyle(
                              color: Colors.white,
                              fontSize: 16,
                              fontWeight: FontWeight.w600,
                            ),
                          ),
                        ),
                        Semantics(
                          slider: true,
                          label: 'Adjust speech rate',
                          child: Slider(
                            key: const Key('speech_rate_slider'),
                            activeColor: const Color(0xFF38BDF8),
                            inactiveColor: Colors.white12,
                            value: _settings.speechRate.clamp(0.2, 1.0),
                            min: 0.2,
                            max: 1.0,
                            divisions: 8,
                            onChanged: (val) {
                              setSheetState(() {});
                              _updateSpeechRate(val);
                            },
                          ),
                        ),
                      ],
                    ),
                  ),
                  const SizedBox(height: 10),
                  Container(
                    decoration: BoxDecoration(
                      color: const Color(0xFF1E293B),
                      borderRadius: BorderRadius.circular(14),
                      border: Border.all(color: Colors.white12),
                    ),
                    child: Semantics(
                      toggled: _settings.alertsEnabled,
                      label: 'Obstacle alerts toggle',
                      child: SwitchListTile(
                        activeThumbColor: const Color(0xFF38BDF8),
                        title: const Text(
                          'Alerts Enabled',
                          style: TextStyle(
                            color: Colors.white,
                            fontWeight: FontWeight.w600,
                          ),
                        ),
                        value: _settings.alertsEnabled,
                        onChanged: (val) async {
                          setSheetState(() {
                            _settings.alertsEnabled = val;
                          });
                          setState(() {});
                          if (widget.enablePlatformInit) {
                            try {
                              await _settings.save();
                            } catch (_) {}
                          }
                        },
                      ),
                    ),
                  ),
                ],
              ),
            );
          },
        );
      },
    );
  }

  Widget _buildLangButton(String label, String localeCode, String shortLang) {
    final bool isSelected = _appLang == shortLang;
    return Semantics(
      button: true,
      selected: isSelected,
      label: 'Select $label language',
      child: InkWell(
        key: Key('lang_btn_$shortLang'),
        onTap: () => _selectLanguage(localeCode),
        borderRadius: BorderRadius.circular(10),
        child: AnimatedContainer(
          duration: const Duration(milliseconds: 180),
          margin: const EdgeInsets.symmetric(horizontal: 3, vertical: 10),
          padding: const EdgeInsets.symmetric(horizontal: 11, vertical: 5),
          decoration: BoxDecoration(
            gradient: isSelected
                ? const LinearGradient(
                    colors: [Color(0xFF38BDF8), Color(0xFF6366F1)],
                  )
                : null,
            color: isSelected ? null : const Color(0xFF1E293B),
            borderRadius: BorderRadius.circular(10),
            border: Border.all(
              color: isSelected ? Colors.white38 : Colors.white12,
            ),
          ),
          alignment: Alignment.center,
          child: Text(
            shortLang.toUpperCase(),
            style: TextStyle(
              fontSize: 13,
              fontWeight: FontWeight.w800,
              letterSpacing: 0.6,
              color: isSelected ? Colors.white : Colors.white70,
            ),
          ),
        ),
      ),
    );
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
    _ttsService.stop();
    super.dispose();
  }

  Future<void> _startStreaming() async {
    if (_isConnectingStream || _isStreaming) {
      return;
    }
    if (_localStream == null) {
      setState(() {
        _statusMessage = "Error: Camera stream not ready yet.";
      });
      return;
    }

    setState(() {
      _isConnectingStream = true;
      _statusMessage = "Connecting to MediaMTX...";
    });

    try {
      await _peerConnection?.dispose();
      final pc = await createPeerConnection({'iceServers': []});
      _peerConnection = pc;

      for (var track in _localStream!.getTracks()) {
        await pc.addTrack(track, _localStream!);
      }

      RTCSessionDescription offer = await pc.createOffer();
      await pc.setLocalDescription(offer);

      await Future.delayed(const Duration(milliseconds: 500));

      RTCSessionDescription? localDesc = await pc.getLocalDescription();

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
        await pc.setRemoteDescription(answer);

        if (!mounted) return;
        setState(() {
          _isStreaming = true;
          _statusMessage = "Streaming Live Successfully!";
        });
      } else {
        if (!mounted) return;
        setState(() {
          _statusMessage = "Failed (${response.statusCode}): ${response.body}";
        });
      }
    } catch (e) {
      if (!mounted) return;
      setState(() {
        _statusMessage = "Connection failed: $e";
      });
    } finally {
      if (mounted) {
        setState(() {
          _isConnectingStream = false;
        });
      }
    }
  }

  @override
  Widget build(BuildContext context) {
    final List<Color> gradientColors = _isRecording
        ? const [Color(0xFF2B070C), Color(0xFF7F1D1D), Color(0xFF0D0406)]
        : _isUploading
        ? const [Color(0xFF07132B), Color(0xFF1E3A8A), Color(0xFF050A17)]
        : const [Color(0xFF050811), Color(0xFF0F172A), Color(0xFF070B14)];

    final Color accentColor = _isRecording
        ? const Color(0xFFFF4D6D)
        : _isUploading
        ? const Color(0xFF60A5FA)
        : const Color(0xFF38BDF8);

    final IconData heroIcon = _isRecording
        ? Icons.graphic_eq_rounded
        : _isUploading
        ? Icons.auto_awesome_rounded
        : Icons.mic_rounded;

    final bool wsConnected = _wsStatusText.contains('Connected');

    return Scaffold(
      backgroundColor: gradientColors.first,
      appBar: AppBar(
        backgroundColor: const Color(0xFF050811),
        surfaceTintColor: Colors.transparent,
        elevation: 0,
        toolbarHeight: 56,
        foregroundColor: Colors.white,
        title: Row(
          children: [
            Container(
              padding: const EdgeInsets.all(7),
              decoration: BoxDecoration(
                gradient: LinearGradient(
                  colors: [
                    accentColor.withValues(alpha: 0.28),
                    const Color(0xFF6366F1).withValues(alpha: 0.22),
                  ],
                ),
                borderRadius: BorderRadius.circular(11),
                border: Border.all(
                  color: accentColor.withValues(alpha: 0.5),
                ),
              ),
              child: Icon(
                Icons.remove_red_eye_rounded,
                size: 17,
                color: accentColor,
              ),
            ),
            const SizedBox(width: 10),
            Expanded(
              child: Semantics(
                header: true,
                label: widget.title,
                child: Text(
                  widget.title,
                  overflow: TextOverflow.ellipsis,
                  style: const TextStyle(
                    fontSize: 17,
                    fontWeight: FontWeight.w800,
                    letterSpacing: 0.4,
                  ),
                ),
              ),
            ),
          ],
        ),
        actions: [
          _buildLangButton('English', 'en-US', 'en'),
          _buildLangButton('Arabic', 'ar', 'ar'),
          Semantics(
            button: true,
            label: 'Open speech and alert settings',
            child: IconButton(
              key: const Key('settings_button'),
              icon: const Icon(Icons.tune_rounded, size: 21),
              tooltip: 'Settings',
              onPressed: _showSettingsSheet,
            ),
          ),
          const SizedBox(width: 4),
        ],
      ),
      body: AnimatedContainer(
        duration: const Duration(milliseconds: 260),
        decoration: BoxDecoration(
          gradient: LinearGradient(
            begin: Alignment.topCenter,
            end: Alignment.bottomCenter,
            colors: gradientColors,
          ),
        ),
        child: SafeArea(
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
                      final double cameraHeight =
                          (viewportConstraints.maxHeight * 0.36).clamp(
                            195.0,
                            295.0,
                          );
                      return SingleChildScrollView(
                        padding: const EdgeInsets.fromLTRB(14, 6, 14, 10),
                        child: ConstrainedBox(
                          constraints: BoxConstraints(
                            minHeight: (viewportConstraints.maxHeight - 16.0)
                                .clamp(0.0, double.infinity),
                          ),
                          child: Column(
                            mainAxisAlignment: MainAxisAlignment.spaceBetween,
                            crossAxisAlignment: CrossAxisAlignment.stretch,
                            children: [
                              // 1. PROMINENT WIDE CAMERA VIEWPORT + STREAM BAR
                              Container(
                                decoration: BoxDecoration(
                                  color: const Color(0xFF0F172A),
                                  borderRadius: BorderRadius.circular(22),
                                  border: Border.all(
                                    color: _isStreaming
                                        ? const Color(0xFF10B981)
                                        : accentColor.withValues(alpha: 0.35),
                                    width: 1.6,
                                  ),
                                  boxShadow: [
                                    BoxShadow(
                                      color:
                                          (_isStreaming
                                                  ? const Color(0xFF10B981)
                                                  : accentColor)
                                              .withValues(alpha: 0.14),
                                      blurRadius: 20,
                                      offset: const Offset(0, 6),
                                    ),
                                  ],
                                ),
                                child: Column(
                                  crossAxisAlignment:
                                      CrossAxisAlignment.stretch,
                                  children: [
                                    if (widget.enablePlatformInit)
                                      Semantics(
                                        label: 'Rear camera preview',
                                        child: SizedBox(
                                          height: cameraHeight,
                                          child: ClipRRect(
                                            borderRadius:
                                                const BorderRadius.vertical(
                                                  top: Radius.circular(20.5),
                                                ),
                                            child: Stack(
                                              fit: StackFit.expand,
                                              children: [
                                                RTCVideoView(
                                                  _localRenderer,
                                                  mirror: false,
                                                  objectFit: RTCVideoViewObjectFit
                                                      .RTCVideoViewObjectFitCover,
                                                ),
                                                // Top floating LIVE / STANDBY badge
                                                Positioned(
                                                  top: 10,
                                                  left: 10,
                                                  child: Container(
                                                    padding:
                                                        const EdgeInsets.symmetric(
                                                          horizontal: 10,
                                                          vertical: 5,
                                                        ),
                                                    decoration: BoxDecoration(
                                                      color: Colors.black
                                                          .withValues(
                                                            alpha: 0.65,
                                                          ),
                                                      borderRadius:
                                                          BorderRadius.circular(
                                                            20,
                                                          ),
                                                      border: Border.all(
                                                        color: _isStreaming
                                                            ? const Color(
                                                                0xFF34D399,
                                                              )
                                                            : Colors.white24,
                                                      ),
                                                    ),
                                                    child: Row(
                                                      mainAxisSize:
                                                          MainAxisSize.min,
                                                      children: [
                                                        Container(
                                                          width: 7,
                                                          height: 7,
                                                          decoration: BoxDecoration(
                                                            shape:
                                                                BoxShape.circle,
                                                            color: _isStreaming
                                                                ? const Color(
                                                                    0xFF34D399,
                                                                  )
                                                                : const Color(
                                                                    0xFFFBBF24,
                                                                  ),
                                                          ),
                                                        ),
                                                        const SizedBox(
                                                          width: 6,
                                                        ),
                                                        Text(
                                                          _isStreaming
                                                              ? 'LIVE CAMERA'
                                                              : 'CAMERA PREVIEW',
                                                          style:
                                                              const TextStyle(
                                                                color: Colors
                                                                    .white,
                                                                fontSize: 11,
                                                                fontWeight:
                                                                    FontWeight
                                                                        .w800,
                                                                letterSpacing:
                                                                    0.6,
                                                              ),
                                                        ),
                                                      ],
                                                    ),
                                                  ),
                                                ),
                                              ],
                                            ),
                                          ),
                                        ),
                                      ),
                                    // Bottom bar inside camera card: Status + Stream Button
                                    Container(
                                      padding: const EdgeInsets.symmetric(
                                        horizontal: 12,
                                        vertical: 10,
                                      ),
                                      decoration: const BoxDecoration(
                                        color: Color(0xFF131C31),
                                        borderRadius: BorderRadius.vertical(
                                          bottom: Radius.circular(20.5),
                                        ),
                                      ),
                                      child: Row(
                                        children: [
                                          Expanded(
                                            child: Semantics(
                                              label:
                                                  'Camera stream status: $_statusMessage',
                                              child: Text(
                                                _statusMessage,
                                                maxLines: 2,
                                                overflow: TextOverflow.ellipsis,
                                                style: TextStyle(
                                                  fontSize: 12.5,
                                                  fontWeight: FontWeight.w600,
                                                  color: _isStreaming
                                                      ? const Color(0xFF34D399)
                                                      : Colors.white70,
                                                ),
                                              ),
                                            ),
                                          ),
                                          const SizedBox(width: 10),
                                          Semantics(
                                            button: true,
                                            label: _isStreaming
                                                ? 'Streaming live'
                                                : 'Start WHIP video stream',
                                            child: ElevatedButton.icon(
                                              style: ElevatedButton.styleFrom(
                                                backgroundColor: _isStreaming
                                                    ? const Color(0xFF059669)
                                                    : const Color(0xFF2563EB),
                                                foregroundColor: Colors.white,
                                                disabledBackgroundColor:
                                                    _isStreaming
                                                    ? const Color(
                                                        0xFF059669,
                                                      ).withValues(alpha: 0.75)
                                                    : Colors.white12,
                                                disabledForegroundColor:
                                                    Colors.white,
                                                padding:
                                                    const EdgeInsets.symmetric(
                                                      horizontal: 14,
                                                      vertical: 9,
                                                    ),
                                                shape: RoundedRectangleBorder(
                                                  borderRadius:
                                                      BorderRadius.circular(12),
                                                ),
                                              ),
                                              onPressed:
                                                  (_isStreaming ||
                                                      _isConnectingStream)
                                                  ? null
                                                  : _startStreaming,
                                              icon: Icon(
                                                _isStreaming
                                                    ? Icons.sensors_rounded
                                                    : Icons.videocam_rounded,
                                                size: 16,
                                              ),
                                              label: Text(
                                                _isStreaming
                                                    ? 'Streaming Live...'
                                                    : _isConnectingStream
                                                    ? 'Connecting...'
                                                    : 'Start WHIP Stream',
                                                style: const TextStyle(
                                                  fontSize: 12.5,
                                                  fontWeight: FontWeight.w700,
                                                ),
                                              ),
                                            ),
                                          ),
                                        ],
                                      ),
                                    ),
                                  ],
                                ),
                              ),

                              const SizedBox(height: 10),

                              // 2. HERO HOLD-TO-TALK & VOICE ANSWER PANEL (high contrast, >= 20 sp per §6.1.3)
                              AnimatedContainer(
                                duration: const Duration(milliseconds: 220),
                                padding: const EdgeInsets.symmetric(
                                  horizontal: 16,
                                  vertical: 14,
                                ),
                                decoration: BoxDecoration(
                                  gradient: LinearGradient(
                                    begin: Alignment.topLeft,
                                    end: Alignment.bottomRight,
                                    colors: [
                                      const Color(
                                        0xFF1E293B,
                                      ).withValues(alpha: 0.85),
                                      const Color(
                                        0xFF0F172A,
                                      ).withValues(alpha: 0.92),
                                    ],
                                  ),
                                  borderRadius: BorderRadius.circular(22),
                                  border: Border.all(
                                    color: accentColor.withValues(alpha: 0.55),
                                    width: 1.4,
                                  ),
                                  boxShadow: [
                                    BoxShadow(
                                      color: accentColor.withValues(
                                        alpha: 0.16,
                                      ),
                                      blurRadius: 22,
                                      spreadRadius: 1,
                                    ),
                                  ],
                                ),
                                child: Column(
                                  children: [
                                    // Glowing double-ring microphone orb
                                    Container(
                                      padding: const EdgeInsets.all(6),
                                      decoration: BoxDecoration(
                                        shape: BoxShape.circle,
                                        color: accentColor.withValues(
                                          alpha: 0.10,
                                        ),
                                      ),
                                      child: AnimatedContainer(
                                        duration: const Duration(
                                          milliseconds: 220,
                                        ),
                                        width: 54,
                                        height: 54,
                                        decoration: BoxDecoration(
                                          shape: BoxShape.circle,
                                          gradient: LinearGradient(
                                            begin: Alignment.topLeft,
                                            end: Alignment.bottomRight,
                                            colors: [
                                              accentColor.withValues(
                                                alpha: 0.35,
                                              ),
                                              accentColor.withValues(
                                                alpha: 0.12,
                                              ),
                                            ],
                                          ),
                                          border: Border.all(
                                            color: accentColor,
                                            width: 2.0,
                                          ),
                                        ),
                                        child: Icon(
                                          heroIcon,
                                          color: Colors.white,
                                          size: 27,
                                        ),
                                      ),
                                    ),
                                    const SizedBox(height: 10),
                                    Semantics(
                                      liveRegion: true,
                                      label:
                                          'Recording status: $_askStatusText',
                                      child: Text(
                                        _askStatusText,
                                        key: const Key('ask_status_text'),
                                        textAlign: TextAlign.center,
                                        style: const TextStyle(
                                          fontSize: 20,
                                          fontWeight: FontWeight.w800,
                                          color: Colors.white,
                                          height: 1.22,
                                        ),
                                      ),
                                    ),
                                    if (_lastTranscript.isNotEmpty) ...[
                                      const SizedBox(height: 8),
                                      Semantics(
                                        label:
                                            'Heard question: $_lastTranscript',
                                        child: Text(
                                          '“$_lastTranscript”',
                                          key: const Key('transcript_text'),
                                          textAlign: TextAlign.center,
                                          style: const TextStyle(
                                            fontSize: 20,
                                            fontStyle: FontStyle.italic,
                                            color: Color(0xFFFDE047),
                                          ),
                                        ),
                                      ),
                                    ],
                                    if (_spokenResponseText.isNotEmpty) ...[
                                      const SizedBox(height: 10),
                                      Semantics(
                                        liveRegion: true,
                                        label:
                                            'Assistant response: $_spokenResponseText',
                                        child: Container(
                                          width: double.infinity,
                                          padding: const EdgeInsets.symmetric(
                                            horizontal: 14,
                                            vertical: 12,
                                          ),
                                          decoration: BoxDecoration(
                                            color: const Color(0xFF050811)
                                                .withValues(alpha: 0.85),
                                            borderRadius: BorderRadius.circular(
                                              16,
                                            ),
                                            border: Border.all(
                                              color: accentColor.withValues(
                                                alpha: 0.35,
                                              ),
                                            ),
                                          ),
                                          child: Text(
                                            _spokenResponseText,
                                            key: const Key(
                                              'spoken_response_text',
                                            ),
                                            textAlign: TextAlign.center,
                                            style: const TextStyle(
                                              fontSize: 20,
                                              fontWeight: FontWeight.w600,
                                              color: Colors.white,
                                              height: 1.28,
                                            ),
                                          ),
                                        ),
                                      ),
                                    ],
                                  ],
                                ),
                              ),

                              const SizedBox(height: 8),

                              // 3. COMPACT TELEMETRY & WEBSOCKET FOOTER
                              Container(
                                padding: const EdgeInsets.symmetric(
                                  horizontal: 10,
                                  vertical: 8,
                                ),
                                decoration: BoxDecoration(
                                  color: const Color(
                                    0xFF050811,
                                  ).withValues(alpha: 0.72),
                                  borderRadius: BorderRadius.circular(14),
                                  border: Border.all(color: Colors.white10),
                                ),
                                child: Column(
                                  crossAxisAlignment: CrossAxisAlignment.start,
                                  children: [
                                    Row(
                                      children: [
                                        Container(
                                          width: 8,
                                          height: 8,
                                          decoration: BoxDecoration(
                                            shape: BoxShape.circle,
                                            color: wsConnected
                                                ? const Color(0xFF34D399)
                                                : const Color(0xFFFBBF24),
                                          ),
                                        ),
                                        const SizedBox(width: 6),
                                        Expanded(
                                          child: Semantics(
                                            label:
                                                'WebSocket connection status: $_wsStatusText',
                                            child: Text(
                                              "WS Status: $_wsStatusText",
                                              overflow: TextOverflow.ellipsis,
                                              style: const TextStyle(
                                                fontWeight: FontWeight.w700,
                                                fontSize: 11.5,
                                                color: Colors.white70,
                                              ),
                                            ),
                                          ),
                                        ),
                                      ],
                                    ),
                                    const SizedBox(height: 4),
                                    Semantics(
                                      label: 'Live event logs',
                                      child: Container(
                                        height: 36,
                                        decoration: BoxDecoration(
                                          color: Colors.black38,
                                          border: Border.all(
                                            color: Colors.white10,
                                          ),
                                          borderRadius: BorderRadius.circular(
                                            8,
                                          ),
                                        ),
                                        child: ListView.builder(
                                          itemCount: _eventLogs.length,
                                          itemBuilder: (context, index) {
                                            return Padding(
                                              padding:
                                                  const EdgeInsets.symmetric(
                                                    horizontal: 8,
                                                    vertical: 2,
                                                  ),
                                              child: Text(
                                                _eventLogs[index],
                                                style: const TextStyle(
                                                  fontFamily: 'monospace',
                                                  fontSize: 10.5,
                                                  color: Colors.white60,
                                                ),
                                              ),
                                            );
                                          },
                                        ),
                                      ),
                                    ),
                                  ],
                                ),
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
      ),
    );
  }
}
