import 'package:flutter/material.dart';
import 'package:flutter/services.dart';
import 'package:flutter_webrtc/flutter_webrtc.dart';
import 'package:http/http.dart' as http;
import 'package:permission_handler/permission_handler.dart';

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
  const MyApp({super.key});

  @override
  Widget build(BuildContext context) {
    return MaterialApp(
      title: 'Vision Assistant WHIP & WS',
      theme: ThemeData(
        colorScheme: ColorScheme.fromSeed(seedColor: Colors.deepPurple),
        useMaterial3: true,
      ),
      home: const MyHomePage(title: 'Vision Assistant Stream & WS'),
    );
  }
}

class MyHomePage extends StatefulWidget {
  const MyHomePage({super.key, required this.title});

  final String title;

  @override
  State<MyHomePage> createState() => _MyHomePageState();
}

class _MyHomePageState extends State<MyHomePage> {
  // Routes through FastAPI proxy on port 8889
  final String whipEndpoint = "http://192.168.8.111:8889/mystream/whip";

  RTCPeerConnection? _peerConnection;
  MediaStream? _localStream;
  final RTCVideoRenderer _localRenderer = RTCVideoRenderer();
  bool _isStreaming = false;
  String _statusMessage = "Ready to stream";

  // VA-18: Session & WebSocket Service instance
  late final SessionWebSocketService _wsService;
  String _wsStatusText = "Initializing session...";
  final List<String> _eventLogs = [];

  @override
  void initState() {
    super.initState();
    _wsService = SessionWebSocketService();
    _initializeCamera();
    _bootstrapAppSession();
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
        setState(() {
          _statusMessage = "Camera or Microphone permission denied.";
        });
        return;
      }

      final Map<String, dynamic> mediaConstraints = {
        'audio': true,
        'video': {'facingMode': 'environment'},
      };

      _localStream = await navigator.mediaDevices.getUserMedia(
        mediaConstraints,
      );
      _localRenderer.srcObject = _localStream;
      setState(() {
        _statusMessage = "Camera loaded successfully!";
      });
    } catch (e) {
      setState(() {
        _statusMessage = "Camera error: $e";
      });
    }
  }

  // VA-18 Step 1 & 2: POST session matching exact backend schema parameters
  Future<void> _bootstrapAppSession() async {
    bool success = await _wsService.initializeSession(
      appLang: "en",
      consentAt: DateTime.now().toUtc().toIso8601String(),
      consentVer: "1.0",
    );

    if (success) {
      setState(() {
        _wsStatusText = "Session established. Connecting WS...";
      });

      _wsService.onEventReceived = (eventData) {
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
    } else {
      setState(() {
        _wsStatusText = "Session creation failed. Check backend.";
      });
    }
  }

  @override
  void dispose() {
    _localRenderer.dispose();
    _localStream?.dispose();
    _peerConnection?.dispose();
    _wsService.disconnect();
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
    return Scaffold(
      appBar: AppBar(
        backgroundColor: Theme.of(context).colorScheme.inversePrimary,
        title: Text(widget.title),
      ),
      body: SafeArea(
        child: SingleChildScrollView(
          padding: const EdgeInsets.all(16.0),
          child: Center(
            child: Column(
              mainAxisAlignment: MainAxisAlignment.center,
              children: [
                SizedBox(
                  width: 220,
                  height: 260,
                  child: ClipRRect(
                    borderRadius: BorderRadius.circular(12),
                    child: RTCVideoView(_localRenderer, mirror: false),
                  ),
                ),
                const SizedBox(height: 8),
                Text(
                  _statusMessage,
                  textAlign: TextAlign.center,
                  style: TextStyle(
                    fontSize: 13,
                    fontWeight: FontWeight.w500,
                    color: _isStreaming ? Colors.green : Colors.black87,
                  ),
                ),
                const SizedBox(height: 10),
                ElevatedButton(
                  onPressed: _isStreaming ? null : _startStreaming,
                  style: ElevatedButton.styleFrom(
                    padding: const EdgeInsets.symmetric(
                      horizontal: 24,
                      vertical: 10,
                    ),
                  ),
                  child: Text(
                    _isStreaming ? 'Streaming Live...' : 'Start WHIP Stream',
                    style: const TextStyle(fontSize: 14),
                  ),
                ),
                const Divider(height: 30),
                Align(
                  alignment: Alignment.centerLeft,
                  child: Text(
                    "WS Status: $_wsStatusText",
                    style: const TextStyle(
                      fontWeight: FontWeight.bold,
                      fontSize: 13,
                      color: Colors.blueGrey,
                    ),
                  ),
                ),
                const SizedBox(height: 6),
                const Align(
                  alignment: Alignment.centerLeft,
                  child: Text(
                    "Live Event Logs (Alerts & Status):",
                    style: TextStyle(fontSize: 12, color: Colors.grey),
                  ),
                ),
                const SizedBox(height: 6),
                Container(
                  height: 120,
                  decoration: BoxDecoration(
                    border: Border.all(color: Colors.grey.shade300),
                    borderRadius: BorderRadius.circular(8),
                  ),
                  child: ListView.builder(
                    itemCount: _eventLogs.length,
                    itemBuilder: (context, index) {
                      return ListTile(
                        dense: true,
                        title: Text(
                          _eventLogs[index],
                          style: const TextStyle(
                            fontFamily: 'monospace',
                            fontSize: 11,
                          ),
                        ),
                      );
                    },
                  ),
                ),
              ],
            ),
          ),
        ),
      ),
    );
  }
}
