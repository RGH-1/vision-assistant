import 'package:flutter/material.dart';
import 'package:flutter/services.dart';
import 'package:flutter_webrtc/flutter_webrtc.dart';
import 'package:http/http.dart' as http;
import 'package:permission_handler/permission_handler.dart';

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
      title: 'Vision Assistant WHIP',
      theme: ThemeData(
        colorScheme: ColorScheme.fromSeed(seedColor: Colors.deepPurple),
        useMaterial3: true,
      ),
      home: const MyHomePage(title: 'WHIP Camera Streamer'),
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
  // Routes through FastAPI proxy on port 8000
  final String whipEndpoint = "http://192.168.8.111:8889/mystream/whip";

  RTCPeerConnection? _peerConnection;
  MediaStream? _localStream;
  final RTCVideoRenderer _localRenderer = RTCVideoRenderer();
  bool _isStreaming = false;
  String _statusMessage = "Ready to stream";

  @override
  void initState() {
    super.initState();
    _initializeCamera();
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

  @override
  void dispose() {
    _localRenderer.dispose();
    _localStream?.dispose();
    _peerConnection?.dispose();
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
                  width: 260,
                  height: 320,
                  child: ClipRRect(
                    borderRadius: BorderRadius.circular(12),
                    child: RTCVideoView(_localRenderer, mirror: false),
                  ),
                ),
                const SizedBox(height: 12),
                Text(
                  _statusMessage,
                  textAlign: TextAlign.center,
                  style: TextStyle(
                    fontSize: 14,
                    fontWeight: FontWeight.w500,
                    color: _isStreaming ? Colors.green : Colors.black87,
                  ),
                ),
                const SizedBox(height: 16),
                ElevatedButton(
                  onPressed: _isStreaming ? null : _startStreaming,
                  style: ElevatedButton.styleFrom(
                    padding: const EdgeInsets.symmetric(
                      horizontal: 32,
                      vertical: 12,
                    ),
                  ),
                  child: Text(
                    _isStreaming ? 'Streaming Live...' : 'Start WHIP Stream',
                    style: const TextStyle(fontSize: 16),
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
