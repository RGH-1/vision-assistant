import 'package:flutter/material.dart';
import 'package:flutter_webrtc/flutter_webrtc.dart';
import 'package:http/http.dart' as http;
import 'package:permission_handler/permission_handler.dart';

void main() {
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
  // Replace with your PC's actual local IPv4 address from ipconfig
  final String whipEndpoint = "http://192.168.8.107:8889/mystream/whip";

  RTCPeerConnection? _peerConnection;
  MediaStream? _localStream;
  final RTCVideoRenderer _localRenderer = RTCVideoRenderer();
  bool _isStreaming = false;

  @override
  void initState() {
    super.initState();
    _localRenderer.initialize();
  }

  @override
  void dispose() {
    _localRenderer.dispose();
    _localStream?.dispose();
    _peerConnection?.dispose();
    super.dispose();
  }

  Future<void> _startStreaming() async {
    try {
      // 1. Explicitly prompt the user for runtime camera and mic permissions
      Map<Permission, PermissionStatus> statuses = await [
        Permission.camera,
        Permission.microphone,
      ].request();

      if (statuses[Permission.camera] != PermissionStatus.granted ||
          statuses[Permission.microphone] != PermissionStatus.granted) {
        debugPrint("Camera or Microphone permission was denied.");
        return;
      }

      // 2. Get camera and microphone stream with flexible constraints for older hardware
      final Map<String, dynamic> mediaConstraints = {
        'audio': true,
        'video': {
          'width': {'exact': 640}, // 640x480 is the sweet spot for stability
          'height': {'exact': 480},
          'frameRate': {
            'exact': 20,
          }, // 20 fps prevents high CPU load while looking smooth
          'facingMode': 'environment',
        },
      };

      _localStream = await navigator.mediaDevices.getUserMedia(
        mediaConstraints,
      );
      _localRenderer.srcObject = _localStream;

      // 3. Create peer connection
      _peerConnection = await createPeerConnection({
        'iceServers': [
          {'urls': 'stun:stun.l.google.com:19302'},
        ],
      });

      for (var track in _localStream!.getTracks()) {
        _peerConnection!.addTrack(track, _localStream!);
      }

      // 4. Create SDP Offer
      RTCSessionDescription offer = await _peerConnection!.createOffer();
      await _peerConnection!.setLocalDescription(offer);

      await Future.delayed(const Duration(milliseconds: 500));

      RTCSessionDescription? localDesc = await _peerConnection!
          .getLocalDescription();

      // 5. Send WHIP POST request to MediaMTX
      final response = await http.post(
        Uri.parse(whipEndpoint),
        headers: {'Content-Type': 'application/sdp'},
        body: localDesc?.sdp ?? '',
      );

      if (response.statusCode == 201 || response.statusCode == 200) {
        RTCSessionDescription answer = RTCSessionDescription(
          response.body,
          'answer',
        );
        await _peerConnection!.setRemoteDescription(answer);

        setState(() {
          _isStreaming = true;
        });
        debugPrint("WHIP Stream Connected Successfully!");
      } else {
        debugPrint(
          "Failed to connect WHIP: ${response.statusCode} - ${response.body}",
        );
      }
    } catch (e) {
      debugPrint("Error starting stream: $e");
    }
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(
        backgroundColor: Theme.of(context).colorScheme.inversePrimary,
        title: Text(widget.title),
      ),
      body: Center(
        child: Column(
          mainAxisAlignment: MainAxisAlignment.center,
          children: [
            SizedBox(
              width: 300,
              height: 400,
              child: RTCVideoView(_localRenderer, mirror: false),
            ),
            const SizedBox(height: 20),
            ElevatedButton(
              onPressed: _isStreaming ? null : _startStreaming,
              child: Text(
                _isStreaming ? 'Streaming Live...' : 'Start WHIP Stream',
              ),
            ),
          ],
        ),
      ),
    );
  }
}
