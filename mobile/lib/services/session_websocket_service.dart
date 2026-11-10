import 'dart:async';
import 'dart:convert';

import 'package:flutter/foundation.dart';
import 'package:http/http.dart' as http;
import 'package:web_socket_channel/web_socket_channel.dart';
import 'package:web_socket_channel/status.dart' as status;

class SessionWebSocketService {
  static const String baseUrl = "http://192.168.8.111:8000";
  static const String wsUrl = "ws://192.168.8.111:8000/v1/ws/phone";

  String? _sessionId;
  WebSocketChannel? _channel;
  Timer? _pingTimer;

  // Reconnection backoff tracking
  int _reconnectAttempts = 0;
  bool _isManuallyClosed = false;

  // Callbacks for UI updates
  Function(Map<String, dynamic>)? onEventReceived;

  /// Step 1: POST /v1/sessions with exact Swagger schema keys (app_lang, consent_at, consent_ver)
  Future<bool> initializeSession({
    required String appLang,
    required String consentAt,
    required String consentVer,
  }) async {
    final payload = {
      'app_lang': appLang,
      'consent_at': consentAt,
      'consent_ver': consentVer,
    };

    if (kDebugMode) {
      debugPrint("SENDING TO BACKEND: ${jsonEncode(payload)}");
    }

    try {
      final response = await http.post(
        Uri.parse('$baseUrl/v1/sessions'),
        headers: {'Content-Type': 'application/json'},
        body: jsonEncode(payload),
      );

      if (response.statusCode == 200 || response.statusCode == 201) {
        final data = jsonDecode(response.body);
        _sessionId = data['session_id'] ?? data['id'];
        if (kDebugMode) {
          debugPrint("SESSION CREATED SUCCESSFULLY: $_sessionId");
        }
        return true;
      } else {
        if (kDebugMode) {
          debugPrint(
            "REJECTED BY SERVER (${response.statusCode}): ${response.body}",
          );
        }
        return false;
      }
    } catch (e) {
      if (kDebugMode) {
        debugPrint("SESSION EXCEPTION: $e");
      }
      return false;
    }
  }

  /// Step 2 & 3: Open WebSocket using session_id query param, send hello, handle ping/pong & backoff
  void connectWebSocket() {
    if (_sessionId == null) return;

    _isManuallyClosed = false;
    final uri = Uri.parse('$wsUrl?session_id=$_sessionId');

    if (kDebugMode) {
      debugPrint("CONNECTING TO WS: $uri");
    }

    try {
      _channel = WebSocketChannel.connect(uri);

      // Listen to incoming messages
      _channel!.stream.listen(
        (message) {
          _reconnectAttempts = 0;
          _handleIncomingMessage(message);
        },
        onDone: () {
          if (kDebugMode) {
            debugPrint("WS CONNECTION CLOSED (onDone)");
          }
          _handleDisconnection();
        },
        onError: (error) {
          if (kDebugMode) {
            debugPrint("WS ERROR: $error");
          }
          _handleDisconnection();
        },
      );

      // Send initial 'hello' message upon connecting
      _sendHello();

      // Start 15-second ping/pong heartbeat timer
      _startHeartbeat();
    } catch (e) {
      if (kDebugMode) {
        debugPrint("WS CONNECTION EXCEPTION: $e");
      }
      _handleDisconnection();
    }
  }

  void _sendHello() {
    final helloPayload = jsonEncode({
      "type": "hello",
      "session_id": _sessionId,
      "client": "mobile_app",
    });
    _channel?.sink.add(helloPayload);
  }

  void _startHeartbeat() {
    _pingTimer?.cancel();
    _pingTimer = Timer.periodic(const Duration(seconds: 15), (timer) {
      final pingPayload = jsonEncode({
        "type": "ping",
        "timestamp": DateTime.now().toUtc().toIso8601String(),
      });
      _channel?.sink.add(pingPayload);
    });
  }

  void _handleIncomingMessage(dynamic message) {
    try {
      final decoded = jsonDecode(message);
      if (kDebugMode) {
        debugPrint("RECEIVED WS EVENT: ${decoded['type']} -> $decoded");
      }
      if (onEventReceived != null) {
        onEventReceived!(decoded);
      }
    } catch (e) {
      if (kDebugMode) {
        debugPrint("WS Message parsing error: $e");
      }
    }
  }

  void _handleDisconnection() {
    _pingTimer?.cancel();
    if (_isManuallyClosed) return;

    // Calculate exponential backoff delays: 1s, 2s, 4s, 8s
    final delays = [1, 2, 4, 8];
    int delaySeconds =
        delays[_reconnectAttempts < delays.length
            ? _reconnectAttempts
            : delays.length - 1];

    if (kDebugMode) {
      debugPrint(
        "Reconnecting in ${delaySeconds}s (Attempt ${_reconnectAttempts + 1})...",
      );
    }

    _reconnectAttempts++;

    Future.delayed(Duration(seconds: delaySeconds), () {
      if (!_isManuallyClosed) {
        connectWebSocket();
      }
    });
  }

  /// Clean disconnect when app closes
  void disconnect() {
    _isManuallyClosed = true;
    _pingTimer?.cancel();
    _channel?.sink.close(status.goingAway);
  }
}
