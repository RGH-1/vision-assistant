class AskEvidence {
  final int? frameId;
  final double secondsAgo;
  final String? thumbUrl;
  final double score;

  const AskEvidence({
    required this.frameId,
    required this.secondsAgo,
    required this.thumbUrl,
    required this.score,
  });

  factory AskEvidence.fromJson(Map<String, dynamic> json) {
    return AskEvidence(
      frameId: (json['frame_id'] as num?)?.toInt(),
      secondsAgo: (json['seconds_ago'] as num?)?.toDouble() ?? 0.0,
      thumbUrl: json['thumb_url'] as String?,
      score: (json['score'] as num?)?.toDouble() ?? 0.0,
    );
  }
}

class AskResponse {
  final int questionId;
  final String transcript;
  final String lang;
  final String intent;
  final String answer;
  final String confidence;
  final bool abstained;
  final List<AskEvidence> evidence;
  final Map<String, double> timingsMs;

  const AskResponse({
    required this.questionId,
    required this.transcript,
    required this.lang,
    required this.intent,
    required this.answer,
    required this.confidence,
    required this.abstained,
    required this.evidence,
    required this.timingsMs,
  });

  factory AskResponse.fromJson(Map<String, dynamic> json) {
    final rawEvidence = json['evidence'] as List<dynamic>? ?? const [];
    final rawTimings =
        json['timings_ms'] as Map<String, dynamic>? ?? const <String, dynamic>{};

    return AskResponse(
      questionId: (json['question_id'] as num).toInt(),
      transcript: json['transcript'] as String? ?? '',
      lang: json['lang'] as String? ?? 'en',
      intent: json['intent'] as String? ?? 'LIVE_SCENE',
      answer: json['answer'] as String? ?? '',
      confidence: json['confidence'] as String? ?? 'low',
      abstained: json['abstained'] as bool? ?? false,
      evidence: rawEvidence
          .whereType<Map<String, dynamic>>()
          .map(AskEvidence.fromJson)
          .toList(),
      timingsMs: rawTimings.map(
        (key, value) => MapEntry(key, (value as num).toDouble()),
      ),
    );
  }
}

class AskError {
  final String code;
  final String message;
  final String speak;

  const AskError({
    required this.code,
    required this.message,
    required this.speak,
  });

  factory AskError.fromJson(Map<String, dynamic> json) {
    final inner = json['error'] is Map<String, dynamic>
        ? json['error'] as Map<String, dynamic>
        : json;
    return AskError(
      code: inner['code'] as String? ?? 'INTERNAL',
      message: inner['message'] as String? ?? 'Unexpected error.',
      speak:
          inner['speak'] as String? ??
          'Sorry, something went wrong. Please try again.',
    );
  }
}

class AskResult {
  final AskResponse? response;
  final AskError? error;

  const AskResult.success(this.response) : error = null;
  const AskResult.failure(this.error) : response = null;

  bool get isSuccess => response != null;

  String get spokenText =>
      response?.answer ??
      error?.speak ??
      'Sorry, something went wrong. Please try again.';
}

