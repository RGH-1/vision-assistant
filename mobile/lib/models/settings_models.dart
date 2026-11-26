import 'package:shared_preferences/shared_preferences.dart';

class SettingsModel {
  static const String defaultServerIp = '192.168.8.111';

  String serverIp;
  String language;
  double speechRate;
  bool alertsEnabled;

  SettingsModel({
    this.serverIp = defaultServerIp,
    this.language = 'en-US',
    this.speechRate = 0.5,
    this.alertsEnabled = true,
  });

  static Future<SettingsModel> load() async {
    final prefs = await SharedPreferences.getInstance();
    return SettingsModel(
      serverIp: prefs.getString('server_ip') ?? defaultServerIp,
      language: prefs.getString('language') ?? 'en-US',
      speechRate: prefs.getDouble('speech_rate') ?? 0.5,
      alertsEnabled: prefs.getBool('alerts_enabled') ?? true,
    );
  }

  Future<void> save() async {
    final prefs = await SharedPreferences.getInstance();
    await prefs.setString('server_ip', serverIp);
    await prefs.setString('language', language);
    await prefs.setDouble('speech_rate', speechRate);
    await prefs.setBool('alerts_enabled', alertsEnabled);
  }
}
