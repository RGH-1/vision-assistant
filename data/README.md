# data/ — not in git

# Data Directory

## Videos (`data/videos/`)
Place local test videos here for streaming development and verification.

* **File:** `classroom.mp4`
* **Source:** [Intel IoT DevKit Sample Videos Repository](https://github.com/intel-iot-devkit/sample-videos)
* **Usage:** Picked up by `infra/scripts/push_test_video.ps1` to loop and push an RTSP stream to MediaMTX.


Runtime files only: `frames/`, `thumbs/`, `originals/`, `videos/` (test walks), `models/` (YOLOE, YuNet…).
Everything here except this file is git-ignored. Frames are deleted after 30 minutes (design §7.5).
