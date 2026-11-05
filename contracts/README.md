# contracts/ — All

API and event contracts (design §8). **Status: DRAFT, not frozen yet.** Freeze on Nov 5, after M1, M2 and M3 agree on the points below; later changes need all three members to agree.

- `openapi.yaml` — REST routes, `AskResponse`, error format (§8.1)
- `events.schema.json` — WebSocket messages (§8.2), plus the types shared with `openapi.yaml`
- `events.examples.json` — one example of each message, for use as test fixtures

Agents build against these files and never edit them.

## To agree on the call

Each point is where the draft adds to or decides something §8 leaves open. Once agreed, §8 gets updated to match and this section is deleted.

| # | Decision in the draft | Why | Affects |
|---|---|---|---|
| 1 | `status` has a new `session_id` field; the newest session is the active one, and creating a session ends the previous one | §8 gives the dashboard no way to learn the session ID its REST calls need | M1, M2, M3 |
| 2 | Answers from the live buffer have `frame_id: null` and `thumb_url: null` in `AskResponse.evidence`, and no `question_evidence` row | Live frames aren't saved, but the schema requires a saved frame for each evidence row. So the dashboard shows no thumbnail for "now" questions | M1, M3 |
| 3 | `status` is the same full message for phone and dashboard, with no "short form" | One schema; the phone ignores what it doesn't use | M1 |
| 4 | Messages are flat: `{"type": "alert", "alert_id": 41, …}`, with no nested `payload` | §8.2 only says "JSON with a `type` field" | M1, M3 |
| 5 | New error codes `BAD_REQUEST`, `NOT_FOUND`, `INTERNAL`, and an HTTP status for each code | Bad input and crashes must also return `speak`, or the phone goes silent | M1, M3 |
| 6 | `alert` has a new `thumb_url`, and `frame_id` can be null | The Alerts panel shows thumbnails; alert rules also run on frames that aren't saved | M1, M3 |
| 7 | `dhash` is left out of `FrameDetail` | A 64-bit integer loses precision in JavaScript, and the dashboard doesn't need it | M1 |

## Also needs a decision (not a contract change)

**Live-buffer frames must be blurred before Gemini.** In §6.3.1 frames enter the live buffer *before* the face-blur step, but §9 says only blurred frames are sent. So `LIVE_SCENE` and `READ_TEXT_NOW` would send unblurred faces to Google unless M2 and M3 agree where the blur runs: when the frame enters the buffer, or in `/v1/ask` before the Gemini call.
