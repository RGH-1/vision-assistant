# contracts/ — All

Frozen API and event contracts (design §8). **Frozen on Nov 5**; changes need all three members to agree.

- `openapi.yaml` — REST routes and `AskResponse` (§8.1)
- `events.schema.json` — WebSocket messages (§8.2)

Agents build against these files and never edit them.
