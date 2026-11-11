# VA-56 hybrid retrieval: EXPLAIN ANALYZE

Generated with 1800 frames in one 30-minute session.

The planner uses the `(session_id, captured_at)` B-tree index, then sorts the small filtered window; it does not use the global HNSW index.

```text
Limit  (cost=7.19..7.20 rows=1 width=116) (actual time=17.541..17.549 rows=30 loops=1)
  Buffers: shared hit=10868
  CTE win
    ->  Index Scan using frames_session_time on frames f  (cost=0.27..6.04 rows=1 width=309) (actual time=0.014..0.608 rows=1800 loops=1)
          Index Cond: ((session_id = 'b1e74bda-1e86-41b1-b9fc-3090e25f12de'::uuid) AND (captured_at >= '2026-11-11 23:01:38+00'::timestamp with time zone) AND (captured_at <= '2026-11-11 23:31:38+00'::timestamp with time zone))
          Buffers: shared hit=64
  ->  Sort  (cost=1.15..1.16 rows=1 width=116) (actual time=17.540..17.544 rows=30 loops=1)
        Sort Key: (((('1'::double precision - (win.clip_emb <=> '<query vector>'::vector)) + COALESCE((max(similarity(text_norm, ''::text))), '0'::real)) + COALESCE((max(d.confidence)), '0'::real))) DESC
        Sort Method: top-N heapsort  Memory: 28kB
        Buffers: shared hit=10868
        ->  Nested Loop Left Join  (cost=1.04..1.14 rows=1 width=116) (actual time=0.070..16.556 rows=1800 loops=1)
              Join Filter: (d.frame_id = win.id)
              Buffers: shared hit=10865
              ->  Nested Loop Left Join  (cost=0.00..0.05 rows=1 width=96) (actual time=0.020..2.814 rows=1800 loops=1)
                    Join Filter: (frame_id = win.id)
                    Buffers: shared hit=64
                    ->  CTE Scan on win  (cost=0.00..0.02 rows=1 width=92) (actual time=0.016..1.972 rows=1800 loops=1)
                          Buffers: shared hit=64
                    ->  HashAggregate  (cost=0.00..0.01 rows=1 width=12) (actual time=0.000..0.000 rows=0 loops=1800)
                          Group Key: frame_id
                          Batches: 1  Memory Usage: 24kB
                          ->  Result  (cost=0.00..0.00 rows=0 width=40) (actual time=0.001..0.001 rows=0 loops=1)
                                One-Time Filter: false
              ->  HashAggregate  (cost=1.04..1.05 rows=1 width=12) (actual time=0.000..0.000 rows=0 loops=1800)
                    Group Key: d.frame_id
                    Batches: 1  Memory Usage: 24kB
                    Buffers: shared hit=1
                    ->  Nested Loop  (cost=0.00..1.03 rows=1 width=12) (actual time=0.005..0.007 rows=0 loops=1)
                          Join Filter: (d.frame_id = win_1.id)
                          Buffers: shared hit=1
                          ->  Seq Scan on detections d  (cost=0.00..1.00 rows=1 width=12) (actual time=0.005..0.006 rows=0 loops=1)
                                Filter: (label = ANY ('{}'::text[]))
                                Buffers: shared hit=1
                          ->  CTE Scan on win win_1  (cost=0.00..0.02 rows=1 width=8) (never executed)
Planning:
  Buffers: shared hit=278
Planning Time: 1.429 ms
Execution Time: 18.051 ms
```
