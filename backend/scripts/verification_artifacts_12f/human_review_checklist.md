# Part 12f — human review checklist

Every automated check in this run passed (see `verification_report.json`
in this same directory): the API returns the exact JSON shape each page's
TypeScript types expect, and every clip/reel file it points at is a real,
ffprobed-playable video. What's left is the part nothing in this sandbox
can do — actually load these pages in a browser and look at them.

Open these with the backend and frontend dev servers both still running:

## 1. Matches list — http://localhost:3000/matches
- [ ] All three matches appear, most-recently-played first.
- [ ] "Court 1 — Ready" shows a **Done** status badge.
- [ ] "Court 2 — Still Processing" shows a **Pending** or **Queued** badge.
- [ ] "Court 3 — Failed" shows a **Failed** status badge (in the danger color).

## 2. Match A (ready) — http://localhost:3000/matches/873c4c0a-830a-48c5-965d-01d9a024225a
- [ ] Statistics section shows 4 stat cards (Total Points, Avg Rally
      Length, Longest Rally, Errors) with real numbers, plus the
      player-stats-pending note underneath.
- [ ] Reel section shows a **Ready** badge and a clip count, and the
      `<video>` element actually plays when you hit play — real footage
      with a burned-in label in the corner of the source clips.
- [ ] Highlights section shows 4 cards, each with its own playable clip,
      correct highlight-type label, time range, and importance bar.
- [ ] Every clip plays without a broken-video icon or a stalled spinner.

## 3. Match B (still processing) — http://localhost:3000/matches/36ae066a-8bfb-44f5-8495-eee180f0339b
- [ ] Statistics section reads "No match statistics yet — these are
      computed once analysis finishes." — not an error, not a blank gap.
- [ ] Reel section reads "The reel will appear here once processing
      finishes." — not "Failed", not a broken player.
- [ ] Highlights section reads "Highlights will appear here once analysis
      finds them."
- [ ] No console errors in the browser devtools on this page.

## 4. Match C (failed) — http://localhost:3000/matches/13f1adc5-b229-4863-95e7-ad7d82e5f014
- [ ] Status badge reads **Failed**.
- [ ] Reel section reads "Processing failed before a reel could be
      generated." (not the "still processing" copy — this is the one
      case that tells those two apart).
- [ ] Highlights section reads "No highlights were tagged for this
      match." (Part 12c's own copy for a terminal video_status with zero
      highlights — same message a genuinely-done-but-empty match would
      show, which is an existing, pre-12f wording choice worth knowing
      about rather than being surprised by here.)

## 5. General
- [ ] No hydration warnings in the browser console on any of the four
      pages above.
- [ ] Network tab shows each page's requests going to the expected
      `/matches...` API routes and returning 200.
