"""
Part 2 sanity test — not a permanent fixture, just proof the models work
end to end against a real database before we build on them.

Run with: python -m scripts.sanity_check_models
(from the backend/ directory, with DATABASE_URL etc. set)
"""

import asyncio
from datetime import datetime, timezone

from app.core.database import AsyncSessionLocal
from app.models import User, Player, Match, MatchPlayer, Video, Highlight, Statistic
from app.models.enums import UserRole, VideoStatus, HighlightType, StatType


async def main():
    async with AsyncSessionLocal() as session:
        # 1. Create a club admin user (the uploader)
        admin = User(
            email="admin@padelclub.example",
            hashed_password="not-a-real-hash",
            full_name="Club Admin",
            role=UserRole.CLUB_ADMIN,
        )
        session.add(admin)
        await session.flush()  # assigns admin.id without committing yet

        # 2. Create two players
        player_a = Player(full_name="Alex Rivera")
        player_b = Player(full_name="Sam Okafor")
        session.add_all([player_a, player_b])
        await session.flush()

        # 3. Create a match, linking the uploader
        match = Match(
            played_at=datetime.now(timezone.utc),
            venue="Central Padel Club, Court 2",
            format="doubles",
            uploaded_by=admin.id,
        )
        session.add(match)
        await session.flush()

        # 3b. Assign players to teams via the MatchPlayer association object —
        # this is the pattern for adding rows to an association table that
        # carries extra data (team_number), instead of `match.players.append(...)`.
        session.add_all([
            MatchPlayer(match=match, player=player_a, team_number=1),
            MatchPlayer(match=match, player=player_b, team_number=2),
        ])
        await session.flush()

        # 4. Attach a video to the match
        video = Video(
            match_id=match.id,
            file_path="uploads/match_001.mp4",
            original_filename="match_001.mp4",
            status=VideoStatus.DONE,
            duration_seconds=5400.0,
            resolution_width=1920,
            resolution_height=1080,
            fps=30.0,
        )
        session.add(video)

        # 5. Add a highlight and a statistic
        highlight = Highlight(
            match_id=match.id,
            event_type=HighlightType.POWERFUL_SMASH,
            start_time_seconds=612.0,
            end_time_seconds=618.5,
            importance_score=0.91,
        )
        stat = Statistic(
            match_id=match.id,
            player_id=player_a.id,
            stat_type=StatType.SERVE_PERCENTAGE,
            value=0.72,
        )
        session.add_all([highlight, stat])

        await session.commit()
        match_id = match.id

    # --- Now query it all back through relationships, in a fresh session ---
    async with AsyncSessionLocal() as session:
        from sqlalchemy import select
        from sqlalchemy.orm import selectinload

        result = await session.execute(
            select(Match)
            .options(
                selectinload(Match.match_players).selectinload(MatchPlayer.player),
                selectinload(Match.video),
                selectinload(Match.highlights),
                selectinload(Match.statistics),
                selectinload(Match.uploaded_by_user),
            )
            .where(Match.id == match_id)
        )
        fetched = result.scalar_one()

        print("Match:", fetched)
        print("  Uploaded by:", fetched.uploaded_by_user)
        print("  Players:", [(mp.player.full_name, f"team {mp.team_number}") for mp in fetched.match_players])
        print("  Video status:", fetched.video.status, "| duration:", fetched.video.duration_seconds, "s")
        print("  Highlights:", [(h.event_type.value, h.importance_score) for h in fetched.highlights])
        print("  Statistics:", [(s.stat_type.value, s.value) for s in fetched.statistics])

    print("\nOK — models, relationships, and migration all verified against real Postgres.")


if __name__ == "__main__":
    asyncio.run(main())
