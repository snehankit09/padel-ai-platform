"""
Placeholder uploader.

Match.uploaded_by is a required FK to users.id (Part 2). We don't have
auth yet (Part 3 is upload only), so every upload needs *some* user to
attribute the match to. Decision from the build log: stub a placeholder
user rather than build auth first, so this module is the single place
that decision lives — swapping to a real `current_user` dependency later
means changing the one call site in the upload route, not hunting through
the codebase for hardcoded UUIDs.

Idempotent by design: looks up a fixed, well-known email first and only
creates the row the first time the app runs against a fresh database.
"""

from __future__ import annotations

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.models.enums import UserRole
from app.models.user import User

PLACEHOLDER_EMAIL = "placeholder-uploader@system.local"


async def get_or_create_placeholder_user(db: AsyncSession) -> User:
    result = await db.execute(select(User).where(User.email == PLACEHOLDER_EMAIL))
    user = result.scalar_one_or_none()
    if user is not None:
        return user

    user = User(
        email=PLACEHOLDER_EMAIL,
        # Not a real login — auth doesn't exist yet, so this hash can never
        # actually be used to authenticate. Replaced once Part 3's
        # placeholder is swapped for real auth.
        hashed_password="unusable-placeholder",
        full_name="Placeholder Uploader",
        role=UserRole.CLUB_ADMIN,
    )
    db.add(user)
    await db.flush()  # populate user.id without committing the outer transaction
    return user
