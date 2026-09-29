"""add highlights.player_track_id

Revision ID: 7bc0ccd22d28
Revises: 7f1a9c2d5e3b
Create Date: 2026-09-27 00:00:00.000000

Highlights Improvement Roadmap, Tier 3b: same small mechanical pattern as
Tier 1a's thumbnail_file_path column (7f1a9c2d5e3b) -- one nullable
column, no backfill needed since every existing Highlight row predates
Tier 3a's tagging logic and simply gets NULL here, same as every
pre-Tier-1a row got a NULL thumbnail_file_path. A raw ByteTrack track_id,
not a Player.id -- see app/models/highlight.py's own column comment for
why this isn't a ForeignKey.
"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = '7bc0ccd22d28'
down_revision: Union[str, Sequence[str], None] = '7f1a9c2d5e3b'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    """Upgrade schema."""
    op.add_column('highlights', sa.Column('player_track_id', sa.Integer(), nullable=True))


def downgrade() -> None:
    """Downgrade schema."""
    op.drop_column('highlights', 'player_track_id')
