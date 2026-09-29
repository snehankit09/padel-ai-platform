"""add highlights.player_label

Revision ID: 6b150fcaf37c
Revises: 7bc0ccd22d28
Create Date: 2026-09-27 00:00:00.000000

Highlights Improvement Roadmap, Tier 3c: same small mechanical pattern as
Tier 1a's thumbnail_file_path and Tier 3b's player_track_id -- one
nullable column, no backfill needed since every existing Highlight row
predates the resolution step that fills this in and simply gets NULL
here. The resolved "Player A"-"Player D"/"Near Team"/"Far Team" label
(see app/models/highlight.py's own column comment for the full
reasoning and app/services/team_side_label_resolution_stage.py for who
writes it and when).
"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = '6b150fcaf37c'
down_revision: Union[str, Sequence[str], None] = '7bc0ccd22d28'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    """Upgrade schema."""
    op.add_column('highlights', sa.Column('player_label', sa.String(length=64), nullable=True))


def downgrade() -> None:
    """Downgrade schema."""
    op.drop_column('highlights', 'player_label')
