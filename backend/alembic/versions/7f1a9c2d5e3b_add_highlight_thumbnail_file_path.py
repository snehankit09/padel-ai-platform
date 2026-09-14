"""add highlights.thumbnail_file_path

Revision ID: 7f1a9c2d5e3b
Revises: 4c3e2398e557
Create Date: 2026-09-13 00:00:00.000000

Highlights Improvement Roadmap, Tier 1a: a per-clip poster/thumbnail
image, same lifecycle as the existing clip_file_path column — NULL until
Part 8b (app/services/clip_extraction_stage.py) successfully extracts a
clip and its accompanying thumbnail frame.
"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = '7f1a9c2d5e3b'
down_revision: Union[str, Sequence[str], None] = '4c3e2398e557'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    """Upgrade schema."""
    op.add_column('highlights', sa.Column('thumbnail_file_path', sa.String(length=1024), nullable=True))


def downgrade() -> None:
    """Downgrade schema."""
    op.drop_column('highlights', 'thumbnail_file_path')
