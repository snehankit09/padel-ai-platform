"""add videos.current_stage

Revision ID: 4c3e2398e557
Revises: 51bdcc8a1557
Create Date: 2026-08-06 00:00:00.000000

Part 4c: pipeline stages need somewhere to record which stage a video is
currently on while status == PROCESSING, so GET /videos/{id}/status can
show real progress instead of a status stuck on "processing" for the
entire pipeline run.
"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = '4c3e2398e557'
down_revision: Union[str, Sequence[str], None] = '51bdcc8a1557'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    """Upgrade schema."""
    op.add_column('videos', sa.Column('current_stage', sa.String(length=50), nullable=True))


def downgrade() -> None:
    """Downgrade schema."""
    op.drop_column('videos', 'current_stage')
