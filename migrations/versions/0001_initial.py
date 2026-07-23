"""Initial ShakePi viewer schema.

Revision ID: 0001_initial
Revises:
Create Date: 2026-07-15
"""

from alembic import op
from shakepi.models import Base


revision = "0001_initial"
down_revision = None
branch_labels = None
depends_on = None


def upgrade() -> None:
    Base.metadata.create_all(op.get_bind())


def downgrade() -> None:
    Base.metadata.drop_all(op.get_bind())
