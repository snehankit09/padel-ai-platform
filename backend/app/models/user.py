"""
User — PRD Section 11: "Accounts for players, coaches, clubs, and
organizers, with role-based access."

Note: User (an account, used to log in) is distinct from Player (a person
who appears in match footage and has stats). A coach User might never be
a Player. A Player might not even have a User account (e.g. an opponent
who never signs up) — that's why Player is its own table, linked loosely
rather than merged into User. We'll wire that relationship in player.py.
"""

import uuid
from datetime import datetime

from sqlalchemy import String, Enum as SAEnum
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.core.database import Base
from app.models.mixins import TimestampedBase
from app.models.enums import UserRole


class User(Base, TimestampedBase):
    __tablename__ = "users"

    email: Mapped[str] = mapped_column(String(255), unique=True, index=True, nullable=False)
    hashed_password: Mapped[str] = mapped_column(String(255), nullable=False)
    full_name: Mapped[str] = mapped_column(String(255), nullable=False)
    role: Mapped[UserRole] = mapped_column(SAEnum(UserRole, name="user_role"), nullable=False)

    # A user (club admin, coach) can upload many matches
    matches: Mapped[list["Match"]] = relationship(back_populates="uploaded_by_user")

    def __repr__(self) -> str:
        return f"<User {self.email} ({self.role})>"
