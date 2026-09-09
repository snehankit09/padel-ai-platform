"""
Importing every model here is what makes them register with
Base.metadata, which Alembic's autogenerate and create_all() both rely on.
"""

from app.models.user import User
from app.models.player import Player
from app.models.match import Match
from app.models.match_player import MatchPlayer
from app.models.video import Video
from app.models.highlight import Highlight
from app.models.reel import Reel
from app.models.reel_highlight import ReelHighlight
from app.models.statistic import Statistic
from app.models.prediction import Prediction
from app.models.report import Report

__all__ = [
    "User",
    "Player",
    "Match",
    "MatchPlayer",
    "Video",
    "Highlight",
    "Reel",
    "ReelHighlight",
    "Statistic",
    "Prediction",
    "Report",
]
