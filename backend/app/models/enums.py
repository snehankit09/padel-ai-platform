"""
Shared enums.

Using real Python enums (mapped to Postgres ENUM types) instead of plain
strings means an invalid value like "procesing" (typo) is rejected by the
database itself, not discovered three joins later in a dashboard query.
"""

import enum


class UserRole(str, enum.Enum):
    PLAYER = "player"
    COACH = "coach"
    CLUB_ADMIN = "club_admin"
    ORGANIZER = "organizer"
    BROADCASTER = "broadcaster"


class VideoStatus(str, enum.Enum):
    """Tracks a video through the pipeline stages in PRD Section 9."""
    PENDING = "pending"           # uploaded, not yet queued
    QUEUED = "queued"             # queued for processing
    PROCESSING = "processing"     # pipeline running
    DONE = "done"                 # all stages complete
    FAILED = "failed"             # a stage failed after retries


class HighlightType(str, enum.Enum):
    """
    From PRD Module 3 — Highlight Detection features.

    Part 7e (ml/pipeline/highlight_tagging.py) rule-tags LONG_RALLY,
    FAST_EXCHANGE, POWERFUL_SMASH, and SPECTACULAR_SAVE from Part 7a/7c's
    output today. WINNING_SHOT, MATCH_POINT, and BREAK_POINT are real
    enum values a future stage can still populate, but nothing in this
    codebase emits them yet — see that module's own docstring for exactly
    which signal each one is still missing (a winner/unforced-error split
    for WINNING_SHOT; live game/set/match score state for the other two).
    """
    LONG_RALLY = "long_rally"
    WINNING_SHOT = "winning_shot"
    SPECTACULAR_SAVE = "spectacular_save"
    MATCH_POINT = "match_point"
    BREAK_POINT = "break_point"
    FAST_EXCHANGE = "fast_exchange"
    POWERFUL_SMASH = "powerful_smash"


class StatType(str, enum.Enum):
    """From PRD Module 5 — Match Statistics features."""
    TOTAL_POINTS = "total_points"
    WINNERS = "winners"
    ERRORS = "errors"
    SERVE_PERCENTAGE = "serve_percentage"
    RALLY_LENGTH_AVG = "rally_length_avg"
    NET_SUCCESS_RATE = "net_success_rate"
    SMASH_SUCCESS_RATE = "smash_success_rate"
    DISTANCE_COVERED = "distance_covered"
    MOVEMENT_SPEED_AVG = "movement_speed_avg"
    LONGEST_RALLY = "longest_rally"
    REACTION_TIME_AVG = "reaction_time_avg"
    MOMENTUM_POSSESSION = "momentum_possession"


class PredictionType(str, enum.Enum):
    """From PRD Module 6 — AI Predictions features."""
    WIN_PROBABILITY = "win_probability"
    MOMENTUM_SHIFT = "momentum_shift"
    FATIGUE = "fatigue"
    STRONGEST_PLAYER = "strongest_player"
    WEAKEST_PLAYER = "weakest_player"
    MOST_CONSISTENT_PLAYER = "most_consistent_player"
    SHOT_SUCCESS_PROBABILITY = "shot_success_probability"
    PERFORMANCE_TREND = "performance_trend"


class ReelStatus(str, enum.Enum):
    PENDING = "pending"
    GENERATING = "generating"
    READY = "ready"
    FAILED = "failed"
