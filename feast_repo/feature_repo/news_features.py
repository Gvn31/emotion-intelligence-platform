"""
Feast Feature Definitions

Defines:
1. Entity
2. PostgreSQL Source
3. Feature View
"""

from datetime import timedelta

from feast import Entity
from feast import FeatureView
from feast import Field

from feast.types import Int64
from feast.types import Float32

from feast.infra.offline_stores.contrib.postgres_offline_store.postgres_source import (
    PostgreSQLSource
)

# -----------------------------
# Entity
# -----------------------------

news_article = Entity(
    name="raw_article_id",
    join_keys=["raw_article_id"]
)

# -----------------------------
# PostgreSQL Source
# -----------------------------

news_source = PostgreSQLSource(
    name="news_feature_source",

    query="""
    SELECT
        raw_article_id,
        text_length,
        word_count,
        unique_word_count,
        avg_word_length,
        sentiment_score,
        created_at
    FROM feature_store
    """,

    timestamp_field="created_at"
)

# -----------------------------
# Feature View
# -----------------------------

news_feature_view = FeatureView(
    name="news_features",

    entities=[news_article],

    ttl=timedelta(days=30),

    schema=[
        Field(
            name="text_length",
            dtype=Int64
        ),

        Field(
            name="word_count",
            dtype=Int64
        ),

        Field(
            name="unique_word_count",
            dtype=Int64
        ),

        Field(
            name="avg_word_length",
            dtype=Float32
        ),

        Field(
            name="sentiment_score",
            dtype=Float32
        )
    ],

    source=news_source,

    online=True
)