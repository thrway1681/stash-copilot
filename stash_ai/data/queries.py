"""
GraphQL query definitions for Stash API.

DEPRECATED: This module is no longer used. All data access has been migrated
to direct SQLite queries for accuracy (matching Stash UI displays).

See:
- stash_ai/tools/database.py for query tools
- stash_ai/data/aggregators.py for aggregation logic

This file is kept for reference only and may be removed in a future version.
"""

# Get basic library statistics
STATS_QUERY = """
query Stats {
    stats {
        scene_count
        scenes_size
        scenes_duration
        image_count
        images_size
        gallery_count
        performer_count
        studio_count
        movie_count
        tag_count
        total_o_count
        total_play_count
        total_play_duration
    }
}
"""

# Get scenes with viewing statistics for aggregation
SCENES_WITH_STATS_QUERY = """
query FindScenes($filter: FindFilterType, $scene_filter: SceneFilterType) {
    findScenes(filter: $filter, scene_filter: $scene_filter) {
        count
        scenes {
            id
            title
            rating100
            play_count
            play_duration
            o_counter
            organized
            files {
                duration
            }
            performers {
                id
                name
            }
            tags {
                id
                name
            }
            studio {
                id
                name
            }
        }
    }
}
"""

# Get performers with scene counts (for top performers)
PERFORMERS_QUERY = """
query FindPerformers($filter: FindFilterType) {
    findPerformers(filter: $filter) {
        count
        performers {
            id
            name
            scene_count
            image_count
        }
    }
}
"""

# Get tags with scene counts
TAGS_QUERY = """
query FindTags($filter: FindFilterType) {
    findTags(filter: $filter) {
        count
        tags {
            id
            name
            scene_count
        }
    }
}
"""

# Get studios with scene counts
STUDIOS_QUERY = """
query FindStudios($filter: FindFilterType) {
    findStudios(filter: $filter) {
        count
        studios {
            id
            name
            scene_count
        }
    }
}
"""

# Get a single scene with full details
SCENE_DETAIL_QUERY = """
query FindScene($id: ID!) {
    findScene(id: $id) {
        id
        title
        details
        date
        rating100
        play_count
        play_duration
        o_counter
        organized
        created_at
        updated_at
        files {
            path
            size
            duration
            video_codec
            audio_codec
            width
            height
            framerate
            bitrate
        }
        performers {
            id
            name
            image_path
        }
        tags {
            id
            name
        }
        studio {
            id
            name
        }
        movies {
            movie {
                id
                name
            }
        }
        markers {
            id
            title
            seconds
        }
    }
}
"""

# Get scenes that have been watched (play_count > 0)
WATCHED_SCENES_QUERY = """
query FindWatchedScenes($filter: FindFilterType) {
    findScenes(
        filter: $filter,
        scene_filter: { play_count: { modifier: GREATER_THAN, value: 0 } }
    ) {
        count
        scenes {
            id
            title
            rating100
            play_count
            play_duration
            o_counter
            files {
                duration
            }
            performers {
                id
                name
            }
            tags {
                id
                name
            }
            studio {
                id
                name
            }
        }
    }
}
"""

# Get scenes by rating range
RATED_SCENES_QUERY = """
query FindRatedScenes($filter: FindFilterType, $min_rating: Int!) {
    findScenes(
        filter: $filter,
        scene_filter: { rating100: { modifier: GREATER_THAN, value: $min_rating } }
    ) {
        count
        scenes {
            id
            title
            rating100
            play_count
        }
    }
}
"""
