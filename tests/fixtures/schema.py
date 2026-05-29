"""SQLite schema creation for mock Stash database."""

import sqlite3


def create_mock_schema(conn: sqlite3.Connection) -> None:
    """
    Create all tables used by database tools.

    This replicates the essential Stash database schema for testing.

    Args:
        conn: SQLite connection to create schema in
    """
    cursor = conn.cursor()

    cursor.executescript("""
        -- Performers table
        CREATE TABLE performers (
            id INTEGER PRIMARY KEY,
            name TEXT NOT NULL,
            disambiguation TEXT,
            birthdate TEXT,
            death_date TEXT,
            gender TEXT,
            ethnicity TEXT,
            country TEXT,
            hair_color TEXT,
            eye_color TEXT,
            height INTEGER,
            weight INTEGER,
            measurements TEXT,
            fake_tits TEXT,
            tattoos TEXT,
            piercings TEXT,
            details TEXT,
            favorite INTEGER DEFAULT 0,
            rating INTEGER,
            career_length TEXT,
            created_at TEXT,
            updated_at TEXT
        );

        CREATE TABLE performer_aliases (
            performer_id INTEGER NOT NULL,
            alias TEXT NOT NULL,
            FOREIGN KEY (performer_id) REFERENCES performers(id)
        );

        CREATE TABLE performers_tags (
            performer_id INTEGER NOT NULL,
            tag_id INTEGER NOT NULL,
            PRIMARY KEY (performer_id, tag_id),
            FOREIGN KEY (performer_id) REFERENCES performers(id),
            FOREIGN KEY (tag_id) REFERENCES tags(id)
        );

        -- Tags table
        CREATE TABLE tags (
            id INTEGER PRIMARY KEY,
            name TEXT NOT NULL,
            description TEXT,
            favorite INTEGER DEFAULT 0,
            created_at TEXT,
            updated_at TEXT
        );

        CREATE TABLE tags_relations (
            parent_id INTEGER NOT NULL,
            child_id INTEGER NOT NULL,
            PRIMARY KEY (parent_id, child_id),
            FOREIGN KEY (parent_id) REFERENCES tags(id),
            FOREIGN KEY (child_id) REFERENCES tags(id)
        );

        -- Studios table
        CREATE TABLE studios (
            id INTEGER PRIMARY KEY,
            name TEXT NOT NULL,
            url TEXT,
            parent_id INTEGER,
            details TEXT,
            favorite INTEGER DEFAULT 0,
            rating INTEGER,
            created_at TEXT,
            updated_at TEXT,
            FOREIGN KEY (parent_id) REFERENCES studios(id)
        );

        -- Groups (formerly movies) table
        CREATE TABLE groups (
            id INTEGER PRIMARY KEY,
            name TEXT NOT NULL,
            date TEXT,
            director TEXT,
            studio_id INTEGER,
            rating INTEGER,
            duration INTEGER,
            synopsis TEXT,
            front_image_blob BLOB,
            back_image_blob BLOB,
            created_at TEXT,
            updated_at TEXT,
            FOREIGN KEY (studio_id) REFERENCES studios(id)
        );

        -- Scenes table
        CREATE TABLE scenes (
            id INTEGER PRIMARY KEY,
            title TEXT,
            code TEXT,
            details TEXT,
            director TEXT,
            url TEXT,
            date TEXT,
            rating INTEGER,
            organized INTEGER DEFAULT 0,
            studio_id INTEGER,
            resume_time REAL DEFAULT 0,
            play_count INTEGER DEFAULT 0,
            play_duration REAL DEFAULT 0,
            o_counter INTEGER DEFAULT 0,
            created_at TEXT,
            updated_at TEXT,
            FOREIGN KEY (studio_id) REFERENCES studios(id)
        );

        -- Junction tables
        CREATE TABLE performers_scenes (
            performer_id INTEGER NOT NULL,
            scene_id INTEGER NOT NULL,
            PRIMARY KEY (performer_id, scene_id),
            FOREIGN KEY (performer_id) REFERENCES performers(id),
            FOREIGN KEY (scene_id) REFERENCES scenes(id)
        );

        CREATE TABLE scenes_tags (
            scene_id INTEGER NOT NULL,
            tag_id INTEGER NOT NULL,
            PRIMARY KEY (scene_id, tag_id),
            FOREIGN KEY (scene_id) REFERENCES scenes(id),
            FOREIGN KEY (tag_id) REFERENCES tags(id)
        );

        CREATE TABLE groups_scenes (
            group_id INTEGER NOT NULL,
            scene_id INTEGER NOT NULL,
            scene_index INTEGER,
            PRIMARY KEY (group_id, scene_id),
            FOREIGN KEY (group_id) REFERENCES groups(id),
            FOREIGN KEY (scene_id) REFERENCES scenes(id)
        );

        -- View history tables
        CREATE TABLE scenes_view_dates (
            scene_id INTEGER NOT NULL,
            view_date TEXT NOT NULL,
            FOREIGN KEY (scene_id) REFERENCES scenes(id)
        );

        CREATE TABLE scenes_o_dates (
            scene_id INTEGER NOT NULL,
            o_date TEXT NOT NULL,
            FOREIGN KEY (scene_id) REFERENCES scenes(id)
        );

        -- File tables
        CREATE TABLE files (
            id INTEGER PRIMARY KEY,
            basename TEXT NOT NULL,
            size INTEGER DEFAULT 0,
            mod_time TEXT,
            created_at TEXT,
            updated_at TEXT
        );

        CREATE TABLE folders (
            id INTEGER PRIMARY KEY,
            path TEXT NOT NULL,
            created_at TEXT,
            updated_at TEXT
        );

        CREATE TABLE scenes_files (
            scene_id INTEGER NOT NULL,
            file_id INTEGER NOT NULL,
            "primary" INTEGER DEFAULT 0,
            PRIMARY KEY (scene_id, file_id),
            FOREIGN KEY (scene_id) REFERENCES scenes(id),
            FOREIGN KEY (file_id) REFERENCES files(id)
        );

        CREATE TABLE video_files (
            file_id INTEGER PRIMARY KEY,
            duration REAL,
            width INTEGER,
            height INTEGER,
            video_codec TEXT,
            audio_codec TEXT,
            frame_rate REAL,
            bit_rate INTEGER,
            interactive INTEGER DEFAULT 0,
            interactive_speed INTEGER,
            FOREIGN KEY (file_id) REFERENCES files(id)
        );

        CREATE TABLE files_fingerprints (
            file_id INTEGER NOT NULL,
            type TEXT NOT NULL,
            fingerprint TEXT NOT NULL,
            PRIMARY KEY (file_id, type),
            FOREIGN KEY (file_id) REFERENCES files(id)
        );

        -- Scene markers table
        CREATE TABLE scene_markers (
            id INTEGER PRIMARY KEY,
            scene_id INTEGER NOT NULL,
            primary_tag_id INTEGER,
            title TEXT,
            seconds REAL NOT NULL,
            end_seconds REAL,
            created_at TEXT,
            updated_at TEXT,
            FOREIGN KEY (scene_id) REFERENCES scenes(id),
            FOREIGN KEY (primary_tag_id) REFERENCES tags(id)
        );

        CREATE TABLE scene_markers_tags (
            scene_marker_id INTEGER NOT NULL,
            tag_id INTEGER NOT NULL,
            PRIMARY KEY (scene_marker_id, tag_id),
            FOREIGN KEY (scene_marker_id) REFERENCES scene_markers(id),
            FOREIGN KEY (tag_id) REFERENCES tags(id)
        );

        -- Galleries table (for completeness)
        CREATE TABLE galleries (
            id INTEGER PRIMARY KEY,
            title TEXT,
            date TEXT,
            rating INTEGER,
            studio_id INTEGER,
            created_at TEXT,
            updated_at TEXT,
            FOREIGN KEY (studio_id) REFERENCES studios(id)
        );

        -- Create indexes for common queries
        CREATE INDEX idx_performers_name ON performers(name);
        CREATE INDEX idx_tags_name ON tags(name);
        CREATE INDEX idx_studios_name ON studios(name);
        CREATE INDEX idx_scenes_date ON scenes(date);
        CREATE INDEX idx_scenes_rating ON scenes(rating);
        CREATE INDEX idx_scenes_view_dates_scene ON scenes_view_dates(scene_id);
        CREATE INDEX idx_scenes_o_dates_scene ON scenes_o_dates(scene_id);
    """)

    conn.commit()


def populate_test_data(conn: sqlite3.Connection) -> None:
    """
    Populate database with deterministic test data.

    Creates a realistic dataset for testing all database tools.

    Args:
        conn: SQLite connection to populate
    """
    cursor = conn.cursor()

    # Insert performers
    # fake_tits: None = natural, "Augmented" = enhanced
    # Format: (id, name, disambiguation, birthdate, death_date, gender, ethnicity, country,
    #          hair_color, eye_color, height, weight, measurements, fake_tits, tattoos, piercings,
    #          details, favorite, rating, career_length, created_at, updated_at)
    performers: list[tuple] = [
        (
            1,
            "Jane Doe",
            None,
            "1990-01-15",
            None,
            "female",
            "Caucasian",
            "USA",
            "brunette",
            "brown",
            165,
            55,
            "34-24-34",
            None,
            None,
            None,
            "Popular performer",
            1,
            5,
            None,
            "2022-01-01",
            "2023-01-01",
        ),
        (
            2,
            "John Smith",
            None,
            "1988-05-20",
            None,
            "male",
            "Caucasian",
            "UK",
            None,
            "blue",
            180,
            80,
            None,
            None,
            None,
            None,
            None,
            0,
            4,
            None,
            "2022-01-01",
            "2023-01-01",
        ),
        (
            3,
            "Alice Wonder",
            "disambiguation1",
            "1992-03-10",
            None,
            "female",
            "Asian",
            "Japan",
            "black",
            "brown",
            160,
            50,
            "32-22-32",
            "Augmented",
            "small back tattoo",
            None,
            "Award-winning performer",
            1,
            5,
            None,
            "2022-01-01",
            "2023-01-01",
        ),
        (
            4,
            "Bob Builder",
            None,
            None,
            None,
            "male",
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            0,
            3,
            None,
            "2022-01-01",
            "2023-01-01",
        ),
        (
            5,
            "Carol Davis",
            None,
            "1985-12-25",
            None,
            "female",
            "African American",
            "USA",
            "blonde",
            "green",
            170,
            60,
            "36-26-36",
            None,
            None,
            "navel",
            None,
            0,
            4,
            None,
            "2022-01-01",
            "2023-01-01",
        ),
    ]
    cursor.executemany(
        """INSERT INTO performers
           (id, name, disambiguation, birthdate, death_date, gender, ethnicity, country,
            hair_color, eye_color, height, weight, measurements, fake_tits, tattoos, piercings,
            details, favorite, rating, career_length, created_at, updated_at)
           VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
        performers,
    )

    # Insert performer aliases
    aliases: list[tuple[int, str]] = [
        (1, "JD"),
        (1, "Jane D"),
        (3, "Alice W"),
    ]
    cursor.executemany("INSERT INTO performer_aliases (performer_id, alias) VALUES (?, ?)", aliases)

    # Insert tags with hierarchy
    tags: list[tuple] = [
        (1, "oral", "Oral activities", 0, "2022-01-01", "2023-01-01"),
        (2, "blowjob", "Specific oral act", 0, "2022-01-01", "2023-01-01"),
        (3, "deepthroat", "Advanced oral technique", 0, "2022-01-01", "2023-01-01"),
        (4, "anal", "Anal activities", 0, "2022-01-01", "2023-01-01"),
        (5, "position", "Position category", 0, "2022-01-01", "2023-01-01"),
        (6, "doggy", "Doggy style position", 0, "2022-01-01", "2023-01-01"),
        (7, "missionary", "Missionary position", 0, "2022-01-01", "2023-01-01"),
        (8, "brunette", "Hair color", 1, "2022-01-01", "2023-01-01"),
        (9, "blonde", "Hair color", 0, "2022-01-01", "2023-01-01"),
        (10, "excluded_parent", "Should be excluded in tests", 0, "2022-01-01", "2023-01-01"),
        (11, "excluded_child", "Child of excluded parent", 0, "2022-01-01", "2023-01-01"),
        (12, "interactive", "Has funscript", 0, "2022-01-01", "2023-01-01"),
    ]
    cursor.executemany(
        """INSERT INTO tags (id, name, description, favorite, created_at, updated_at)
           VALUES (?, ?, ?, ?, ?, ?)""",
        tags,
    )

    # Insert tag relations (hierarchy)
    tag_relations: list[tuple[int, int]] = [
        (1, 2),  # oral -> blowjob
        (2, 3),  # blowjob -> deepthroat
        (5, 6),  # position -> doggy
        (5, 7),  # position -> missionary
        (10, 11),  # excluded_parent -> excluded_child
    ]
    cursor.executemany(
        "INSERT INTO tags_relations (parent_id, child_id) VALUES (?, ?)", tag_relations
    )

    # Insert performer tags (direct tags on performers)
    performer_tags: list[tuple[int, int]] = [
        (1, 8),  # Jane Doe is brunette
        (5, 9),  # Carol Davis is blonde
    ]
    cursor.executemany(
        "INSERT INTO performers_tags (performer_id, tag_id) VALUES (?, ?)", performer_tags
    )

    # Insert studios with hierarchy
    # Format: (id, name, url, parent_id, details, favorite, rating, created_at, updated_at)
    studios: list[tuple] = [
        (
            1,
            "Big Studio",
            "https://bigstudio.com",
            None,
            "Major studio network",
            1,
            5,
            "2022-01-01",
            "2023-01-01",
        ),
        (2, "Sub Studio A", "https://substudioa.com", 1, None, 0, 4, "2022-01-01", "2023-01-01"),
        (3, "Sub Studio B", "https://substudiob.com", 1, None, 0, 4, "2022-01-01", "2023-01-01"),
        (
            4,
            "Independent Studio",
            "https://independent.com",
            None,
            "Independent producer",
            0,
            3,
            "2022-01-01",
            "2023-01-01",
        ),
        (
            5,
            "Another Network",
            "https://anothernetwork.com",
            None,
            None,
            0,
            4,
            "2022-01-01",
            "2023-01-01",
        ),
        (6, "Sub Studio C", "https://substudioc.com", 5, None, 0, 3, "2022-01-01", "2023-01-01"),
    ]
    cursor.executemany(
        """INSERT INTO studios
           (id, name, url, parent_id, details, favorite, rating, created_at, updated_at)
           VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)""",
        studios,
    )

    # Insert groups
    groups: list[tuple] = [
        (
            1,
            "Movie Series 1",
            "2023-01-01",
            "Director A",
            1,
            5,
            7200,
            "First movie",
            None,
            None,
            "2022-01-01",
            "2023-01-01",
        ),
        (
            2,
            "Movie Series 2",
            "2023-06-15",
            "Director B",
            4,
            4,
            5400,
            "Second movie",
            None,
            None,
            "2022-01-01",
            "2023-01-01",
        ),
    ]
    cursor.executemany(
        """INSERT INTO groups
           (id, name, date, director, studio_id, rating, duration,
            synopsis, front_image_blob, back_image_blob, created_at, updated_at)
           VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
        groups,
    )

    # Insert scenes (20 scenes for robust testing)
    # Rating scale: 1-100 (where 100=5 stars, 80=4 stars, 60=3 stars, 40=2 stars, 20=1 star)
    scenes: list[tuple] = [
        (
            1,
            "Scene One",
            "SC001",
            "First scene details",
            None,
            None,
            "2023-01-15",
            100,
            1,
            1,
            None,
            3,
            1800.0,
            2,
            "2022-01-01",
            "2023-01-01",
        ),
        (
            2,
            "Scene Two",
            "SC002",
            None,
            None,
            None,
            "2023-02-20",
            80,
            0,
            2,
            120.5,
            2,
            900.0,
            0,
            "2022-01-01",
            "2023-01-01",
        ),
        (
            3,
            "Scene Three",
            "SC003",
            "Third scene",
            None,
            None,
            "2023-03-10",
            100,
            1,
            3,
            None,
            5,
            2400.0,
            3,
            "2022-01-01",
            "2023-01-01",
        ),
        (
            4,
            "Scene Four",
            "SC004",
            None,
            None,
            None,
            "2023-04-01",
            60,
            0,
            4,
            300.0,
            1,
            600.0,
            0,
            "2022-01-01",
            "2023-01-01",
        ),
        (
            5,
            "Scene Five",
            "SC005",
            None,
            None,
            None,
            "2023-05-25",
            80,
            0,
            1,
            None,
            4,
            1500.0,
            0,
            "2022-01-01",
            "2023-01-01",
        ),
        (
            6,
            "Scene Six",
            "SC006",
            None,
            None,
            None,
            "2023-06-10",
            40,
            0,
            2,
            None,
            0,
            0.0,
            0,
            "2022-01-01",
            "2023-01-01",
        ),
        (
            7,
            "Scene Seven",
            "SC007",
            "Great scene",
            None,
            None,
            "2023-07-20",
            100,
            1,
            3,
            60.0,
            8,
            3600.0,
            4,
            "2022-01-01",
            "2023-01-01",
        ),
        (
            8,
            "Scene Eight",
            "SC008",
            None,
            None,
            None,
            "2023-08-15",
            80,
            0,
            4,
            None,
            2,
            1200.0,
            0,
            "2022-01-01",
            "2023-01-01",
        ),
        (
            9,
            "Scene Nine",
            "SC009",
            None,
            None,
            None,
            "2023-09-01",
            60,
            0,
            1,
            None,
            1,
            450.0,
            0,
            "2022-01-01",
            "2023-01-01",
        ),
        (
            10,
            "Scene Ten",
            "SC010",
            "Popular scene",
            None,
            None,
            "2023-10-10",
            100,
            1,
            2,
            None,
            10,
            4500.0,
            4,
            "2022-01-01",
            "2023-01-01",
        ),
        (
            11,
            "Scene Eleven",
            "SC011",
            None,
            None,
            None,
            "2022-01-01",
            80,
            0,
            1,
            None,
            3,
            1350.0,
            0,
            "2022-01-01",
            "2022-06-01",
        ),
        (
            12,
            "Scene Twelve",
            "SC012",
            None,
            None,
            None,
            "2022-06-01",
            60,
            0,
            2,
            None,
            2,
            900.0,
            0,
            "2022-01-01",
            "2022-12-01",
        ),
        (
            13,
            "Interactive Scene",
            "SC013",
            "Has funscript",
            None,
            None,
            "2023-11-01",
            100,
            1,
            1,
            None,
            6,
            2700.0,
            2,
            "2022-01-01",
            "2023-11-01",
        ),
        (
            14,
            "Unwatched Scene",
            "SC014",
            "Never watched",
            None,
            None,
            "2023-12-01",
            None,
            0,
            3,
            None,
            0,
            0.0,
            0,
            "2022-01-01",
            "2023-12-01",
        ),
        (
            15,
            "Resume Scene",
            "SC015",
            None,
            None,
            None,
            "2023-12-15",
            80,
            0,
            4,
            500.0,
            1,
            300.0,
            0,
            "2022-01-01",
            "2023-12-15",
        ),
        (
            16,
            None,
            "SC016",
            None,
            None,
            None,
            "2023-06-01",
            60,
            0,
            1,
            None,
            1,
            600.0,
            0,
            "2022-01-01",
            "2023-06-01",
        ),
        (
            17,
            "Duplicate A",
            "SC017",
            None,
            None,
            None,
            "2023-07-01",
            80,
            0,
            2,
            None,
            2,
            1200.0,
            0,
            "2022-01-01",
            "2023-07-01",
        ),
        (
            18,
            "Duplicate B",
            "SC018",
            None,
            None,
            None,
            "2023-07-02",
            80,
            0,
            2,
            None,
            1,
            600.0,
            0,
            "2022-01-01",
            "2023-07-02",
        ),
        (
            19,
            "Shared Scene",
            "SC019",
            "Multiple performers",
            None,
            None,
            "2023-08-01",
            100,
            1,
            1,
            None,
            3,
            1500.0,
            0,
            "2022-01-01",
            "2023-08-01",
        ),
        (
            20,
            "Solo Scene",
            "SC020",
            None,
            None,
            None,
            "2023-09-01",
            80,
            0,
            4,
            None,
            2,
            900.0,
            0,
            "2022-01-01",
            "2023-09-01",
        ),
    ]
    cursor.executemany(
        """INSERT INTO scenes
           (id, title, code, details, director, url, date, rating, organized,
            studio_id, resume_time, play_count, play_duration, o_counter,
            created_at, updated_at)
           VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
        scenes,
    )

    # Link performers to scenes
    performers_scenes: list[tuple[int, int]] = [
        # Jane Doe - 7 scenes (most popular performer)
        (1, 1),
        (1, 3),
        (1, 5),
        (1, 7),
        (1, 10),
        (1, 13),
        (1, 19),
        # John Smith - 6 scenes
        (2, 1),
        (2, 2),
        (2, 4),
        (2, 6),
        (2, 8),
        (2, 19),
        # Alice Wonder - 6 scenes
        (3, 2),
        (3, 3),
        (3, 7),
        (3, 9),
        (3, 11),
        (3, 14),
        # Bob Builder - 4 scenes
        (4, 4),
        (4, 5),
        (4, 8),
        (4, 12),
        # Carol Davis - 5 scenes
        (5, 6),
        (5, 9),
        (5, 10),
        (5, 15),
        (5, 20),
    ]
    cursor.executemany(
        "INSERT INTO performers_scenes (performer_id, scene_id) VALUES (?, ?)", performers_scenes
    )

    # Link tags to scenes
    scenes_tags_data: list[tuple[int, int]] = [
        (1, 1),
        (1, 2),
        (1, 8),  # Scene 1: oral, blowjob, brunette
        (2, 1),
        (2, 6),  # Scene 2: oral, doggy
        (3, 2),
        (3, 3),
        (3, 4),  # Scene 3: blowjob, deepthroat, anal
        (4, 5),
        (4, 7),  # Scene 4: position, missionary
        (5, 1),
        (5, 6),
        (5, 9),  # Scene 5: oral, doggy, blonde
        (6, 7),  # Scene 6: missionary
        (7, 2),
        (7, 4),
        (7, 6),  # Scene 7: blowjob, anal, doggy
        (8, 5),  # Scene 8: position
        (9, 1),  # Scene 9: oral
        (10, 1),
        (10, 2),
        (10, 4),  # Scene 10: oral, blowjob, anal
        (11, 8),  # Scene 11: brunette
        (12, 6),  # Scene 12: doggy
        (13, 12),  # Scene 13: interactive (for funscript)
        (14, 8),  # Scene 14: brunette (unwatched)
        (19, 1),
        (19, 2),  # Scene 19: oral, blowjob
        (20, 9),  # Scene 20: blonde
    ]
    cursor.executemany("INSERT INTO scenes_tags (scene_id, tag_id) VALUES (?, ?)", scenes_tags_data)

    # Link groups to scenes
    groups_scenes_data: list[tuple[int, int, int]] = [
        (1, 1, 1),
        (1, 3, 2),
        (1, 5, 3),  # Group 1: scenes 1, 3, 5
        (2, 2, 1),
        (2, 4, 2),  # Group 2: scenes 2, 4
    ]
    cursor.executemany(
        "INSERT INTO groups_scenes (group_id, scene_id, scene_index) VALUES (?, ?, ?)",
        groups_scenes_data,
    )

    # Insert view history (spans 2022-2023)
    view_dates: list[tuple[int, str]] = [
        # Scene 1: 3 views
        (1, "2023-01-20 14:30:00"),
        (1, "2023-02-15 18:00:00"),
        (1, "2023-03-10 21:00:00"),
        # Scene 2: 2 views
        (2, "2023-02-25 10:00:00"),
        (2, "2023-03-05 16:00:00"),
        # Scene 3: 5 views
        (3, "2023-03-15 12:00:00"),
        (3, "2023-04-01 20:00:00"),
        (3, "2023-05-10 22:00:00"),
        (3, "2023-06-20 19:00:00"),
        (3, "2023-07-05 21:00:00"),
        # Scene 5: 4 views
        (5, "2023-05-30 15:00:00"),
        (5, "2023-06-15 17:00:00"),
        (5, "2023-07-20 14:00:00"),
        (5, "2023-08-10 16:00:00"),
        # Scene 7: 8 views (most viewed)
        (7, "2023-07-25 20:00:00"),
        (7, "2023-08-01 21:00:00"),
        (7, "2023-08-15 22:00:00"),
        (7, "2023-09-01 23:00:00"),
        (7, "2023-09-15 20:00:00"),
        (7, "2023-10-01 21:00:00"),
        (7, "2023-10-15 22:00:00"),
        (7, "2023-11-01 23:00:00"),
        # Scene 10: 10 views
        (10, "2023-10-15 14:00:00"),
        (10, "2023-10-20 15:00:00"),
        (10, "2023-10-25 16:00:00"),
        (10, "2023-11-01 17:00:00"),
        (10, "2023-11-05 18:00:00"),
        (10, "2023-11-10 19:00:00"),
        (10, "2023-11-15 20:00:00"),
        (10, "2023-11-20 21:00:00"),
        (10, "2023-11-25 22:00:00"),
        (10, "2023-12-01 23:00:00"),
        # Scene 11: 3 views (older scene from 2022)
        (11, "2022-02-01 10:00:00"),
        (11, "2022-03-15 11:00:00"),
        (11, "2022-04-20 12:00:00"),
        # Scene 12: 2 views
        (12, "2022-07-01 14:00:00"),
        (12, "2022-08-15 15:00:00"),
        # Scene 13: 6 views (interactive)
        (13, "2023-11-05 20:00:00"),
        (13, "2023-11-10 21:00:00"),
        (13, "2023-11-15 22:00:00"),
        (13, "2023-11-20 23:00:00"),
        (13, "2023-11-25 20:00:00"),
        (13, "2023-12-01 21:00:00"),
        # Various other views
        (4, "2023-04-10 10:00:00"),
        (8, "2023-08-20 12:00:00"),
        (8, "2023-09-10 14:00:00"),
        (9, "2023-09-05 11:00:00"),
        (15, "2023-12-20 16:00:00"),
        (16, "2023-06-15 13:00:00"),
        (17, "2023-07-10 15:00:00"),
        (17, "2023-07-20 17:00:00"),
        (18, "2023-07-15 14:00:00"),
        (19, "2023-08-10 18:00:00"),
        (19, "2023-08-20 19:00:00"),
        (19, "2023-08-30 20:00:00"),
        (20, "2023-09-10 12:00:00"),
        (20, "2023-09-20 14:00:00"),
    ]
    cursor.executemany(
        "INSERT INTO scenes_view_dates (scene_id, view_date) VALUES (?, ?)", view_dates
    )

    # Insert O history
    o_dates: list[tuple[int, str]] = [
        # Scene 1: 2 O's
        (1, "2023-01-20 14:35:00"),
        (1, "2023-03-10 21:05:00"),
        # Scene 3: 3 O's
        (3, "2023-04-01 20:10:00"),
        (3, "2023-05-10 22:15:00"),
        (3, "2023-07-05 21:20:00"),
        # Scene 7: 4 O's
        (7, "2023-08-01 21:10:00"),
        (7, "2023-09-01 23:15:00"),
        (7, "2023-10-01 21:20:00"),
        (7, "2023-11-01 23:25:00"),
        # Scene 10: 4 O's
        (10, "2023-10-20 15:10:00"),
        (10, "2023-11-01 17:15:00"),
        (10, "2023-11-15 20:20:00"),
        (10, "2023-12-01 23:25:00"),
        # Scene 13: 2 O's (interactive)
        (13, "2023-11-10 21:10:00"),
        (13, "2023-11-25 20:15:00"),
    ]
    cursor.executemany("INSERT INTO scenes_o_dates (scene_id, o_date) VALUES (?, ?)", o_dates)

    # Insert files
    files_data: list[tuple] = [
        (1, "scene1.mp4", 1073741824, "2023-01-15 10:00:00", "2023-01-15", "2023-01-15"),
        (2, "scene2.mp4", 536870912, "2023-02-20 10:00:00", "2023-02-20", "2023-02-20"),
        (3, "scene3.mp4", 2147483648, "2023-03-10 10:00:00", "2023-03-10", "2023-03-10"),
        (4, "scene4.mp4", 268435456, "2023-04-01 10:00:00", "2023-04-01", "2023-04-01"),
        (5, "scene5.mp4", 1610612736, "2023-05-25 10:00:00", "2023-05-25", "2023-05-25"),
        (6, "scene6.mp4", 805306368, "2023-06-10 10:00:00", "2023-06-10", "2023-06-10"),
        (7, "scene7.mp4", 3221225472, "2023-07-20 10:00:00", "2023-07-20", "2023-07-20"),
        (8, "scene8.mp4", 1073741824, "2023-08-15 10:00:00", "2023-08-15", "2023-08-15"),
        (
            13,
            "scene13_interactive.mp4",
            2684354560,
            "2023-11-01 10:00:00",
            "2023-11-01",
            "2023-11-01",
        ),
        (14, "scene14.mp4", 1073741824, "2023-12-01 10:00:00", "2023-12-01", "2023-12-01"),
        (15, "scene15.mp4", 536870912, "2023-12-15 10:00:00", "2023-12-15", "2023-12-15"),
        (17, "duplicate_a.mp4", 1073741824, "2023-07-01 10:00:00", "2023-07-01", "2023-07-01"),
        (18, "duplicate_b.mp4", 1073741824, "2023-07-02 10:00:00", "2023-07-02", "2023-07-02"),
        (19, "scene19.mp4", 1610612736, "2023-08-01 10:00:00", "2023-08-01", "2023-08-01"),
        (20, "scene20.mp4", 805306368, "2023-09-01 10:00:00", "2023-09-01", "2023-09-01"),
    ]
    cursor.executemany(
        """INSERT INTO files (id, basename, size, mod_time, created_at, updated_at)
           VALUES (?, ?, ?, ?, ?, ?)""",
        files_data,
    )

    # Link scenes to files
    scenes_files_data: list[tuple[int, int, int]] = [
        (1, 1, 1),
        (2, 2, 1),
        (3, 3, 1),
        (4, 4, 1),
        (5, 5, 1),
        (6, 6, 1),
        (7, 7, 1),
        (8, 8, 1),
        (13, 13, 1),
        (14, 14, 1),
        (15, 15, 1),
        (17, 17, 1),
        (18, 18, 1),
        (19, 19, 1),
        (20, 20, 1),
    ]
    cursor.executemany(
        'INSERT INTO scenes_files (scene_id, file_id, "primary") VALUES (?, ?, ?)',
        scenes_files_data,
    )

    # Insert video file metadata
    video_files_data: list[tuple] = [
        (1, 1800.0, 1920, 1080, "h264", "aac", 30.0, 5000000, 0, None),
        (2, 1200.0, 1920, 1080, "h264", "aac", 30.0, 4000000, 0, None),
        (3, 2700.0, 3840, 2160, "h265", "aac", 60.0, 10000000, 0, None),
        (4, 900.0, 1280, 720, "h264", "aac", 30.0, 3000000, 0, None),
        (5, 2100.0, 1920, 1080, "h264", "aac", 30.0, 5000000, 0, None),
        (6, 1500.0, 1920, 1080, "h264", "aac", 30.0, 4500000, 0, None),
        (7, 3600.0, 3840, 2160, "h265", "aac", 60.0, 12000000, 0, None),
        (8, 1800.0, 1920, 1080, "h264", "aac", 30.0, 5000000, 0, None),
        (13, 2400.0, 1920, 1080, "h264", "aac", 30.0, 6000000, 1, 150),
        (14, 1800.0, 1920, 1080, "h264", "aac", 30.0, 5000000, 0, None),
        (15, 1200.0, 1920, 1080, "h264", "aac", 30.0, 4000000, 0, None),
        (17, 1800.0, 1920, 1080, "h264", "aac", 30.0, 5000000, 0, None),
        (18, 1800.0, 1920, 1080, "h264", "aac", 30.0, 5000000, 0, None),
        (19, 2100.0, 1920, 1080, "h264", "aac", 30.0, 5500000, 0, None),
        (20, 1500.0, 1920, 1080, "h264", "aac", 30.0, 4500000, 0, None),
    ]
    cursor.executemany(
        """INSERT INTO video_files
           (file_id, duration, width, height, video_codec, audio_codec,
            frame_rate, bit_rate, interactive, interactive_speed)
           VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
        video_files_data,
    )

    # Insert fingerprints (for duplicate detection)
    fingerprints: list[tuple[int, str, str]] = [
        (1, "phash", "abc123def456"),
        (2, "phash", "xyz789ghi012"),
        (3, "phash", "unique_hash_3"),
        (4, "phash", "unique_hash_4"),
        (5, "phash", "unique_hash_5"),
        (6, "phash", "unique_hash_6"),
        (7, "phash", "unique_hash_7"),
        (8, "phash", "unique_hash_8"),
        (13, "phash", "interactive_hash"),
        (17, "phash", "duplicate_hash"),  # Same hash as 18
        (18, "phash", "duplicate_hash"),  # Same hash as 17 - duplicate
    ]
    cursor.executemany(
        "INSERT INTO files_fingerprints (file_id, type, fingerprint) VALUES (?, ?, ?)", fingerprints
    )

    # Insert scene markers
    markers: list[tuple] = [
        (1, 1, 1, "Opening", 0.0, 30.0, "2023-01-15", "2023-01-15"),
        (2, 1, 2, "Main Action", 120.0, 600.0, "2023-01-15", "2023-01-15"),
        (3, 3, 3, "Climax", 1800.0, 2100.0, "2023-03-10", "2023-03-10"),
        (4, 7, 4, "Start", 0.0, 60.0, "2023-07-20", "2023-07-20"),
        (5, 7, 6, "Position Change", 1200.0, 1800.0, "2023-07-20", "2023-07-20"),
    ]
    cursor.executemany(
        """INSERT INTO scene_markers
           (id, scene_id, primary_tag_id, title, seconds, end_seconds,
            created_at, updated_at)
           VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
        markers,
    )

    conn.commit()
