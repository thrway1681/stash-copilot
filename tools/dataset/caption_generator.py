"""Generate natural language captions from Stash scene tags."""
from tools.dataset.constants import (
    ADMIN_TAGS, BODY_TYPE_TAGS, PHYSICAL_ATTRIBUTE_TAGS,
    ACT_TAGS, OUTCOME_TAGS, POSITION_TAGS, CONTENT_STYLE_TAGS, SETTING_TAGS,
)


def generate_caption(
    tags: list[str],
    performers: list[str] | None = None,
    studio: str | None = None,
    visual_notes: str | None = None,
) -> str:
    """Generate a natural language caption for a scene from its tags."""
    content_tags = {t for t in tags if t not in ADMIN_TAGS}

    body_types    = content_tags & BODY_TYPE_TAGS
    physical      = content_tags & PHYSICAL_ATTRIBUTE_TAGS
    acts          = content_tags & ACT_TAGS
    outcomes      = content_tags & OUTCOME_TAGS
    positions     = content_tags & POSITION_TAGS
    styles        = content_tags & CONTENT_STYLE_TAGS
    settings      = content_tags & SETTING_TAGS
    other         = content_tags - body_types - physical - acts - outcomes - positions - styles - settings

    parts: list[str] = []

    # --- Lead with production style ---
    style_words: list[str] = []
    if "Non-Nude" in styles:
        style_words.append("non-nude")
    elif "Softcore" in styles:
        style_words.append("softcore")
    elif "Hardcore" in styles:
        style_words.append("hardcore")
    elif "Amateur" in styles or "Homemade" in styles:
        style_words.append("amateur")
    if "POV" in styles or "Male POV" in styles:
        style_words.append("POV")
    if "VR" in styles:
        style_words.append("VR")
    style_lead = " ".join(style_words) if style_words else "adult"

    # --- Performer description ---
    perf_parts: list[str] = []
    if body_types:
        perf_parts.append(_join(body_types))
    if "Solo" in styles:
        perf_parts.append("solo")

    phys_ordered: list[str] = []
    tit_tags = physical & {"Big Tits","Medium Tits","Small Tits","Natural Tits",
                           "Fake Tits","Perfect Tits","Saggy Tits"}
    if tit_tags:
        phys_ordered.append(_join(tit_tags).lower())
    hair_tags = physical & {"Blonde Hair", "Colored Hair", "Pigtails"}
    if hair_tags:
        phys_ordered.append(_join(hair_tags).lower())
    remainder_phys = physical - tit_tags - hair_tags - {"BBC"}
    if remainder_phys:
        phys_ordered.extend(t.lower() for t in sorted(remainder_phys))

    ethnicity_tags = other & {"Asian","Asian Woman","Black","White","Latina",
                              "Filipino","Japanese","Interracial","PAAG","PAWG"}
    if ethnicity_tags and not body_types:
        perf_parts.insert(0, _join(ethnicity_tags).lower())

    perf_desc = "a performer"
    if perf_parts:
        perf_desc = f"a {', '.join(perf_parts)} performer"
    if phys_ordered:
        perf_desc += f" with {', '.join(phys_ordered)}"

    if performers:
        first = performers[0]
        perf_desc = f"{first} ({', '.join(perf_parts) if perf_parts else 'performer'})"
        if phys_ordered:
            perf_desc += f" with {', '.join(phys_ordered)}"

    parts.append(f"A {style_lead} scene featuring {perf_desc}.")

    # --- Acts + positions ---
    act_strs: list[str] = []
    if acts:
        primary_acts = acts - {"Oral Sex", "Couple Sex", "Outercourse"}
        if primary_acts:
            act_strs.append(_join(primary_acts).lower())
        elif acts:
            act_strs.append(_join(acts).lower())
    if positions:
        act_strs.append(f"{_join(positions).lower()} position")
    if outcomes:
        act_strs.append(f"ending with {_join(outcomes).lower()}")
    if act_strs:
        parts.append(f"The scene features {', '.join(act_strs)}.")

    # --- Setting ---
    setting_strs: list[str] = []
    if settings:
        setting_strs.append(_join(settings).lower())
    if studio:
        setting_strs.append(f"from {studio}")
    if setting_strs:
        parts.append(f"Shot {', '.join(setting_strs)}.")

    if visual_notes:
        parts.append(visual_notes.strip())

    return " ".join(parts)


def _join(tags: set[str], separator: str = ", ") -> str:
    """Join a set of tags into a readable string."""
    return separator.join(sorted(tags))
