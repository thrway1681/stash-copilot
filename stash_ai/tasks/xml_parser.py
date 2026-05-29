"""XML parsing utilities for structured LLM responses."""

import re
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from typing import Any, TypedDict


class StructuredTag(TypedDict):
    """A single tag with confidence and optional timestamp."""

    name: str
    confidence: int  # 0-100
    timestamp: str | None  # "0:45" format or None
    timestamp_seconds: float | None  # Converted to seconds
    reasoning: str  # Why this tag was suggested


class StructuredTagResponse(TypedDict):
    """Parsed structured tag response from XML."""

    tags: list[StructuredTag]
    raw_xml: str  # Original XML for debugging


@dataclass
class XMLParseResult:
    """Result of XML parsing attempt."""

    success: bool
    data: Any | None = None
    error: str | None = None
    fallback_used: bool = False


def parse_timestamp_to_seconds(timestamp: str | None) -> float | None:
    """
    Convert timestamp string to seconds.

    Handles formats: "0:45", "1:30", "1:30:45"
    Returns None if parsing fails.
    """
    if not timestamp:
        return None

    try:
        # Clean up the timestamp
        timestamp = timestamp.strip()

        parts = timestamp.split(":")
        if len(parts) == 2:
            # MM:SS
            minutes, seconds = int(parts[0]), int(parts[1])
            return float(minutes * 60 + seconds)
        elif len(parts) == 3:
            # HH:MM:SS
            hours, minutes, seconds = int(parts[0]), int(parts[1]), int(parts[2])
            return float(hours * 3600 + minutes * 60 + seconds)
        else:
            return None
    except (ValueError, IndexError):
        return None


def extract_xml_block(text: str, root_tag: str) -> str | None:
    """
    Extract XML block from text that may contain other content.

    Handles cases where LLM adds text before/after XML.
    """
    # Try to find the XML block with regex
    pattern = rf"<{root_tag}[^>]*>.*?</{root_tag}>"
    match = re.search(pattern, text, re.DOTALL | re.IGNORECASE)
    if match:
        return match.group(0)
    return None


def _clean_xml_string(xml_str: str) -> str:
    """Clean common XML issues from LLM output."""
    # Strip whitespace
    xml_str = xml_str.strip()

    # Handle unescaped ampersands (but not already escaped ones)
    xml_str = re.sub(r"&(?!amp;|lt;|gt;|quot;|apos;|#\d+;|#x[0-9a-fA-F]+;)", "&amp;", xml_str)

    # Handle unescaped < and > inside text content (tricky - best effort)
    # This is a common LLM mistake

    return xml_str


def parse_tags_xml(response: str) -> XMLParseResult:
    """
    Parse structured tags XML from LLM response.

    Expected format:
    <suggested_tags>
      <tag name="blowjob" confidence="95" timestamp="0:45">Clearly visible oral action</tag>
      <tag name="brunette" confidence="90">Performer has dark hair</tag>
    </suggested_tags>

    Returns XMLParseResult with success=True and StructuredTagResponse data on success,
    or success=False with error message on failure.
    """
    # Try to extract XML block
    xml_str = extract_xml_block(response, "suggested_tags")
    if not xml_str:
        return XMLParseResult(
            success=False,
            error="No <suggested_tags> block found in response",
            fallback_used=True,
        )

    try:
        # Clean up common XML issues
        xml_str = _clean_xml_string(xml_str)

        root = ET.fromstring(xml_str)

        tags: list[StructuredTag] = []
        for tag_el in root.findall("tag"):
            name = tag_el.get("name", "").strip().lower()
            if not name:
                continue

            # Parse confidence (default to 50 if not specified or invalid)
            confidence_str = tag_el.get("confidence", "50")
            try:
                confidence = max(0, min(100, int(confidence_str)))
            except ValueError:
                confidence = 50

            # Parse optional timestamp
            timestamp = tag_el.get("timestamp")
            if timestamp:
                timestamp = timestamp.strip()
                if not timestamp:
                    timestamp = None

            # Parse reasoning (tag text content)
            reasoning = (tag_el.text or "").strip()

            tags.append(
                StructuredTag(
                    name=name,
                    confidence=confidence,
                    timestamp=timestamp,
                    timestamp_seconds=parse_timestamp_to_seconds(timestamp),
                    reasoning=reasoning,
                )
            )

        result: StructuredTagResponse = {
            "tags": tags,
            "raw_xml": xml_str,
        }

        return XMLParseResult(success=True, data=result)

    except ET.ParseError as e:
        return XMLParseResult(
            success=False,
            error=f"XML parse error: {e!s}",
            fallback_used=True,
        )


# ============================================================================
# Structured Description Parsing
# ============================================================================


class PerformerDescription(TypedDict):
    """Description of a performer's appearance."""

    body_type: str  # e.g., "petite", "athletic", "curvy"
    hair: str  # e.g., "long brown hair"
    skin_tone: str  # e.g., "fair", "tan", "dark"
    notable_features: str  # e.g., "tattoos on back", "large breasts"


class ClothingDescription(TypedDict):
    """Description of what the performer is wearing."""

    initial: str  # What they're wearing at start
    changes: list[str]  # Any clothing changes during the scene
    final: str  # What they're wearing at the end (or "nude")


class ActionSegment(TypedDict):
    """A segment of action with timestamp."""

    timestamp_start: str | None  # "0:00" format
    timestamp_end: str | None  # "1:30" format
    description: str  # What happens in this segment


class StructuredDescription(TypedDict):
    """Fully structured scene description."""

    performer: PerformerDescription
    clothing: ClothingDescription
    actions: list[ActionSegment]
    setting: str  # e.g., "bedroom with white sheets"
    camera_angles: str  # e.g., "POV style with occasional wide shots"
    summary: str  # 2-3 sentence summary


class StructuredDescriptionResponse(TypedDict):
    """Parsed structured description from XML."""

    description: StructuredDescription
    raw_xml: str


def parse_description_xml(response: str) -> XMLParseResult:
    """
    Parse structured description XML from LLM response.

    Expected format:
    <scene_description>
      <performer>
        <body_type>petite</body_type>
        <hair>long brown hair</hair>
        <skin_tone>fair</skin_tone>
        <notable_features>small tattoo on hip</notable_features>
      </performer>
      <clothing>
        <initial>red lace lingerie set</initial>
        <changes>
          <change>removes bra at 1:30</change>
          <change>fully nude by 3:00</change>
        </changes>
        <final>nude</final>
      </clothing>
      <actions>
        <segment start="0:00" end="1:00">Teasing and dancing</segment>
        <segment start="1:00" end="2:30">Masturbation with fingers</segment>
      </actions>
      <setting>bedroom with pink lighting</setting>
      <camera_angles>POV style throughout</camera_angles>
      <summary>A petite brunette teases in lingerie before masturbating</summary>
    </scene_description>

    Returns XMLParseResult with structured data on success.
    """
    xml_str = extract_xml_block(response, "scene_description")
    if not xml_str:
        return XMLParseResult(
            success=False,
            error="No <scene_description> block found in response",
            fallback_used=True,
        )

    try:
        xml_str = _clean_xml_string(xml_str)
        root = ET.fromstring(xml_str)

        # Parse performer
        performer_el = root.find("performer")
        performer: PerformerDescription = {
            "body_type": _get_text(performer_el, "body_type", "unknown"),
            "hair": _get_text(performer_el, "hair", "unknown"),
            "skin_tone": _get_text(performer_el, "skin_tone", "unknown"),
            "notable_features": _get_text(performer_el, "notable_features", "none"),
        }

        # Parse clothing
        clothing_el = root.find("clothing")
        changes: list[str] = []
        if clothing_el is not None:
            changes_el = clothing_el.find("changes")
            if changes_el is not None:
                for change_el in changes_el.findall("change"):
                    if change_el.text:
                        changes.append(change_el.text.strip())

        clothing: ClothingDescription = {
            "initial": _get_text(clothing_el, "initial", "unknown"),
            "changes": changes,
            "final": _get_text(clothing_el, "final", "unknown"),
        }

        # Parse actions
        actions: list[ActionSegment] = []
        actions_el = root.find("actions")
        if actions_el is not None:
            for segment_el in actions_el.findall("segment"):
                actions.append(
                    ActionSegment(
                        timestamp_start=segment_el.get("start"),
                        timestamp_end=segment_el.get("end"),
                        description=(segment_el.text or "").strip(),
                    )
                )

        # Build full description
        description: StructuredDescription = {
            "performer": performer,
            "clothing": clothing,
            "actions": actions,
            "setting": _get_text(root, "setting", "unknown"),
            "camera_angles": _get_text(root, "camera_angles", "unknown"),
            "summary": _get_text(root, "summary", ""),
        }

        result: StructuredDescriptionResponse = {
            "description": description,
            "raw_xml": xml_str,
        }

        return XMLParseResult(success=True, data=result)

    except ET.ParseError as e:
        return XMLParseResult(
            success=False,
            error=f"XML parse error: {e!s}",
            fallback_used=True,
        )


def _get_text(parent: ET.Element | None, tag: str, default: str = "") -> str:
    """Get text content from a child element."""
    if parent is None:
        return default
    el = parent.find(tag)
    if el is None or el.text is None:
        return default
    return el.text.strip()


def structured_description_to_text(desc: StructuredDescription) -> str:
    """
    Convert structured description back to human-readable text.

    This is useful for displaying to users or passing to tag LLM.
    """
    lines: list[str] = []

    # Summary first
    if desc["summary"]:
        lines.append(desc["summary"])
        lines.append("")

    # Performer
    p = desc["performer"]
    lines.append(f"**Performer:** {p['body_type']} build, {p['hair']}, {p['skin_tone']} skin")
    if p["notable_features"] and p["notable_features"] != "none":
        lines.append(f"  Notable features: {p['notable_features']}")

    # Clothing
    c = desc["clothing"]
    lines.append(f"**Clothing:** Starts in {c['initial']}")
    for change in c["changes"]:
        lines.append(f"  - {change}")
    if c["final"] != c["initial"]:
        lines.append(f"  Ends: {c['final']}")

    # Actions
    if desc["actions"]:
        lines.append("")
        lines.append("**Action progression:**")
        for action in desc["actions"]:
            ts = ""
            if action["timestamp_start"]:
                ts = f"[{action['timestamp_start']}"
                if action["timestamp_end"]:
                    ts += f"-{action['timestamp_end']}"
                ts += "] "
            lines.append(f"  - {ts}{action['description']}")

    # Setting & camera
    if desc["setting"] != "unknown":
        lines.append(f"\n**Setting:** {desc['setting']}")
    if desc["camera_angles"] != "unknown":
        lines.append(f"**Camera:** {desc['camera_angles']}")

    return "\n".join(lines)


# ============================================================================
# Structured Prompt Templates
# ============================================================================


STRUCTURED_DESCRIPTION_PROMPT = """Analyze this adult video. {frame_count} frames are shown at timestamps: {frame_timestamps}

{performer_context}

Respond with structured XML describing the scene. Use this exact format:

<scene_description>
  <performer>
    <body_type>e.g., petite, athletic, curvy, slim, thick</body_type>
    <hair>color and style, e.g., long brown hair, short blonde pixie cut</hair>
    <skin_tone>e.g., fair, tan, olive, dark</skin_tone>
    <notable_features>tattoos, piercings, birthmarks, or "none"</notable_features>
  </performer>
  <clothing>
    <initial>what they wear at the start, e.g., red lace lingerie, nude</initial>
    <changes>
      <change>describe any clothing changes with timestamps</change>
    </changes>
    <final>what they wear at the end, e.g., nude, panties only</final>
  </clothing>
  <actions>
    <segment start="0:00" end="1:00">describe what happens in this time range</segment>
    <segment start="1:00" end="2:30">describe next action segment</segment>
  </actions>
  <setting>describe the location/environment</setting>
  <camera_angles>describe filming style, e.g., POV, wide shots, closeups</camera_angles>
  <summary>2-3 sentence summary of the entire scene</summary>
</scene_description>

Be explicit and accurate. Describe what you ACTUALLY SEE in the frames. Use casual terms (tits, ass, pussy)."""


STRUCTURED_TAG_PROMPT = """Based on the following scene description, suggest tags with confidence scores.

**Scene Description:**
{description}

**Available Tags in Database:**
{available_tags}

Respond with structured XML. Include confidence (0-100) and optional timestamp if the tag only applies to part of the scene:

<suggested_tags>
  <tag name="tag_name" confidence="95" timestamp="0:45">Brief reason why this tag applies</tag>
  <tag name="another_tag" confidence="80">Brief reason</tag>
</suggested_tags>

Rules:
- Only suggest tags from the available list OR obvious tags that should exist
- Use exact tag names from the database when possible
- Confidence 90+: Clearly visible/certain
- Confidence 70-89: Likely present
- Confidence 50-69: Possibly present
- Below 50: Don't include"""
