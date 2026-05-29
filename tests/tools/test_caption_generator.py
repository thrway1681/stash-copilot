from tools.dataset.caption_generator import generate_caption

def test_non_nude_solo_body_tags() -> None:
    tags = ["Non-Nude", "Solo", "PAWG", "Big Ass", "Small Tits", "Tan"]
    caption = generate_caption(tags, performers=["Mikaela Lafuente"])
    assert "non-nude" in caption.lower() or "solo" in caption.lower()
    assert "PAWG" in caption or "big ass" in caption.lower()
    assert "small tits" in caption.lower()

def test_blowjob_scene() -> None:
    tags = ["Amateur", "Blowjob", "Deepthroat", "POV", "Big Tits", "Blonde Hair"]
    caption = generate_caption(tags)
    assert "blowjob" in caption.lower() or "oral" in caption.lower()
    assert "pov" in caption.lower() or "POV" in caption
    assert "big tits" in caption.lower()

def test_admin_tags_excluded() -> None:
    tags = ["Embedded", "Blowjob", "Big Tits", "To Script", "Funscript"]
    caption = generate_caption(tags)
    assert "embedded" not in caption.lower()
    assert "funscript" not in caption.lower()
    assert "blowjob" in caption.lower()

def test_empty_meaningful_tags_returns_generic() -> None:
    tags = ["Embedded", "To Embed"]
    caption = generate_caption(tags)
    assert len(caption) > 10

def test_performer_included_when_provided() -> None:
    tags = ["Amateur", "Big Tits"]
    caption = generate_caption(tags, performers=["Jane Doe"])
    assert "Jane Doe" in caption or "performer" in caption.lower()
