from tools.dataset.constants import ADMIN_TAGS, BODY_TYPE_TAGS, ACT_TAGS

def test_admin_tags_are_frozenset() -> None:
    assert isinstance(ADMIN_TAGS, frozenset)
    assert "Embedded" in ADMIN_TAGS
    assert "Funscript" in ADMIN_TAGS

def test_no_overlap_between_act_and_admin() -> None:
    assert not (ADMIN_TAGS & ACT_TAGS)
    assert not (ADMIN_TAGS & BODY_TYPE_TAGS)

def test_embedded_is_admin_not_content() -> None:
    assert "Embedded" in ADMIN_TAGS
    assert "Embedded" not in ACT_TAGS
    assert "Embedded" not in BODY_TYPE_TAGS
