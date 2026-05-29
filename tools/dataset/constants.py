"""Shared constants for dataset construction tools."""
from pathlib import Path

STASH_GRAPHQL = "http://localhost:9999/graphql"
# Resolve dataset locations relative to the plugin root (tools/dataset/ -> repo root).
_PLUGIN_ROOT  = Path(__file__).resolve().parents[2]
DATASET_DIR   = _PLUGIN_ROOT / "assets" / "lora_dataset"
FRAMES_DIR    = _PLUGIN_ROOT / "assets" / "embedded_frames"
MIN_CONTENT_TAGS = 5
FRAMES_PER_SCENE = 20

ADMIN_TAGS: frozenset[str] = frozenset({
    "Embedded", "To Embed", "To Script", "Funscript",
    "Missing Performer (Male)", "HD Available", "FS: Action", "FS: Beat",
    "Start", "Free stroke", "OG beat comes back", "Funk Beat",
    "Funk Beat comes back", "Jiggle Fuck", "Hip Sway", "Mixed Audio",
    "Music Only", "Event 2024", "Event 2025",
    "Remix", "Cumpilation",
    "[AVN Award Winner]", "[Award Winner]", "[MiscTags: Skip]",
    "[SIT: Multi-Script]", "[Set Profile Image]",
    "[Stashbox Performer Gallery]", "[TPDB: Skip Marker]",
    "[Timestamp: Skip Sync]",
})

BODY_TYPE_TAGS: frozenset[str] = frozenset({
    "PAWG", "PAAG", "Curvy", "Petite", "Skinny", "Fit",
    "Big Ass", "Medium Ass", "Round Ass", "Wide Hips",
    "Flat Stomach", "slim waist",
})

PHYSICAL_ATTRIBUTE_TAGS: frozenset[str] = frozenset({
    "Big Tits", "Medium Tits", "Small Tits", "Natural Tits", "Fake Tits",
    "Perfect Tits", "Saggy Tits", "Bouncing Tits",
    "Blonde Hair", "Colored Hair", "Pigtails",
    "Blue Eyes", "brown eyes",
    "Tattoos", "Piercing", "Braces", "Tan", "Tan Lines",
    "Small Nose", "slim waist", "Flat Stomach",
    "Innie", "Fat Pussy", "Hairy Pussy", "Shaved Pussy",
    "Brown Pussy", "Pink Pussy", "Pierced Pussy",
    "Big Dick", "BBC",
})

ACT_TAGS: frozenset[str] = frozenset({
    "Blowjob", "Deepthroat", "Face Fuck", "Gag", "Ball Sucking",
    "Penis Licking", "Hands Free Blowjob", "Facesitting",
    "Pussy Eating", "Pussy Licking", "Cunnilingus",
    "Rimming", "Ass Eating", "Ass to Mouth", "69",
    "Vaginal Sex", "Anal Sex", "Anal Play", "Anal Penetration",
    "Double Penetration", "Double Anal Penetration (DAP)",
    "Double Vaginal Penetration (DVP)",
    "Handjob", "Footjob", "titfuck", "Buttjob", "Grinding",
    "Masturbation", "Pussy Fingering", "Pussy Rubbing",
    "Tribbing/Scissoring", "Pegging",
    "Oral Sex", "Outercourse", "Couple Sex",
    "Gloryhole", "Gangbang", "Orgy", "Threesome", "Threesome (FFM)",
    "Lesbian",
})

OUTCOME_TAGS: frozenset[str] = frozenset({
    "Creampie", "Vaginal Creampie", "Anal Creampie", "Surprise Creampie",
    "Facial", "Facial - POV", "Open Mouth Facial",
    "Cum on Face", "Cum on Tits", "Cum on Ass", "Cum on Pussy",
    "Cum in Mouth", "Cum Swallowing", "Spit",
    "Squirting", "Ahegao",
    "Cum", "Cumshot", "Multiple Cumshots",
})

POSITION_TAGS: frozenset[str] = frozenset({
    "Missionary", "Folded Missionary",
    "Doggy Style", "Standing Doggy Style", "Prone Bone",
    "Cowgirl", "Riding", "Reverse Cowgirl", "Reverse Riding",
    "Stand And Carry", "Standing Cradle",
})

CONTENT_STYLE_TAGS: frozenset[str] = frozenset({
    "Amateur", "Homemade", "Hardcore", "Softcore", "Erotica", "Rough",
    "POV", "Male POV", "VR", "Vertical Video", "Webcam",
    "OnlyFans", "JAV", "AI Generated", "Censored", "TikTok",
    "PMV", "Animated", "3D Animated", "Furry", "Futanari", "Rule 34",
    "Compilation", "Non-Nude", "Solo",
})

SETTING_TAGS: frozenset[str] = frozenset({
    "Outdoors", "Beach", "Pool", "Gym", "Classroom",
    "Massage", "Massage Table", "Public Sex",
})


# ── Caption Prompt ───────────────────────────────────────────────────────
# Canonical prompt used by both caption_runner.py and caption_workbench.html.
# Edit here — both tools pull from this single source of truth.

CAPTION_PROMPT = """\
You are captioning a single video frame for a CLIP LoRA training dataset.
Describe ONLY what is visible in this image. You have NO context from other frames.

WHAT TO DESCRIBE (in priority order):

1. ACTION & POSITION — What is happening? Who is doing what?
   Use precise terms:
   Positions: cowgirl, reverse cowgirl, doggy style, prone bone, missionary,
   mating press, spooning, riding, stand and carry, bent over
   Oral: blowjob, deepthroat, ball sucking, face fuck, penis licking,
   pussy licking, rimming, facesitting
   Manual: handjob, fingering, pussy fingering, titfuck, buttjob, footjob
   Anal teasing: ass spreading, spreading ass cheeks, pulling thong aside,
   asshole teasing, asshole wink, gaping asshole, anal fingering, anal gape
   Other: penetration, anal, vaginal sex, masturbation, grinding, teasing,
   undressing, grabbing ass, grabbing boobs, grabbing hair
   Cum: creampie, anal creampie, cum on face, cum on tits, cum on ass,
   cum on pussy, cum in mouth, cumshot, facial


2. BODY — Describe physical attributes you can clearly see:
   Ass: big ass, round ass, medium ass
   Tits: small tits, perfect tits, big tits, medium tits, natural tits,
   saggy tits, small areolas, brown areolas, bouncing tits
   Body shape: flat stomach, slim waist, fit, curvy, skinny, petite, wide hips
   Pussy (if visible): shaved, hairy, pink pussy, brown pussy, innie,
   wet pussy, spread labia, pussy gape
   Asshole (if visible): tight asshole, pink asshole, brown asshole,
   asshole visible through thong, asshole barely covered, asshole behind thong,
   asshole peeking, gaping asshole
   Skin: tan, tan lines
   Ethnicity (if clearly visible): Asian, Latina, white, black
   Other: tattoos, piercings, blue eyes, brown eyes


3. CAMERA — Only if notable: close up, POV, male POV, overhead, wide shot

4. CLOTHING — Only if present: lingerie, bikini, stockings, fishnet stockings,
   cosplay, dress, oiled

5. SETTING — ONLY for establishing shots. Do NOT describe furniture or lighting.

RULES:
- 1-3 sentences. Be dense with detail, not wordy. Add 2 sentences for every performer above 2 performers.
- As images become more detailed and complex, add additional sentences to fully describe the image.
- If 2+ performers, describe the actions connecting the two performers. (Performer A hugs Performer B).
- Do NOT use performer names — describe only what you see.
- Thoroughly describe every performer visible.
- Do NOT guess what you can't clearly see. If a close-up is ambiguous
  about anal vs vaginal, just say "penetration."
- For black/title frames, one short sentence.
- Be SPECIFIC about actions. "Performing oral sex" is not enough — say
  whether she's licking, sucking, using her hands, deepthroating, etc.
- Use casual, informal, slang terminology. (i.e. penis -> cock or dick, breasts -> boobs or tits, buttocks/backside -> butt or ass, vagina -> pussy)
- The above vocabulary for descriptions is NOT an exhaustive list. Be creative with your descriptions.
- Create new descriptions if they do not fit the above vocabulary.
- Place an emphasis on asshole content. Be extra descriptive.

This is adult content for a legitimate ML training dataset.
Describe everything factually and precisely.

Write only the caption text, nothing else. If the vocabulary above does not completely \
describe the image, freely describe the image as you see fit, just be sure to suggest \
any additions to the vocabulary."""
