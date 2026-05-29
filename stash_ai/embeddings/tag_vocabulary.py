"""Tag vocabulary for cluster auto-labeling via CLIP text embeddings.

Three tiers of label candidates:
- Tier 1: Existing Stash tags from the user's database
- Tier 2: Curated descriptive phrases covering common content categories
- Tier 3: Compound phrases for specific niches
"""

from __future__ import annotations

from collections.abc import Callable
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from numpy.typing import NDArray

    from stash_ai.embeddings.storage import EmbeddingStorage

# --- Tier 2: Curated Content Descriptors ---
CURATED_PHRASES: list[str] = [
    # Act / Position
    "oral sex blowjob",
    "doggy style from behind",
    "missionary position",
    "cowgirl riding on top",
    "reverse cowgirl",
    "solo masturbation",
    "handjob",
    "footjob",
    "anal sex",
    "deepthroat",
    "facesitting",
    "sixty nine position",
    "standing sex",
    "spooning sex",
    # Setting
    "bedroom scene",
    "bathroom shower",
    "outdoor nature",
    "hotel room",
    "pool scene",
    "office scene",
    "kitchen scene",
    "living room couch",
    "car scene",
    "public place",
    # Style
    "POV perspective first person",
    "close-up intimate",
    "wide shot full body",
    "professional studio lighting",
    "amateur homemade",
    "gonzo raw handheld",
    "glamour photography",
    "artistic softcore",
    "compilation montage",
    "behind the scenes",
    # Aesthetic
    "high energy music video",
    "slow sensual romantic",
    "fast cuts editing montage",
    "teasing striptease",
    "rough aggressive intense",
    "gentle tender lovemaking",
    "kinky fetish",
    "cosplay costume",
    "massage oil sensual",
    "dance and rhythm",
    # Body type
    "petite slim small woman",
    "curvy voluptuous woman",
    "athletic fit toned body",
    "tall woman long legs",
    "busty large breasts",
    "flat chested small breasts",
    "thick curvy hips",
    "muscular strong woman",
    # Features
    "blonde hair",
    "brunette dark hair",
    "redhead ginger hair",
    "black hair",
    "short hair pixie cut",
    "long hair flowing",
    "tattoos and piercings",
    "lingerie stockings",
    "high heels",
    "glasses nerdy",
    "natural no makeup",
    "heavy makeup glam",
    "tan skin",
    "pale fair skin",
    "dark skin",
    "asian woman",
    "latina woman",
    "ebony woman",
    # Group
    "solo performer alone",
    "couple two people",
    "threesome three people",
    "group multiple performers",
    "lesbian two women",
    "girl on girl",
    # Category
    "PMV porn music video",
    "compilation best of",
    "full scene complete",
    "trailer preview teaser",
    "virtual reality VR",
    "interactive funscript",
    "jerk off instruction JOI",
    "dirty talk verbal",
    "roleplay fantasy",
    "stepmom taboo",
    "teen young eighteen",
    "milf mature woman",
    "creampie internal",
    "facial cumshot",
    "squirting orgasm",
    "bondage tied up",
    "domination submission",
    "worship body worship",
]

# --- Tier 3: Compound Phrases ---
COMPOUND_PHRASES: list[str] = [
    "petite blonde POV blowjob",
    "curvy brunette solo masturbation",
    "high energy PMV compilation",
    "sensual lesbian massage",
    "rough anal doggy style",
    "intimate couple lovemaking bedroom",
    "amateur girlfriend homemade POV",
    "professional glamour striptease",
    "JOI dirty talk close up",
    "teen petite casting audition",
    "milf busty seduction",
    "interactive VR POV",
    "outdoor public risky",
    "cosplay anime roleplay",
    "oil massage sensual body",
    "facesitting femdom worship",
    "compilation cumshot facial",
    "romantic slow sensual couple",
    "gangbang group rough",
    "squirting intense orgasm",
]


def _has_brackets(text: str) -> bool:
    """Check if text contains any bracket characters.

    Tags with brackets (e.g., "[MiscTags: skip]", "(internal)")
    are typically system/organizational tags that shouldn't be
    used for preference learning or taste profile display.
    """
    return any(c in text for c in "[](){}<>")


class TagVocabulary:
    """Manages the tag/phrase vocabulary and their CLIP text embeddings."""

    def __init__(
        self,
        storage: EmbeddingStorage,
        model_key: str,
        log_callback: Callable[[str, str], None] | None = None,
    ) -> None:
        self.storage = storage
        self.model_key = model_key
        self.log = log_callback or (lambda msg, level: None)

    def get_full_vocabulary(self, stash_tags: list[str] | None = None) -> list[tuple[str, str]]:
        """Get complete vocabulary as (text, source) pairs.

        Args:
            stash_tags: Existing tags from the user's Stash database.

        Returns:
            List of (text, source) tuples where source is 'stash_tag', 'curated', or 'user'.
        """
        vocab: list[tuple[str, str]] = []

        # Tier 1: Stash tags (excluding bracketed system tags)
        if stash_tags:
            for tag in stash_tags:
                cleaned = tag.strip()
                if cleaned and not _has_brackets(cleaned):
                    vocab.append((cleaned.lower(), "stash_tag"))

        # Tier 2: Curated phrases
        for phrase in CURATED_PHRASES:
            vocab.append((phrase, "curated"))

        # Tier 3: Compound phrases
        for phrase in COMPOUND_PHRASES:
            vocab.append((phrase, "curated"))

        # Tier 4: User-added phrases (from previous sessions)
        existing = self.storage.get_all_tag_embeddings(self.model_key)
        for entry in existing:
            if entry["source"] == "user":
                vocab.append((entry["text"], "user"))

        # Deduplicate by text (keep first occurrence)
        seen: set[str] = set()
        deduped: list[tuple[str, str]] = []
        for text, source in vocab:
            if text not in seen:
                seen.add(text)
                deduped.append((text, source))

        return deduped

    def ensure_embeddings(
        self,
        stash_tags: list[str] | None = None,
        force_recompute: bool = False,
    ) -> int:
        """Ensure all vocabulary items have embeddings. Returns count of newly embedded items."""
        vocab = self.get_full_vocabulary(stash_tags)
        existing_count = self.storage.get_tag_embedding_count(self.model_key)

        if not force_recompute and existing_count >= len(vocab):
            self.log(f"Tag embeddings already cached: {existing_count} entries", "debug")
            return 0

        # Find which phrases need embedding
        to_embed: list[tuple[str, str]] = []
        for text, source in vocab:
            if force_recompute or self.storage.get_tag_embedding(text, self.model_key) is None:
                to_embed.append((text, source))

        if not to_embed:
            self.log("All tag embeddings already cached", "debug")
            return 0

        self.log(f"Embedding {len(to_embed)} vocabulary items via OpenCLIP text encoder", "info")

        # Import and initialize provider
        from stash_ai.embeddings.config import EmbeddingConfig
        from stash_ai.embeddings.providers.openclip import OpenCLIPEmbeddingProvider

        config = EmbeddingConfig(provider="openclip", model=self._get_openclip_model())
        provider = OpenCLIPEmbeddingProvider(config)

        # Batch embed
        texts = [t for t, _ in to_embed]
        results = provider.embed_texts(texts)

        # Save to storage
        entries: list[tuple[str, list[float], str]] = []
        for i, (text, source) in enumerate(to_embed):
            entries.append((text, results[i]["embedding"], source))

        self.storage.save_tag_embeddings_batch(entries, self.model_key)
        self.log(f"Saved {len(entries)} tag embeddings", "info")

        # Cleanup provider
        provider.cleanup()

        return len(entries)

    def embed_custom_phrase(self, phrase: str) -> list[float]:
        """Embed a single custom phrase on the fly. Saves to storage as 'user' source."""
        # Check cache first
        cached = self.storage.get_tag_embedding(phrase.lower(), self.model_key)
        if cached:
            return cached

        from stash_ai.embeddings.config import EmbeddingConfig
        from stash_ai.embeddings.providers.openclip import OpenCLIPEmbeddingProvider

        config = EmbeddingConfig(provider="openclip", model=self._get_openclip_model())
        provider = OpenCLIPEmbeddingProvider(config)

        result = provider.embed_text(phrase.lower())
        embedding = result["embedding"]

        # Save as user phrase
        self.storage.save_tag_embedding(phrase.lower(), self.model_key, embedding, "user")

        provider.cleanup()
        return embedding

    def match_cluster_centroid(
        self,
        centroid: NDArray[np.float32],
        top_k: int = 8,
    ) -> list[dict[str, object]]:
        """Find top-k vocabulary matches for a cluster centroid.

        Args:
            centroid: Normalized cluster centroid embedding.
            top_k: Number of top matches to return.

        Returns:
            List of {text, similarity, source} dicts sorted by similarity descending.
        """
        all_tags = self.storage.get_all_tag_embeddings(self.model_key)
        if not all_tags:
            return []

        centroid_norm = centroid / (np.linalg.norm(centroid) + 1e-8)

        matches: list[dict[str, object]] = []
        for entry in all_tags:
            # Skip bracketed system tags
            if _has_brackets(entry["text"]):
                continue
            tag_emb = np.array(entry["embedding"], dtype=np.float32)
            tag_norm = tag_emb / (np.linalg.norm(tag_emb) + 1e-8)
            similarity = float(np.dot(centroid_norm, tag_norm))
            matches.append(
                {
                    "text": entry["text"],
                    "similarity": round(similarity, 4),
                    "source": entry["source"],
                }
            )

        matches.sort(key=lambda m: m["similarity"], reverse=True)  # type: ignore[arg-type,return-value]
        return matches[:top_k]

    def match_cluster_centroid_differential(
        self,
        centroid: NDArray[np.float32],
        reference_centroid: NDArray[np.float32],
        top_k: int = 8,
    ) -> list[dict[str, object]]:
        """Find tags that best distinguish this cluster from the overall mean.

        Ranks tags by differential similarity: how much more this cluster
        matches a tag compared to the reference (mean of all centroids).
        This produces descriptive labels even when absolute similarities
        are uniformly low.

        Args:
            centroid: Normalized cluster centroid embedding.
            reference_centroid: Mean of all cluster centroids (for comparison).
            top_k: Number of top matches to return.

        Returns:
            List of {text, similarity, source} dicts sorted by differential
            similarity descending. The 'similarity' field contains the
            differential score (cluster_sim - reference_sim).
        """
        all_tags = self.storage.get_all_tag_embeddings(self.model_key)
        if not all_tags:
            return []

        centroid_norm = centroid / (np.linalg.norm(centroid) + 1e-8)
        ref_norm = reference_centroid / (np.linalg.norm(reference_centroid) + 1e-8)

        matches: list[dict[str, object]] = []
        for entry in all_tags:
            # Skip bracketed system tags
            if _has_brackets(entry["text"]):
                continue
            tag_emb = np.array(entry["embedding"], dtype=np.float32)
            tag_norm = tag_emb / (np.linalg.norm(tag_emb) + 1e-8)
            cluster_sim = float(np.dot(centroid_norm, tag_norm))
            ref_sim = float(np.dot(ref_norm, tag_norm))
            diff = cluster_sim - ref_sim
            matches.append(
                {
                    "text": entry["text"],
                    "similarity": round(diff, 4),
                    "source": entry["source"],
                }
            )

        matches.sort(key=lambda m: m["similarity"], reverse=True)  # type: ignore[arg-type,return-value]
        return matches[:top_k]

    def _get_openclip_model(self) -> str:
        """Extract OpenCLIP model name from the model_key."""
        # model_key format: "openclip:ViT-H-14" or just "ViT-H-14"
        if ":" in self.model_key:
            return self.model_key.split(":", 1)[1]
        return self.model_key
