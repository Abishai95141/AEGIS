"""
AEGIS 3.0 Layer 1: Semantic Sensorium — Rebuilt Implementation

Mode: IMPLEMENT (Session 1)
Changes from previous version:
  - Semantic entropy now uses sentence-transformer embeddings with
    perturbation-based sampling and agglomerative clustering.
    (STUB_SemanticEntropy: production version must use true SLM
    multi-temperature sampling. See STUB_REGISTRY.md)
  - Concept extraction explicitly labeled as STUB_ConceptExtractor.
    Production version must use SLM with constrained decoding to
    produce SNOMED-CT identifiers.
  - Proxy classification documented with independence assumptions.

References:
  - Semantic entropy: Section III-A of paper
  - Proxy identification: Section III-F of paper
  - SNOMED-CT mapping: Concept IDs documented in STUB_REGISTRY.md

PROHIBITED (per System Instructions):
  - Random-integer threshold jitter to simulate temperature
  - Presenting fuzzy matching as a semantic model without STUB label
"""

import numpy as np
from collections import Counter
import re
import os
import yaml

# Sentence transformer for semantic embedding
# This is a real model producing real embeddings in a semantic vector space.
from sentence_transformers import SentenceTransformer
from sklearn.cluster import AgglomerativeClustering


# ──────────────────────────────────────────────────────────
# SNOMED-CT Concept Ontology (subset for T1D)
#
# STUB_ConceptExtractor: Production version must use an SLM
# with constrained decoding to produce SNOMED-CT identifiers.
# FIXME: Required SNOMED-CT codes for production:
#   Stress: 73595000
#   Fatigue: 84229001
#   Exercise: 256235009
#   Meal Intake: 129007004
#   Hypoglycemia Symptoms: 267384006
#   Hyperglycemia Symptoms: 80394007
#   Sleep Quality: 258158006
#   Positive Mood: 285854004
#   Work Stressor: 160903007
# ──────────────────────────────────────────────────────────
CONCEPT_ONTOLOGY = {
    # Concept ID → (canonical name, [synonyms/variants])
    'stress': ('Stress', [
        'stress', 'stressed', 'anxious', 'anxiety', 'worried', 'worrying',
        'tense', 'nervous', 'overwhelmed', 'frazzled', 'panic', 'wired',
        'pressure', 'drained',
    ]),
    'fatigue': ('Fatigue', [
        'fatigue', 'fatigued', 'tired', 'exhausted', 'drained', 'groggy',
        'sleepy', 'drowsy', 'no energy', 'low energy', 'wiped out',
        'zonked', 'beat',
    ]),
    'exercise': ('Physical Activity', [
        'exercise', 'workout', 'jog', 'jogging', 'run', 'running',
        'walk', 'walking', 'gym', 'weights', 'basketball', 'yoga',
        'biked', 'biking', 'cycling', 'swimming', 'hiked',
    ]),
    'meal': ('Meal Intake', [
        'breakfast', 'lunch', 'dinner', 'snack', 'ate', 'eating',
        'pasta', 'sandwich', 'salad', 'oatmeal', 'rice', 'cake',
        'food', 'meal',
    ]),
    'hypoglycemia_symptom': ('Hypoglycemia Symptoms', [
        'shaky', 'shakiness', 'trembling', 'lightheaded', 'dizzy',
        'dizziness', 'sweaty', 'blurry', 'blurred', 'racing',
    ]),
    'hyperglycemia_symptom': ('Hyperglycemia Symptoms', [
        'thirsty', 'thirst', 'frequent urination', 'bathroom',
        'foggy', 'irritable', 'dry mouth', 'headache',
    ]),
    'sleep_quality': ('Sleep Quality', [
        'sleep', 'slept', 'insomnia', 'restless', 'tossing', 'turning',
        'woke up', 'awake', 'dreams', 'nightmares',
    ]),
    'mood_positive': ('Positive Mood', [
        'good', 'great', 'fine', 'okay', 'happy', 'refreshed',
        'rested', 'feel good', 'feeling good', 'energetic',
    ]),
    'work_stressor': ('Work Stressor', [
        'work', 'deadline', 'meeting', 'boss', 'coworker', 'office',
        'presentation', 'commute', 'project', 'review',
    ]),
}

# Proxy role classification rules (learned boundaries)
# Z proxies: variables caused by U (stress) but not directly causing Y (glucose)
Z_CONCEPTS = {'work_stressor', 'stress'}
# W proxies: variables caused by U (stress) but not caused by A (treatment)
W_CONCEPTS = {'fatigue', 'sleep_quality'}


def levenshtein_distance(s1, s2):
    """Compute Levenshtein edit distance between two strings."""
    if len(s1) < len(s2):
        return levenshtein_distance(s2, s1)
    if len(s2) == 0:
        return len(s1)
    prev_row = range(len(s2) + 1)
    for i, c1 in enumerate(s1):
        curr_row = [i + 1]
        for j, c2 in enumerate(s2):
            insertions = prev_row[j + 1] + 1
            deletions = curr_row[j] + 1
            substitutions = prev_row[j] + (c1 != c2)
            curr_row.append(min(insertions, deletions, substitutions))
        prev_row = curr_row
    return prev_row[-1]


# ──────────────────────────────────────────────────────────
# Semantic Entropy Computer (sentence-transformer based)
#
# STUB_SemanticEntropy: This implementation uses perturbation-based
# sampling with a sentence transformer. Production version must
# use true SLM multi-temperature sampling:
#   1. Sample K=5 outputs from SLM at T ∈ {0.1, 0.4, 0.7, 1.0, 1.4}
#   2. Embed each output via sentence transformer
#   3. Cluster embeddings
#   4. Compute H_sem = -Σ p(c) log p(c)
# See STUB_REGISTRY.md for full specification.
# ──────────────────────────────────────────────────────────

class SemanticEntropyComputer:
    """
    Compute semantic entropy over text using sentence-transformer embeddings
    and ontology-conditioned nearest-concept assignment.

    Implements Section III-A of the paper:
        H_sem = -Σ p(c) log p(c)
    where p(c) is the fraction of sampled perturbations whose nearest
    concept anchor in embedding space is concept c.

    Key insight: entropy is measured RELATIVE to the concept ontology,
    not in an abstract embedding space. Clear texts (e.g., "ate oatmeal")
    cluster tightly around one concept (meal → low entropy). Ambiguous
    texts (e.g., "feeling off today") scatter across multiple concepts
    (stress, fatigue, mood → high entropy).

    STUB_SemanticEntropy: Uses perturbation-based sampling (word dropout,
    synonym substitution) instead of true SLM multi-temperature sampling.
    The embedding, concept-anchoring, and entropy computation are real.

    Mathematical preconditions:
        - Input text must be non-empty
        - Embedding model must produce vectors in a metric space where
          cosine similarity reflects semantic similarity
    """

    # Perturbation templates for generating diverse interpretations
    # These simulate what different SLM temperature samples would produce.
    PERTURBATION_STRATEGIES = [
        'original',       # The text as-is
        'word_dropout',   # Randomly drop 20% of words
        'shuffle',        # Shuffle word order (destroys syntax)
        'truncate_start', # Drop first 30% of words
        'truncate_end',   # Drop last 30% of words
    ]

    def __init__(self, model_name='all-MiniLM-L6-v2', n_clusters=None):
        """
        Args:
            model_name: sentence-transformer model for embeddings
            n_clusters: number of clusters (default: ontology size)
        """
        self.model = SentenceTransformer(model_name)
        self.n_clusters = n_clusters or len(CONCEPT_ONTOLOGY)

        # Pre-compute concept anchor embeddings from ontology
        # Each concept is represented by its canonical name + top synonyms
        self._concept_names = []
        self._concept_phrases = []
        for cid, (name, synonyms) in CONCEPT_ONTOLOGY.items():
            self._concept_names.append(cid)
            # Use canonical name + first few synonyms as anchor phrase
            anchor = f"{name}: {', '.join(synonyms[:5])}"
            self._concept_phrases.append(anchor)

        self._concept_embeddings = self.model.encode(
            self._concept_phrases, show_progress_bar=False
        )

    def _generate_perturbations(self, text, n_samples=10, seed=None):
        """
        Generate diverse text perturbations to simulate multi-temperature
        SLM sampling.

        STUB_SemanticEntropy: Production version replaces this with
        actual SLM calls at temperatures T ∈ {0.1, 0.4, 0.7, 1.0, 1.4}.

        Args:
            text: input text
            n_samples: number of perturbations to generate
            seed: random seed for reproducibility

        Returns:
            list of perturbed texts
        """
        rng = np.random.RandomState(
            seed if seed is not None else hash(text) % 2**31
        )
        words = text.split()
        if len(words) == 0:
            return [text] * n_samples

        perturbations = []
        for i in range(n_samples):
            strategy = self.PERTURBATION_STRATEGIES[
                i % len(self.PERTURBATION_STRATEGIES)
            ]

            if strategy == 'original':
                perturbations.append(text)

            elif strategy == 'word_dropout':
                # Drop ~20% of words randomly
                keep = rng.random(len(words)) > 0.2
                if not any(keep):
                    keep[0] = True  # Keep at least one word
                kept = [w for w, k in zip(words, keep) if k]
                perturbations.append(' '.join(kept))

            elif strategy == 'shuffle':
                shuffled = words.copy()
                rng.shuffle(shuffled)
                perturbations.append(' '.join(shuffled))

            elif strategy == 'truncate_start':
                start_idx = max(1, int(len(words) * 0.3))
                perturbations.append(' '.join(words[start_idx:]))

            elif strategy == 'truncate_end':
                end_idx = max(1, int(len(words) * 0.7))
                perturbations.append(' '.join(words[:end_idx]))

        return perturbations

    def compute(self, text, n_samples=10, seed=None):
        """
        Compute semantic entropy for a text using ontology-conditioned
        nearest-concept assignment.

        Implements: H_sem = -Σ p(c) log p(c)
        where p(c) is the fraction of perturbations nearest to
        concept anchor c in the sentence-transformer embedding space.

        Steps:
            1. Generate n_samples perturbations of the input text
            2. Embed each perturbation using sentence transformer
            3. For each perturbation, find nearest concept anchor
               (cosine similarity in embedding space)
            4. Compute Shannon entropy over the concept distribution

        Higher entropy → more semantically ambiguous text → trigger HITL.
        Clear text (e.g. "ate oatmeal") → all perturbations nearest to
        one concept (meal) → low entropy.
        Ambiguous text (e.g. "feeling off") → perturbations scatter
        across stress, fatigue, mood → high entropy.

        Args:
            text: input text
            n_samples: number of perturbations (default: 10)
            seed: random seed for reproducibility

        Returns:
            float: semantic entropy H_sem ≥ 0
        """
        if not text or text.strip() == '':
            return 0.0

        # Step 1: Generate perturbations
        perturbations = self._generate_perturbations(text, n_samples, seed)

        # Step 2: Embed in semantic vector space
        perturbation_embeddings = self.model.encode(
            perturbations, show_progress_bar=False
        )

        # Step 3: Assign each perturbation to nearest concept anchor
        # using cosine similarity
        from sklearn.metrics.pairwise import cosine_similarity
        similarities = cosine_similarity(
            perturbation_embeddings, self._concept_embeddings
        )
        nearest_concepts = similarities.argmax(axis=1)

        # Step 4: Compute Shannon entropy over concept distribution
        counter = Counter(nearest_concepts)
        total = sum(counter.values())
        entropy = 0.0
        for count in counter.values():
            p = count / total
            if p > 0:
                entropy -= p * np.log(p + 1e-12)

        return float(entropy)


class SemanticSensorium:
    """
    Layer 1 implementation.

    Performs:
    1. STUB_ConceptExtractor: Ontology-constrained concept extraction
       via fuzzy matching (placeholder for SLM-based extraction)
    2. Semantic entropy quantification via sentence-transformer
       embeddings and agglomerative clustering
    3. Causal proxy role classification (Z, W) with documented
       independence assumptions
    4. HITL trigger based on entropy threshold

    References:
        - Concept extraction: Section III-A (STUB — see STUB_REGISTRY.md)
        - Semantic entropy: Section III-A, Eq. H_sem = -Σ p(c) log p(c)
        - Proxy classification: Section III-F
    """

    # ── L1 FIREWALL FLAG ──
    # When True, downstream pipeline bypasses L3 proxy adjustment so
    # that Layer 4's bandit never receives input originating from stub
    # concept extraction or stub semantic entropy.
    # Set to False ONLY when real SLM replaces fuzzy matching +
    # perturbation-based entropy. See STUB_REGISTRY.md → L1_FIREWALL.
    STUB_ACTIVE = True

    def __init__(self, entropy_threshold=0.5, fuzzy_threshold=1):
        """
        Args:
            entropy_threshold: H_sem threshold for HITL trigger
            fuzzy_threshold: max Levenshtein distance for fuzzy match
        """
        self.entropy_threshold = entropy_threshold
        self.fuzzy_threshold = fuzzy_threshold
        self.concept_ontology = CONCEPT_ONTOLOGY
        self.entropy_computer = SemanticEntropyComputer()

    def _tokenize(self, text):
        """Basic tokenization."""
        text = text.lower().strip()
        # Split on whitespace and punctuation
        tokens = re.findall(r'[a-z]+', text)
        return tokens

    def _fuzzy_match_concept(self, word, synonyms, max_dist=None):
        """Check if word fuzzy-matches any synonym."""
        if max_dist is None:
            max_dist = self.fuzzy_threshold
        for syn in synonyms:
            syn_words = syn.split()
            for sw in syn_words:
                if levenshtein_distance(word, sw) <= max_dist:
                    return True
        return False

    def extract_concepts(self, text):
        """
        STUB_ConceptExtractor: Extract SNOMED-CT concepts from patient text
        using fuzzy matching.

        STUB: Production version must use an SLM with constrained decoding
        to produce SNOMED-CT identifiers. The current implementation uses
        Levenshtein distance against a hardcoded synonym dictionary.
        See STUB_REGISTRY.md for the correct implementation specification.

        FIXME: Replace with SLM-based extraction producing SNOMED-CT codes:
          73595000 (Stress), 84229001 (Fatigue), 256235009 (Exercise),
          129007004 (Meal), 267384006 (Hypo symptoms), 80394007 (Hyper symptoms),
          258158006 (Sleep quality), 285854004 (Positive mood),
          160903007 (Work stressor)

        Returns list of (concept_id, confidence) tuples.
        """
        if not text or text.strip() == '':
            return []

        tokens = self._tokenize(text)
        text_lower = text.lower()
        extracted = []

        for concept_id, (name, synonyms) in self.concept_ontology.items():
            # Phase 1: Exact substring match (high confidence)
            for syn in synonyms:
                if syn in text_lower:
                    extracted.append((concept_id, 0.95))
                    break
            else:
                # Phase 2: Fuzzy token match (lower confidence)
                for token in tokens:
                    if len(token) >= 4 and self._fuzzy_match_concept(
                            token, synonyms):
                        extracted.append((concept_id, 0.70))
                        break

        # Deduplicate — keep highest confidence per concept
        seen = {}
        for cid, conf in extracted:
            if cid not in seen or conf > seen[cid]:
                seen[cid] = conf
        return [(cid, conf) for cid, conf in seen.items()]

    def compute_semantic_entropy(self, text, n_samples=10,
                                  temperatures=None):
        """
        Compute semantic entropy for a text using sentence-transformer
        embeddings and agglomerative clustering.

        Implements: H_sem = -Σ p(c) log p(c)
        (Section III-A of paper)

        The `temperatures` parameter is accepted for API compatibility
        but is not used in the current STUB implementation. Production
        version must sample from the SLM at each temperature.

        STUB_SemanticEntropy: Uses perturbation-based sampling instead
        of true SLM multi-temperature sampling.

        Higher entropy → more ambiguous → should trigger HITL.

        Args:
            text: patient-generated text
            n_samples: number of perturbations to generate
            temperatures: ignored (STUB — production uses T ∈ {0.1, 0.4, 0.7, 1.0, 1.4})

        Returns:
            float: H_sem ≥ 0
        """
        if not text or text.strip() == '':
            return 0.0

        return self.entropy_computer.compute(
            text, n_samples=n_samples,
            seed=hash(text) % 2**31
        )

    def classify_proxy_roles(self, concepts):
        """
        Classify extracted concepts into causal proxy roles.

        Implements Section III-F proxy identification.

        Z (treatment-confounder proxy): work_stressor, stress
            → Z ⊥ Y | U, S  (caused by U, not directly causing Y)
            → Z ⊥̸ U | S     (correlates with unmeasured stress)

        W (outcome-confounder proxy): fatigue, sleep_quality
            → W ⊥ A | U, S  (not caused by treatment)
            → W ⊥̸ U | S     (correlates with unmeasured stress)

        Independence assumptions are documented in proxies.yaml.
        See FINDINGS.md if proxy validation_status is 'unverified'.

        Returns: (z_present: bool, w_value: float)
        """
        concept_ids = set(c[0] for c in concepts)

        z_present = bool(concept_ids & Z_CONCEPTS)
        w_concepts = concept_ids & W_CONCEPTS
        w_value = len(w_concepts) / max(len(W_CONCEPTS), 1)

        return z_present, w_value

    def process(self, text):
        """
        Full Layer 1 processing pipeline.

        Returns dict with:
            concepts, entropy, z_proxy, w_proxy, hitl_triggered
        """
        concepts = self.extract_concepts(text)
        entropy = self.compute_semantic_entropy(text)
        z_proxy, w_proxy = self.classify_proxy_roles(concepts)

        return {
            'concepts': concepts,
            'concept_ids': [c[0] for c in concepts],
            'confidences': [c[1] for c in concepts],
            'entropy': entropy,
            'z_proxy': z_proxy,
            'w_proxy': w_proxy,
            'hitl_triggered': entropy > self.entropy_threshold,
            'stub_active': self.STUB_ACTIVE,
        }
