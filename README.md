# ✈️ Causal Graph Reasoning for Explainable ADREP Classification

> **Course:** 18-786 Intro to Deep Learning · Carnegie Mellon University Africa  
> **Team:** Whoopie Wanjiru (MSEAI27) · Theophilus Owiti (MSEAI27) · Ronnie Delyon (MSEAI26)

---

## 🧩 Project Overview

Civil aviation authorities worldwide struggle to manually classify safety reports under the ICAO **ADREP** (Accident/Incident Reporting) taxonomy, leading to inconsistent labeling, processing delays, and missed trends.

A prior system — the **CMU-Africa SDCPS pipeline** — automated this using a hybrid transformer + multi-LLM approach, achieving **92.96% accuracy** on 3,600 reports. However, **26.94% of reports** were conservatively labelled **"OTHER"** — cases where flat text classification failed to resolve implicit, multi-factor causal narratives.

This project builds a **causal refinement module** that applies specifically to those OTHER reports. The goal: re-express ambiguous narratives as **structured causal event tuples**, then use graph-based reasoning to attempt secondary ADREP reclassification.

---

## 🏗️ System Architecture (4-Layer Pipeline)

```
Raw Narrative (classified "OTHER" by SDCPS)
        │
        ▼
┌───────────────────────────────────────┐
│  Layer 1 — Event Extraction (NER)     │  ← current work
│  Extract: ACTOR, SYSTEM, PHASE,       │
│           TRIGGER, OUTCOME            │
└───────────────────────────────────────┘
        │
        ▼
┌───────────────────────────────────────┐
│  Layer 2 — Causal Graph Construction  │  ← next step
│  Assemble entities into a DAG         │
│  (e.g., Ice → Engine Stall → Divert) │
└───────────────────────────────────────┘
        │
        ▼
┌───────────────────────────────────────┐
│  Layer 3 — Graph Reasoning            │  ← next step
│  GAT / HGT message passing           │
│  → "incident signature" embedding     │
└───────────────────────────────────────┘
        │
        ▼
┌───────────────────────────────────────┐
│  Layer 4 — ADREP Reclassification     │  ← next step
│  Map embedding → ICAO ADREP class     │
│  If confidence < threshold → keep OTHER│
└───────────────────────────────────────┘
```

---

## ✅ Current Progress: Layer 1 — Event Extraction

### What We Extract

For each aviation safety narrative, we extract a structured event tuple:

| Field | Source | Method |
|---|---|---|
| **ACTOR** | Narrative text | Fine-tuned BERT NER |
| **SYSTEM** | Narrative text | Fine-tuned BERT NER |
| **TRIGGER** | Narrative text | Fine-tuned BERT NER |
| **PHASE** | Existing `phase` column | Direct mapping (~98% coverage) |
| **OUTCOME** | Existing `events`/`events_6` columns | Direct mapping (100% coverage) |

### Example Output

```
Report #73
  Synopsis: During approach, crew received TCAS RA advisory...

  ACTOR:   ['Captain', 'First Officer']
  SYSTEM:  ['TCAS', 'autopilot']
  PHASE:   Approach
  TRIGGER: ['loss of situational awareness', 'communication breakdown']
  OUTCOME: ['AIRPROX', 'Near Miss']

  Causal Edges:
    Captain ──[caused]──▶ loss of situational awareness
    loss of situational awareness ──[led_to]──▶ AIRPROX
    TCAS ──[involved_in]──▶ loss of situational awareness
```

---

## 🧠 Approach: Silver-Label Fine-tuning

Since no gold-annotated NER labels exist for our dataset, we use a **silver-labeling** strategy:

1. **Build rule-based extractors** — regex + aviation domain dictionaries across 3 entity types
2. **Align spans to BERT tokens** — character-level span → BIO (Begin-Inside-Outside) tags
3. **Fine-tune BERT** on silver labels — model learns contextual patterns beyond the rules
4. **Run inference** on all 173 reports → export structured event tuples

#### Entity Pattern Categories

| Entity | Pattern Types |
|---|---|
| **ACTOR** | Flight crew roles (Captain, FO, PIC, SIC), ATC roles, pronominal pilot references |
| **SYSTEM** | Navigation (GPS, ILS, RNAV), safety systems (TCAS, GPWS), autopilot, comms, aircraft components |
| **TRIGGER** | Causal language (due to, failed to, resulted from), human factors (fatigue, distraction), environmental (turbulence, icing), mechanical (engine failure, malfunction) |

---

## 📊 Results

### Model Comparison: BERT-base vs SafeAeroBERT

We compared standard `bert-base-uncased` against `NASA-AIML/MIKA_SafeAeroBERT`, a domain-specific BERT pre-trained on aviation safety corpora.

| Metric | BERT-base | SafeAeroBERT | Winner |
|---|---|---|---|
| **Test F1** | 0.741 | **0.830** | ✅ SafeAeroBERT |
| **Precision** | 0.655 | **0.786** | ✅ SafeAeroBERT |
| **Recall** | 0.857 | **0.881** | ✅ SafeAeroBERT |
| **Test Loss** | 0.690 | **0.495** | ✅ SafeAeroBERT |
| **Best Val F1** | 0.731 (epoch 9) | **0.758** (epoch 10) | ✅ SafeAeroBERT |
| Parameters | 108.9M | 108.9M | — |

**SafeAeroBERT wins** — domain pre-training on aviation text gives it a meaningful edge on this task (~+9 F1 points).

### Manual Validation (3 reports, human-annotated)

| Entity | Precision | Recall | F1 |
|---|---|---|---|
| ACTOR | 1.00 | 0.50 | 0.67 |
| SYSTEM | 1.00 | 1.00 | **1.00** |
| TRIGGER | 1.00 | 1.00 | **1.00** |
| **Overall** | **1.00** | **0.67** | **0.80** |

> Quality rating: **EXCELLENT** — zero false positives; missed some ACTOR mentions (known limitation of implicit pronoun references).

### Training Configuration

```
Model:         bert-base-uncased / NASA-AIML/MIKA_SafeAeroBERT
Max Seq Len:   256 tokens (with overlapping chunking for longer texts)
Batch Size:    8
Learning Rate: 3e-5
Epochs:        10 (with early stopping, patience=3)
Warmup:        10% of total steps
Loss:          Weighted CrossEntropy (to handle O-class imbalance)
Optimizer:     AdamW (weight decay=0.01)
Data Split:    70% train / 15% val / 15% test
Dataset Size:  173 aviation safety reports
```

---

## 📁 Repository Structure

```
.
├── event_extraction_model.py       # Main training & inference pipeline (Layer 1)
├── event_extraction_brainstorm.py  # Early prototyping & experimentation
├── event_extraction_brainstorm.ipynb
├── manual_validation.py            # Human-annotation validation script
│
├── data_aviation.csv               # Aviation safety dataset (173 reports)
├── extracted_events.json           # Full extraction output (all 173 reports)
├── extracted_events_summary.csv    # Tabular extraction summary
├── manual_annotations.json         # Hand-labeled validation set
│
├── best_event_extractor.pt         # Best BERT-base model checkpoint
├── best_aerobert_event_extractor.pt # Best SafeAeroBERT model checkpoint
├── event_extractor_config.json     # Saved model config & label mapping
│
├── model_comparison_results.json   # BERT vs SafeAeroBERT comparison
├── validation_report.json          # Manual validation metrics
│
├── training_curves.png             # Loss & F1 training curves
├── model_comparison_bar.png        # Model comparison bar chart
├── model_comparison_curves.png     # Model comparison learning curves
├── entity_distribution.png         # Top extracted entities visualization
│
├── DeepLearning_Project_Proposal.pdf
├── multi_llm_aviation_incident_classification.pdf  # Prior SDCPS paper
├── adrep_project_critique.md
└── proposal_text.txt
```

---

## 🚀 Running the Code

### Requirements

```bash
pip install torch transformers seqeval pandas numpy matplotlib
```

### Run Event Extraction

```bash
python event_extraction_model.py
```

This will:
1. Load `data_aviation.csv`
2. Generate silver BIO labels via rule-based extraction
3. Fine-tune BERT for token classification
4. Evaluate on test set + run manual validation
5. Extract structured event tuples from all 173 reports
6. Save results to `extracted_events.json` and `extracted_events_summary.csv`

> ⚠️ **GPU strongly recommended.** Training on CPU is significantly slower. The best model checkpoints (`.pt` files) are already included if you want to skip training and run inference only.

---

## 🔭 What's Next (Layers 2–4)

- **Layer 2:** Build heterogeneous DAGs from extracted event tuples using `torch_geometric`
- **Layer 3:** Train a Graph Attention Network (GAT) or Heterogeneous Graph Transformer (HGT) for incident signature embeddings
- **Layer 4:** Map graph embeddings → ADREP category predictions; measure OTHER reduction rate

---

## 📚 References

1. Aviation Safety Network Database — https://aviation-safety.net/database/
2. Delyon, Manyara, Iliya, Gachomba. *Intelligent Aviation Safety Data Analysis using Transformer and Multi-LLM Consensus*. CMU-Africa Capstone, 2026.
3. FAA Aviation Safety Information Analysis — https://www.asias.faa.gov
4. NASA ASRS Database — https://asrs.arc.nasa.gov/search/database.html
5. New & Wallace. *Classifying Aviation Safety Reports using Supervised NLP*. Safety, 11(1):7, 2025.
6. NTSB Aviation Data — https://data.ntsb.gov/avdata
7. Wang et al. *Self-Consistency Improves Chain of Thought Reasoning in Language Models*. ICLR 2023.
