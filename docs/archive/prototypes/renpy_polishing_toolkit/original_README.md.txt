---
description: Iterative Literacy Polish Workflow for Ren'Py Translations
---

# Iterative Literary Polish Workflow

This workflow describes the systematic process for elevating the quality of Ren'Py game translations (specifically Chinese) through iterative extraction, polish proposal, and batch application.

## Prerequisites
- A Ren'Py project with a translation folder (e.g., `game/tl/schinese`).
- Python installed for running utility scripts.

## Phase 1: Setup & Extraction

1.  **Generate Comprehensive Dialogue Dump**:
    Run a script to extract all dialogue lines from the target `.rpy` files into a single text file (e.g., `full_dialogue_extraction.txt`). This allows for linear reading and context-aware polishing.
    *   *Tool:* `_Tools/extract_dialogue_pairs.py` (Custom script needed)
    *   *Format:* `[LineNumber] | [Speaker] | [Original] | [Current_Translation]`

2.  **Initialize Task Tracking**:
    Create a `task.md` to track progress by batches (e.g., every 800-1000 lines).

## Phase 2: The Loop (Batch Processing)

For each batch defined in `task.md`:

1.  **Draft Polish Proposal**:
    Read the next chunk of `full_dialogue_extraction.txt`. Identify lines that are functionally correct but lack literary flair, character voice, or correct tone.
    Create a `proposed_polish_batch_N.md` file using the following format:
    ```markdown
    # Proposed Polish Batch N (Focus Area)
    
    ## Candidates
    ### S[Batch]-[ID]: [Brief Description]
    **Context**: [Who is speaking to whom, situation]
    **Original**: [English text]
    **Current**: [Current translation]
    **Critique**: [Why it needs improvement]
    **Proposed**: [New translation]
    ```

2.  **Review (Human or Self-Correction)**:
    Review the proposed changes. Check for:
    -   Consistency with Glossary (e.g., specific terms).
    -   Character Voice (e.g., is Noakai stern? Is Kirik sassy?).
    -   Accuracy (ensure meaning isn't lost).

3.  **Apply Changes**:
    Use a script or manual editing tools to apply the changes to the actual `.rpy` files.
    *   *Method:* Search for the exact "Current" string and replace it with "Proposed".
    *   *Tool:* `_Tools/apply_polish_batch.py`
    *   *Verification:* Ensure no `SyntaxError` is introduced.

4.  **Update Tracking**:
    Mark the batch as complete in `task.md`.

## Phase 3: Verification & Cleanup

1.  **Verify Integrity**:
    Run `verify_polish_status.py` to ensure all proposed strings made it into the codebase.
    Run `verify_glossary.py` to ensure no glossary terms broke.

2.  **Consolidate**:
    Move applied proposals to a `Polish_Proposals_Resolved` folder to keep the workspace clean.

## Scripts Reference

### `extract_dialogue_pairs.py` (Concept)
Reads all `.rpy` files, regex matches `translate schinese label:` blocks, and outputs the English comment line and the translated line pair.

### `apply_polish_batch.py` (Concept)
Parses the markdown proposal file, extracts the "Current" and "Proposed" strings, and performs a safe find-and-replace in the target directories.
