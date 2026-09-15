#!/bin/bash
# Step 3 (conversation history / query condensation) test script.
# Tests directly against rag-service's /v1/query (bypasses the backend,
# same pattern as every other test this session) - so this only tests
# the condensation mechanism itself, not the backend wiring that fetches
# history from chat_messages. Run this first; test through the real UI
# separately once the backend wiring is deployed.

BASE_URL="http://localhost:8001/v1/query"
DOCUMENT_ID="ae08b6b1-fdc6-4304-8592-d02473a84ecc"
DOCUMENT_NAME="Protocol_Ulcerative Colitis.pdf"
ORGANIZATION_ID="ee5b5b9f-1848-4b78-8d55-5662f02ad471"
OUTPUT_FILE="conversation_history_test_$(date +%Y%m%d_%H%M%S).log"

run_query() {
  local label="$1"
  local body="$2"

  echo "=================================================================" | tee -a "$OUTPUT_FILE"
  echo "[$label]" | tee -a "$OUTPUT_FILE"
  echo "-----------------------------------------------------------------" | tee -a "$OUTPUT_FILE"

  curl -s -X POST "$BASE_URL" \
    -H "Content-Type: application/json" \
    -d "$body" \
    | python3 -m json.tool | tee -a "$OUTPUT_FILE"

  echo "" | tee -a "$OUTPUT_FILE"
  sleep 1
}

echo "Starting conversation history test run - $(date)" | tee "$OUTPUT_FILE"
echo "Document: $DOCUMENT_NAME ($DOCUMENT_ID)" | tee -a "$OUTPUT_FILE"
echo "" | tee -a "$OUTPUT_FILE"

# ---------------------------------------------------------------------
# Test 1: Standalone baseline - no history sent at all.
# Should behave identically to every query run earlier this session.
# This is your regression check for "standalone questions are unaffected."
# ---------------------------------------------------------------------
run_query "Test 1 - Standalone (no history)" '{
  "query": "What are the main exclusion criteria?",
  "document_id": "'"$DOCUMENT_ID"'",
  "document_name": "'"$DOCUMENT_NAME"'",
  "organization_id": "'"$ORGANIZATION_ID"'"
}'

# ---------------------------------------------------------------------
# Test 2: Real refinement follow-up - the core acceptance criterion.
# Turn 1 establishes exclusion criteria. Turn 2 is a genuine "be more
# specific" follow-up that only makes sense with turn 1's context.
# PASS: condensed query resolves "the hepatic ones" against exclusion
# criteria (e.g. "What are the hepatic/liver-related exclusion criteria?"),
# NOT a fresh unrelated topic.
# ---------------------------------------------------------------------
run_query "Test 2 - Refinement follow-up" '{
  "query": "Can you be more specific about the hepatic ones?",
  "document_id": "'"$DOCUMENT_ID"'",
  "document_name": "'"$DOCUMENT_NAME"'",
  "organization_id": "'"$ORGANIZATION_ID"'",
  "conversation_history": [
    {"role": "user", "content": "What are the main exclusion criteria?"},
    {"role": "assistant", "content": "The exclusion criteria include pregnancy, colostomy history, chronic liver disease, active tuberculosis, hepatitis B or C, and several other conditions - 27 items in total."}
  ]
}'

# ---------------------------------------------------------------------
# Test 3: Refusal must survive a follow-up (safety-critical case).
# Turn 1 is a real unanswerable (D-category) question. Turn 2 is a vague
# follow-up referencing it.
# PASS: still refuses with the exact "I don't have this information"
# phrase - condensation must not make the model MORE willing to guess.
# FAIL: confidently answers something about approval anywhere.
# ---------------------------------------------------------------------
run_query "Test 3 - Refusal survives follow-up" '{
  "query": "What about in other countries?",
  "document_id": "'"$DOCUMENT_ID"'",
  "document_name": "'"$DOCUMENT_NAME"'",
  "organization_id": "'"$ORGANIZATION_ID"'",
  "conversation_history": [
    {"role": "user", "content": "What is the FDA approval status of TJ301?"},
    {"role": "assistant", "content": "I do not have this information regarding the FDA approval status of TJ301. The protocol only describes it as under development in Phase II trials."}
  ]
}'

# ---------------------------------------------------------------------
# Test 4: Ambiguous pronoun follow-up - the exact ticket example.
# Turn 1 establishes CRP's role. Turn 2 is the literal ticket phrase.
# PASS: condensed query resolves "more specific" against CRP context,
# not a generic/empty rewrite.
# ---------------------------------------------------------------------
run_query "Test 4 - Can you be more specific? (ticket example)" '{
  "query": "Can you be more specific?",
  "document_id": "'"$DOCUMENT_ID"'",
  "document_name": "'"$DOCUMENT_NAME"'",
  "organization_id": "'"$ORGANIZATION_ID"'",
  "conversation_history": [
    {"role": "user", "content": "How is CRP used for stratification in this study?"},
    {"role": "assistant", "content": "CRP is not used as a current stratification factor. It was used in an earlier version (v1.0) but was replaced. CRP is currently tracked as an exploratory biomarker."}
  ]
}'

# ---------------------------------------------------------------------
# Test 5: Topic-change should NOT be dragged back to old topic.
# Turn 1 is about CRP. Turn 2 is a genuinely new, standalone question
# that happens to follow it, but does not reference it.
# PASS: condensed query is unchanged / not distorted toward CRP -
# confirms condensation doesn't over-eagerly attach unrelated new
# questions to old context.
# ---------------------------------------------------------------------
run_query "Test 5 - Genuine topic change, not a follow-up" '{
  "query": "What is the primary endpoint of the study?",
  "document_id": "'"$DOCUMENT_ID"'",
  "document_name": "'"$DOCUMENT_NAME"'",
  "organization_id": "'"$ORGANIZATION_ID"'",
  "conversation_history": [
    {"role": "user", "content": "How is CRP used for stratification in this study?"},
    {"role": "assistant", "content": "CRP is not used as a current stratification factor. It is tracked as an exploratory biomarker."}
  ]
}'

echo "=================================================================" | tee -a "$OUTPUT_FILE"
echo "Test run complete. Full output saved to: $OUTPUT_FILE" | tee -a "$OUTPUT_FILE"
echo "" | tee -a "$OUTPUT_FILE"
echo "Check the 'retrieval_query' field (if logged) or rag_service.log's" | tee -a "$OUTPUT_FILE"
echo "trace lines to see what each follow-up was actually condensed to." | tee -a "$OUTPUT_FILE"