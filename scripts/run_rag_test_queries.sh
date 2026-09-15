#!/bin/bash
# RAG test queries - runs everything from RAG_Test_Queries.docx EXCEPT
# Category F (follow-up chains) - those need conversation/chat history,
# which isn't implemented yet, so they'd fail on infrastructure grounds
# rather than testing anything real.
#
# Includes Category G (out-of-scope capability checks) since they're
# still worth running for visibility - but per the doc, failures there
# are tracked separately and never count against the system.
#
# Categories C and D are the permanent regression checks - read those
# results most carefully.

BASE_URL="http://localhost:8001/v1/query"
DOCUMENT_ID="ae08b6b1-fdc6-4304-8592-d02473a84ecc"
DOCUMENT_NAME="Protocol_Ulcerative Colitis.pdf"
ORGANIZATION_ID="ee5b5b9f-1848-4b78-8d55-5662f02ad471"
OUTPUT_FILE="rag_test_results_$(date +%Y%m%d_%H%M%S).log"

run_query() {
  local category="$1"
  local num="$2"
  local query="$3"

  echo "=================================================================" | tee -a "$OUTPUT_FILE"
  echo "[$category-$num] $query" | tee -a "$OUTPUT_FILE"
  echo "-----------------------------------------------------------------" | tee -a "$OUTPUT_FILE"

  curl -s -X POST "$BASE_URL" \
    -H "Content-Type: application/json" \
    -d "{\"query\": \"$query\", \"document_id\": \"$DOCUMENT_ID\", \"document_name\": \"$DOCUMENT_NAME\", \"organization_id\": \"$ORGANIZATION_ID\"}" \
    | python3 -m json.tool | tee -a "$OUTPUT_FILE"

  echo "" | tee -a "$OUTPUT_FILE"
  sleep 1  # be polite to the LLM API between calls
}

echo "Starting RAG test run - $(date)" | tee "$OUTPUT_FILE"
echo "Document: $DOCUMENT_NAME ($DOCUMENT_ID)" | tee -a "$OUTPUT_FILE"
echo "Categories: A, B, C, D, E, G (F excluded - no chat history yet)" | tee -a "$OUTPUT_FILE"
echo "" | tee -a "$OUTPUT_FILE"

# --- Category A: In-scope factual (16 queries) ---
run_query "A" 1  "What is the primary endpoint of the study?"
run_query "A" 2  "What is the investigational medicinal product?"
run_query "A" 3  "What is the indication for this study?"
run_query "A" 4  "What phase is this trial?"
run_query "A" 5  "How many patients will be randomised and in what ratio?"
run_query "A" 6  "What is the dosing schedule of the study drug?"
run_query "A" 7  "How is randomisation stratified in this study?"
run_query "A" 8  "What is the eligible age range for patients?"
run_query "A" 9  "What are the study periods and their durations?"
run_query "A" 10 "How many site visits are there and what are they?"
run_query "A" 11 "What is the infusion time for the study drug?"
run_query "A" 12 "How is the study drug stored and prepared?"
run_query "A" 13 "What is the planned sample size and statistical power?"
run_query "A" 14 "In which countries is the trial being conducted?"
run_query "A" 15 "What is the definition of clinical and endoscopic remission?"
run_query "A" 16 "What is the size of the PK subgroup and where is it based?"

# --- Category B: Completeness / multi-part (5 queries) ---
run_query "B" 17 "What are all the secondary endpoints?"
run_query "B" 18 "What are the key inclusion criteria?"
run_query "B" 19 "What are the main exclusion criteria?"
run_query "B" 20 "What exploratory biomarkers are measured in this study?"
run_query "B" 21 "What assessments are performed at the Week 12 (End of Treatment) visit?"

# --- Category C: Current-vs-historical / CRP + amendment trap (6 queries) ---
# Permanent regression check - read these most carefully.
run_query "C" 22 "How is CRP used for stratification in this study?"
run_query "C" 23 "What are the current randomisation stratification factors?"
run_query "C" 24 "What changed in the randomisation stratification between protocol versions?"
run_query "C" 25 "Which countries changed between protocol versions?"
run_query "C" 26 "What is the name of the safety committee in the current protocol?"
run_query "C" 27 "What does CRP refer to and how is it used in this protocol?"

# --- Category D: Unanswerable - correct result is honest refusal (10 queries) ---
# Permanent regression check - proves the grounded-refusal fix is working.
run_query "D" 28 "What is the prevalence of ulcerative colitis in China?"
run_query "D" 29 "When was Olamkicept (TJ301) first discovered?"
run_query "D" 30 "How much does the study drug cost?"
run_query "D" 31 "What were the results of the trial?"
run_query "D" 32 "Which site recruited the most patients?"
run_query "D" 33 "What is the FDA approval status of TJ301?"
run_query "D" 34 "What is the molecular mechanism of action of TJ301?"
run_query "D" 35 "How many patients have been enrolled so far?"
run_query "D" 36 "What is the sponsor's annual revenue?"
run_query "D" 37 "Which competing ulcerative colitis drugs are on the market?"

# --- Category E: Ambiguous - should ask a clarifying question, not guess (5 queries) ---
run_query "E" 38 "Can you be more specific?"
run_query "E" 39 "What about corticosteroids?"
run_query "E" 40 "Tell me about CRP."
run_query "E" 41 "What are the requirements?"
run_query "E" 42 "What is the dose?"

# --- Category F SKIPPED (43-46) - follow-up chains need conversation/chat
# history, which isn't implemented yet. Running these now would just
# confirm the known gap, not test anything new.

# --- Category G: Out-of-scope capability (4 queries) ---
# Per the doc: track separately, NEVER count as a failure.
run_query "G" 47 "Why was prior corticosteroid use chosen as a stratification factor?"
run_query "G" 48 "Which visits require an ECG?"
run_query "G" 49 "What fields should the eCRF include based on this protocol?"
run_query "G" 50 "Which countries are likely to recruit patients fastest?"

echo "=================================================================" | tee -a "$OUTPUT_FILE"
echo "Test run complete. Full output saved to: $OUTPUT_FILE" | tee -a "$OUTPUT_FILE"
echo "" | tee -a "$OUTPUT_FILE"
echo "Read categories C and D first - permanent regression checks." | tee -a "$OUTPUT_FILE"
echo "Category G failures don't count against the system - track separately." | tee -a "$OUTPUT_FILE"