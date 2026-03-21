# NPA Management Tool

Streamlit-based application for managing Non-Performing Assets (NPA) using an advanced RAG pipeline, LangChain/LangGraph, and OpenAI GPT-4o.

## Highlights

- **Master Agent (Auto-route)**: One chat interface that automatically routes queries to the right tool.
- **Manual Tools**: Dedicated tabs for Policy Q&A, Account Recommendation, and Branch Analysis.
- **Policy Q&A (RAG)**: Parent-document retrieval + hybrid search + optional Cohere re-ranking + query decomposition + table-aware extraction.
- **Account Recommendation**: Uses `data/accounts.csv` to generate corrective actions and an OTS/compromise estimate.
- **Branch Analysis**: Portfolio overview by branch or consolidated view for `ALL`.

## Setup

1. Ensure required files exist:
   - `data/npa_policy.pdf` (required)
   - `data/accounts.csv` (required)
   - `data/ots.pdf` (optional; currently not used in the pipeline)

2. Create `config/.env` and set your API keys:
   ```
   OPENAI_API_KEY=your_actual_api_key
   # Optional: enables Cohere re-ranking
   COHERE_API_KEY=your_cohere_api_key
   ```

3. Install dependencies:
   ```
   pip install -r requirements.txt
   ```

4. Run the app:
   ```
   streamlit run app.py
   ```

The app will start on `http://localhost:8501` by default.

## NPA Classification Codes (Accounts CSV)

- `4` - Standard / SMA (not NPA)
- `5` - Sub-standard
- `6` - Doubtful I
- `7` - Doubtful II
- `8` - Loss Asset

These labels are used in account recommendations and branch analysis outputs.

## Project Structure

- `app.py` - Streamlit UI (Master Agent + Manual Tools)
- `src/rag_setup.py` - RAG pipeline for policy Q&A
- `src/master_agent.py` - Tool-routing agent
- `src/action_recommendation.py` - Account lookup + recommendations
- `src/branch_analysis.py` - Branch-level analytics
- `data/` - PDFs and `accounts.csv`

## Notes

- `data/accounts.csv` is mock data. Replace it with real account data as needed.
