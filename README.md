# PaperTuner: Research Assistant Model Toolkit

## Overview
PaperTuner is a Python toolkit for creating high-quality research methodology QA datasets from scientific papers and fine-tuning large language models (LLMs) to serve as research assistants. It automates the extraction of technical Q&A pairs from research papers, validates them, and provides tools for training and evaluating custom research assistant models.

## Features
- **Automated Dataset Creation:** Extracts and validates question-answer pairs from research papers (PDFs, arXiv, etc.)
- **Custom Model Training:** Fine-tunes LLMs (e.g., Phi, Qwen) for research methodology Q&A
- **Hugging Face Integration:** Load and push datasets/models to the Hugging Face Hub
- **Dataset Validation & Statistics:** Ensures QA quality and provides dataset analytics
- **Extensible & Configurable:** Easily adapt to new domains or model architectures

## Installation
PaperTuner requires Python 3.10+ and CUDA-enabled hardware for training. Install via pip:

```bash
pip install .
```
Or, for development:
```bash
git clone https://github.com/yourusername/papertuner.git
cd papertuner
pip install -e .
```

## Dependencies
Key dependencies (see `pyproject.toml` for full list):
- `huggingface_hub`, `datasets`, `transformers`, `trl`, `torch`, `vllm`, `unsloth`
- `PyMuPDF`, `arxiv`, `google-genai`, `tenacity`, `tqdm`, `requests`

## Configuration
Set the following environment variables for API access and Hugging Face integration:
- `GEMINI_API_KEY` – Google Generative AI API key (for QA generation)
- `BESPOKE_API_KEY` – Bespoke Labs API key (for fact-checking rewards)
- `HF_TOKEN` – Hugging Face access token
- `HF_REPO_ID` – (optional) Hugging Face dataset repo ID (default: `user/ml-papers-qa`)

## Usage

### 1. Dataset Creation
Create a dataset from arXiv papers or PDFs:

```python
from papertuner.dataset import ResearchPaperProcessor
processor = ResearchPaperProcessor()
# Process up to 3 papers on "large language models"
processor.process_papers(max_papers=3, search_query="large language models")
# Validate the processed dataset
results = processor.validate_dataset()
print(results)
```
Or run the example script:
```bash
python dataset_example.py
```

### 2. Model Training
Fine-tune a research assistant model on your dataset:

```python
from papertuner.train import ResearchAssistantTrainer
trainer = ResearchAssistantTrainer(model_name="unsloth/DeepSeek-R1-Distill-Qwen-1.5B-GGUF")
trainer.train(
    dataset_name="your_hf_dataset_id",  # e.g., "user/ml-papers-qa"
    push_to_hf=True,
    hf_username="your_hf_username",
    hf_model_name="your_model_name",
    hf_token="your_hf_token",
    bespoke_api_token="your_bespoke_api_token"
)
```
Or run the example script:
```bash
python training_example.py
```

### 3. Inference
After training, run inference with your model:
```python
response = trainer.run_inference(model, tokenizer, "What are the key considerations for designing a neural network architecture for image classification?")
print(response)
```

## Dataset Format
Each QA entry contains:
- `question`: Technical research methodology question
- `answer`: Detailed methodology answer
- `category`: Question category/type (e.g., Theoretical Foundations, Architecture & Design, etc.)
- `paper_id`: Source paper identifier
- `paper_title`: Title of the source paper
- `categories`: arXiv or domain categories

## Question Categories
- Theoretical Foundations
- Architecture & Design
- Ethical Considerations
- Analysis & Interpretation
- Implementation Strategy & Techniques
- Comparative Assessment
- Handling Specific Challenges
- Adaptation & Transfer
- Future Directions
- Methodology & Approach

## Command-Line Tools
- `papertuner-dataset` – Dataset creation and processing
- `papertuner-train` – Model training and fine-tuning

## Contributing
Contributions are welcome! Please open issues or pull requests on [GitHub](https://github.com/yourusername/papertuner).

## License
MIT License. See [LICENSE](LICENSE) for details.

## Links
- [Homepage](https://github.com/yourusername/papertuner)
- [Bug Tracker](https://github.com/yourusername/papertuner/issues)
- [Documentation](https://github.com/yourusername/papertuner#readme)

    # Research Methodology QA Dataset

    ## Overview
    - Contains 2528 validated question-answer pairs
    - Derived from 1288 research papers
    - Domains: 92B05, 14-01, 13P25, 12Y05, 97M60, cond-mat.mes-hall, astro-ph.HE, 60J27, 97C42, 37Fxx, cond-mat.mtrl-sci, math.CT, q-fin.EC, physics.comp-ph, nucl-th, G.2.1, 92D15, 92D50, 82C40, 60J25, 60J75, q-bio.SC, 65, 70, 74, 76, 82, 92, 93, 94, cs.CY, cs.CR, q-bio.GN, cs.LO, G.3; I.5.3; J.3; K.8.1, 60, 92, 37N25, 92-10, 92C37, 92C40, math.RA, math.GT, 92-08, cs.MS, physics.ins-det, physics.class-ph, cond-mat.soft, 92Bxx, 92B99, cs.MA, D.2.12, 92D30, J.3, 14QXX, math.AP, cs.OH, physics.pop-ph, 37N25, 34C25, 34F05, 92D25, I.2.1, 37N25 (Primary), 34-04 (Secondary), cs.LG, q-bio.CB, astro-ph.SR, cs.AI, cs.RO, cond-mat.dis-nn, physics.atm-clus, q-fin.PR, 92D25 (Primary), 60J05 (Secondary), 34C05, 80A30, 68W15, A.1; I.2.1, 00-XX, 0-XX, 05c85 92b05, physics.soc-ph, cs.CG, I.6.6, E.1; F.2.2; G.2.1; J.3, cs.SI, 68, cs.DL, stat.AP, stat.CO, 92C42, cs.CL, 92-02, cond-mat.str-el, cs.IR, cs.DS, 92-04, nlin.PS, cs.SY, cs.SC, math.GM, 35Q92, 92-08, 49Q22, 91-10, 92-04, 74-10, 70-10, 35Q92, 37N25, 82C22, stat.ME, q-bio.OT, nlin.CD, 53C43, 53C07, 83C22, physics.ao-ph, math.CA, C.4, 35K25, 35K57, 35R09, 68, 92, cond-mat, J.3;F.4.2, 93A30; J3; 92B05; 65K05; 90C20, eess.IV, cs.IT, 05C10, 57M15, 35J70, 35B65, 42B37, 35B30 (Primary) 35J60, 35K55 (Secondary), 37N25, 37G15, 92-10, 34D20, 76-10 (Primary) 76T30, 76T06, 74A50, 80A22 (Secondary), G.3; J.3, Primary 60J27, 60J28, secondary 92B05, 92E20, 92C42, q-bio.PE, F.2;J.3, nlin.AO, physics.flu-dyn, math-ph, physics.optics, q-bio.TO, math.PR, math.GR, 37N25, 80A30, 92C45, 92E20, 14M25, 11Z05, eess.SP, quant-ph, 68T07, cs.HC, physics.acc-ph, physics.chem-ph, cs.ET, eess.SY, q-bio.NC, 68, 81, 92, cs.NA, 92Exx, 92C42, 14P10, 37N25, 14P05, physics.data-an, nlin.CG, 35R30, 35Q92, 35R11, 35K40, physics.med-ph, 37C10, 80A30, 92C40, 92D25, cs.CE, cs.SE, math.DG, 92E10;20B99, q-bio.BM, 93B52;93D20;35B35, 60H35, 65C99, 92C40, 92C37, math.HO, 92C50, physics.hist-ph, cs.NE, cs.AR, astro-ph, G.1.7; G.1.5, cs.DM, 92C37 92C42, stat.ML, math.LO, cs.CL, G.3 G.2.1, 78A70.82D55.35K25.35E05, adap-org, 93B15, 92B05, 05C85, cond-mat.other, I.2.7, cs.CC, hep-th, 92C42, 90B15, F.4; G.4; I.6; J.3, cs.GL, math.ST, 68Q25, 68Q17, 68R10, 05C85, 68W25, 92C42, 92C40, chao-dyn, q-bio, cs.PL, cs.PF, 92B05, 94A15, 94A17, 94-01, cond-mat.stat-mech, q-fin.CP, astro-ph.CO, math.DS, q-bio.QM, 60J85, 92D15, I.2, math.AG, G.1.6; J.3, 05A15, 06A25, 05C05, 92, J.3; I.2.8, 60J22, 65C05, 65Z05, 82B31, 92E20, econ.GN, 03B80, 92B05, 03B35, 03F07, math.NA, G.3; I.6.5; J.2; J.3, I.2; I.4, stat.TH, 92B05, 92C17, 92C15, 92D15, 34B60, 92B99, 65L05, 92C42 (Primary) 62F15, 97K80 (Secondary), Primary: 60J60 Secondary: 92D10, 92D15, 92D25, 01A75, 60-03, I.2.7; J.3; H.2.8, 92C37 (Primary), 35R35 (Secondary), physics.ed-ph, 92B08, cond-mat.supr-con, 94B 05B, stat.OT, astro-ph.EP, I.2.1; J.3, math.OC, 92B15, 62P10, physics.gen-ph, G.1.7; I.2.0, 92Cxx, math.IT, 92D25, 92B05, 60J85, 92-10, 92D10, 60J20, J.3; K.3; I.3.8, 92C17, 11T99 ; 05C20, cs.DB, cs.FL, 92C42, 62H30, 68T10, 37N25 34C15 37G15 34C26 92C60 92C80, math.CO, 37N25, 92C15, 92C42, comp-gas, 62P10, 92.08, cs.DC, 35K57, 35K55, 92D25, 92D99, math.MP, 92C42, 92C37, 92B05, physics.bio-ph, physics.app-ph, I.6.3;I.6.4;I.6.5, G.3, Comptuational science, math.AC, cs.CV, cs.GR, 82B20 (82B26), q-bio.MN, math.MG, 58J45, 35K57, 41A05, 41A25, 41A30, 41A63, 65D25, 65M20, 65M70,
  46E22, 35B36, I.5.2

    ## Question Categories
    Theoretical Foundations, Architecture & Design, Ethical Considerations, Analysis & Interpretation, Implementation Strategy & Techniques, Comparative Assessment, Handling Specific Challenges, Adaptation & Transfer, Future Directions, Methodology & Approach

    ## Fields
    - `question`: Technical research methodology question
    - `answer`: Detailed methodology answer
    - `category`: Question category/type
    - `paper_id`: Source paper identifier
    - `paper_title`: Title of the source paper
    - `categories`: arXiv categories
    