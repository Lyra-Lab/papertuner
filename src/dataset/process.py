# dataset/process.py
import os
import json
import openai
from tqdm import tqdm
from pathlib import Path
from tenacity import retry, stop_after_attempt, wait_exponential

class DatasetProcessor:
    def __init__(self, raw_dir="raw_dataset", processed_dir="processed_dataset"):
        self.raw_dir = Path(raw_dir)
        self.processed_dir = Path(processed_dir)
        self.processed_dir.mkdir(exist_ok=True)
        
        # OpenRouter configuration
        self.client = openai.OpenAI(
            base_url="https://openrouter.ai/api/v1",
            api_key="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"
            # api_key=os.getenv("OPENROUTER_API_KEY")
        )
        self.model = "google/gemini-2.0-pro-exp-02-05:free"  # Free tier model
        # self.model = "anthropic/claude-2"  # For higher quality (needs credits)

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=4, max=10))
    def _api_call(self, prompt, max_tokens=1500):
        response = self.client.chat.completions.create(
            model=self.model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.3,
            max_tokens=max_tokens
        )
        return response.choices[0].message.content.strip()

    def _extract_sections(self, text):
        """Extract research sections using LLM"""
        prompt = f"""Identify and extract these sections from the research paper text:
        - Problem Statement
        - Literature Review
        - Hypothesis
        - Methodology
        
        Return ONLY a JSON format with these keys. If a section isn't found, set its value to null.
        Text: {text[:6000]}  # Truncate to stay within context limits
        """
        
        try:
            response = self._api_call(prompt)
            return json.loads(response.split("```json")[-1].split("```")[0])
        except Exception as e:
            print(f"Section extraction failed: {e}")
            return {}

    def _generate_qa(self, sections):
        """Generate QA pair from extracted sections"""
        prompt = f"""Create a research question and methodology answer using this context:
        Problem: {sections.get('Problem Statement', '')}
        Literature: {sections.get('Literature Review', '')}
        Hypothesis: {sections.get('Hypothesis', '')}

        Format output as:
        Q: [Question about research methodology]
        A: [Detailed methodology answering the question]
        """
        
        try:
            response = self._api_call(prompt, max_tokens=2000)
            q = response.split("Q: ")[1].split("\nA: ")[0].strip()
            a = response.split("\nA: ")[1].strip()
            return {"question": q, "answer": a}
        except:
            return None

    def process_paper(self, paper_path):
        """Process a single paper"""
        with open(paper_path) as f:
            paper = json.load(f)
        
        # Extract sections
        sections = self._extract_sections(paper['full_text'])
        if not sections:
            return None
            
        # Generate QA pair
        qa = self._generate_qa(sections)
        if not qa:
            return None
            
        return {
            "metadata": {
                "id": paper["id"],
                "title": paper["title"],
                "categories": paper["categories"]
            },
            "sections": sections,
            "qa": qa
        }

    def run(self, batch_size=50):
        """Process entire dataset with rate limiting"""
        paper_files = list((self.raw_dir / "papers").glob("*.json"))
        manifest = []
        
        for i, paper_file in enumerate(tqdm(paper_files)):
            # Process paper
            processed = self.process_paper(paper_file)
            if not processed:
                continue
            
            # Save processed file
            output_path = self.processed_dir / f"processed_{paper_file.name}"
            with open(output_path, 'w') as f:
                json.dump(processed, f, indent=2)
            
            manifest.append({
                "original_file": paper_file.name,
                "processed_file": output_path.name,
                "sections_found": list(processed["sections"].keys())
            })
            
            # Rate limiting
            if (i + 1) % batch_size == 0:
                time.sleep(60)  # Pause to avoid rate limits

        # Save processing manifest
        with open(self.processed_dir / "process_manifest.json", 'w') as f:
            json.dump(manifest, f, indent=2)

if __name__ == "__main__":
    # Set your API key first: export OPENROUTER_API_KEY="your-key"
    processor = DatasetProcessor(
            raw_dir="../../data/raw_dataset",
            processed_dir="../../data/processed_dataset"
            )
    processor.run()
