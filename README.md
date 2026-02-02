# Intro RAG Bake-Off: Technical Q&A

This project explores different Retrieval-Augmented Generation (RAG) pipelines for a multiple-
choice Q&A, based on a Meta’s research paper introducing REFRAG. Using a curated dataset,
we compared Okapi BM25, Dense and Hybrid retrieval methods over both paragraph based
and recursively built chunks, and we assessed their improvements over the baseline LLM.

## 📋 Report
The accompanying [**report**](report.pdf) provides a comprehensive analysis of our RAG benchmarking study, covering:

- **Theoretical Foundations**: Detailed explanations of RAG concepts, retrieval methods (BM25, dense, hybrid), and chunking strategies

- **Methodology**: Complete experimental setup, dataset construction, and evaluation metrics

- **Implementation Details**: Technical specifications of our pipeline and methods implementation

- **Results Analysis**: Comparative performance across all tested configurations with statistical validation

- **Discussion, Insights & Future Work**: Practical recommendations for RAG system design based on our findings adn suggested research directions

The report adopts a hybrid approach, combining theoretical background with empirical analysis, to serve both newcomers to RAG and experienced practitioners looking for actionable insights.
## 

## 🚀 Quick Start

**1. Installation**:

```
git clone https://github.com/blablita3/rag-bake-off.git
cd rag-bake-off
pip install -r requirements.txt
```

**2. Required API Key**:
This project uses Google's Gemini API. You must obtain an API key and set it as an environment variable:

```
# On Linux/Mac:
export GOOGLE_API_KEY="your-api-key-here"

# On Windows (Command Prompt):
set GOOGLE_API_KEY=your-api-key-here

# On Windows (PowerShell):
$env:GOOGLE_API_KEY="your-api-key-here"
```

**3. Usage**:
```
python main.py <command> [options]

Commands:

all          Run the full pipeline (chunking, experiments and figures)
chunks       Generate chunks
experiments  Run experiments and generate metrics
figures      Generate plots
publish      Promote current directory to final one

Options:

--small     Use a smaller dataset (for testing)
```

Running the program will rewritte the `/current` directory with the obtained results. Running `python main.py publish` will save the current results into `/final` to avoid loosing them. Be careful when publishing since you will rewrite the final directory. Sparse commands like `python main.py <experiments / figures>` will load the chunks / metrics saved in the `/current` directory.

## 🎓 Acknowledgments

This project was part of the *Conurso de Modelización de Problemas de Empresa*, an event organised in the [*Facultad de Ciencias Matemáticas, UCM*](https://www.matematicas.ucm.es). The [problem statement](./problem_statement.pdf) was proposed by the enterprise [*Management Solutions*](https://www.managementsolutions.com/). 

**Author**: *Ángel Valencia*

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.