import re
from tqdm import tqdm
import json
import pandas as pd
import os
import time
from langchain_google_genai import ChatGoogleGenerativeAI
from src.retrievers import initialize_retriever


LLM_NAME = "models/gemini-2.5-flash-lite"
RETRIEVAL_METHODS = ["baseline", "bm25", "dense", "hybrid"]
TOP_K = 3
SLEEP_TIME = 0.2


# --- Experiments Helper Functions ---

def initialize_llm(llm_name=LLM_NAME, temperature=0):
    
    # Get API key
    api_key = os.getenv("GOOGLE_API_KEY")

    # Build model
    llm = ChatGoogleGenerativeAI(
        model=llm_name, 
        google_api_key=api_key,
        temperature=temperature
    )

    return llm


def ask_llm(llm, q, context_chunks):
    """
    Prompts an llm to answer a question q (str) with context chunks (list of Documents) as context.
    """

    # Format options
    options_text = "\n".join([f"{k}: {v}" for k, v in q["answers"].items()])

    # Rag case
    if len(context_chunks) > 0:

        # Format context
        context_text = "\n\n".join([f"[{c.metadata['id']}]\n{c.page_content}" for c in context_chunks])

    # Baseline case
    else:
        context_text = "No context for this question"


    prompt_template =  """
    You are a technical expert. Answer this multiple choice question using the provided chunks as context.

    Context (Chunks IDs in brackets):
    {context}

    Question:
    {question}

    Options:
    {options}

    Please provide:
    - Your answer (one letter only, no explanation, no punctuation, just an uppercase letter).
    - The IDs of the retrieved chunks you used to answer (comma-separated, e.g. chunk_000, chunk_123).

    You must use this format:

    Answer: [single letter]
    Sources: [comma-separated ids]

    If the answer is not present in the provided context, respond EXACTLY:

    Answer: X
    Sources: NONE
    """ 
    prompt = prompt_template.format(context=context_text, question=q["question"], options=options_text)

    # Call llm
    llm_output_obj = llm.invoke(prompt)

    return llm_output_obj


def extract_llm_response(llm_output_obj):
    """Uses regular expresions for extracting the answer and cited sources form an llm output"""

    # Get text from the ouput object
    llm_output = llm_output_obj.content

    # Extract Answer
    answer_match = re.search(r"Answer:\s*([A-DX])", llm_output)
    llm_answer = answer_match.group(1) if answer_match else None
    
    # Extract Sources
    sources_match = re.search(r"Sources:\s*(.*)", llm_output)
    sources_raw = sources_match.group(1).strip() if sources_match else "NONE"

    if sources_raw.upper() == "NONE":
        llm_sources = []
    else:
        llm_sources = [s.strip() for s in sources_raw.split(",") if s.strip()]
    
    return llm_answer, llm_sources


def is_source_correct(llm, sources, retrieved_chunks, question, paper_reference):
    """
    Prompts an llm to check if cited sources (list of chunks ids (str)) are relevant for answering a question (str).
    Also takes chunks (list of Documents) and a paper reference (str) for performing this task.
    """

    # Format context
    chunks_text = "\n\n".join([c.page_content for c in retrieved_chunks if c.metadata["id"] in sources])

    # Call the llm
    llm_output = llm.invoke(
        f"""
        You are a scientific assistant. Your task is to determine wheter a collection of text chunks from a scientific paper is a valid source of information to answer a question.

        You will consider the following:
        - Question: the question to consider.
        - Reference: a excerpt from the paper containing the information needed to answer the question.
        - Chunks: the collection of chunks to consider. It could be a single chunk.

        You must consider the collection as valid if and only if verifies both conditions:
        - It contains some part of the reference (accept some variation in text formatting).
        - The part of the reference the chunk contains has the main information needed to answer the question.

        Your anser must consist of just one letter (no explanation or punctuation, just an uppercase letter):
        - 'Y' (if the chunk is valid)
        - 'N' (if the chunk is not valid)

        Question:
        {question}

        Reference:
        {paper_reference}

        Chunks:
        {chunks_text}
        """
    )

    # Extract answer
    answer_match = re.search(r"\s*([YN])\s*", llm_output.content)
    llm_answer = answer_match.group(1) if answer_match else None

    if llm_answer == "Y":
        return True
    elif llm_answer == "N":
        return False
    else:
        return llm_output
    

def compute_experiment_metrics(results):
    """
    Returns metrics (dict) for a method experiment results (list(dics))
    """

    # Extract methods
    first = results[0]
    chunking_method = first["chunking_method"]
    retrieval_method = first["retrieval_method"]

    df = pd.DataFrame(results)

    # Number of questions and runs
    num_runs = df["run"].nunique()
    num_questions = df["question_id"].nunique()

    # Coverage: fraction of questions where llm_answer is not "X" or empty
    coverage = df["llm_answer"].apply(lambda x: x not in [None, "X"]).mean()

    # Overall accuracy
    answer_acc = df["answer_correct"].mean()
    
    # Overall source accuracy
    source_acc = df["source_correct"].mean()

    # Answer accuracy std
    ans_acc_per_run = df.groupby("run")["answer_correct"].mean()
    answer_std = ans_acc_per_run.std(ddof=0)

    # Source accuracy std
    src_acc_per_run = df.groupby("run")["source_correct"].mean()
    source_std = src_acc_per_run.std(ddof=0)

    # Average latency
    avg_latency = df["latency"].mean()

    # Save metrics
    metrics = {
        "chunking_method": chunking_method,
        "retrieval_method": retrieval_method,
        "num_runs": num_runs,
        "num_questions": num_questions,
        "answer_accuracy": round(answer_acc, 2),
        "source_accuracy": round(source_acc, 2),
        "coverage": round(coverage, 2),
        "answer_std": round(answer_std, 2),
        "source_std": round(source_std, 2),
        "avg_latency": round(avg_latency, 2)
    }

    return metrics


def save_all_experiments(all_results, all_metrics, saving_dir):
    """
    Save all the experiments results and metrics, both (list(dict)), as csv and json.
    """

    # Build directory if necessary
    saving_dir.mkdir(parents=True, exist_ok=True)

    for data, type in [(all_results, "results"), (all_metrics, "metrics")]:
            
        # Saving paths
        csv_path = saving_dir / f"all_{type}.csv"
        json_path = saving_dir / f"all_{type}.json"

        # Save as csv
        df = pd.DataFrame(data)
        df.to_csv(csv_path, index=False)

        # Save as json
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)

        print(f"Saved all {type} at {csv_path} (csv) and {json_path} (json)\n")

    return None


# --- Main Experimenting Functions ---

def run_experiment(llm, retriever, questions, num_runs=1, sleep_time=SLEEP_TIME):
    """
    Runs rag with a given retriever (Obj), built for a given chunking and retrieval method.
    Call an llm (Obj) for answering all the questions (dict), for a number of runs (int).
    
    Computes and returns experimetn results (list(dict)) and metrics (dict).
    """ 
    # Extract combination
    retrieval_method = retriever._retrieval_method
    chunking_method = retriever._chunking_method

    print("-"*10, "Running Experiment ... ", "-"*10)
    print(f" --> Chunking Method: {chunking_method}")
    print(f" --> Retrieval Method: {retrieval_method}\n")

    experiment_results = []

    # Multiple runs
    for run in range(1, num_runs + 1):

        print(f"Processing run {run}/{num_runs}")

        # Go through questions
        for i, q in enumerate(tqdm(questions, desc="Processing Questions")):

            # Measure latency
            start = time.time()

            # Retrieve chunks
            retrieved_chunks = retriever.invoke(q["question"])
            retrieved_ids = [c.metadata["id"] for c in retrieved_chunks]

            # Call llm
            llm_output_obj = ask_llm(llm, q, retrieved_chunks)

            end = time.time()
            
            # Extract answer, explanation, sources
            llm_answer, llm_sources = extract_llm_response(llm_output_obj)

            if llm_answer is None:
                print(f"---\nWarning: could not extract llm answer from llm output on question {i+1} on run {run}\nLLM output when answering:\n{llm_output_obj}\n---")
            
            if not llm_sources:
                source_correct = None
            else:
                source_correct = is_source_correct(llm, llm_sources, retrieved_chunks, q["question"], q["paper_reference"])

                if not isinstance(source_correct, bool):
                    print(f"---\nWarning: could not check source correctness for question {i + 1} on run {run}.\nLLM output when checking (see finish_reason, probably reached max tokens):\n{source_correct}\n---")
                    source_correct = None

            # Save results
            experiment_results.append({

                "retrieval_method": retrieval_method,
                "chunking_method": chunking_method,

                "run": run,
                "question_id": i + 1,
                **q,

                "retrieved_chunks": retrieved_ids,
                
                "llm_answer": llm_answer,
                "llm_source": llm_sources,

                "is_x": (llm_answer == "X"),
                "answer_correct": (llm_answer == q["correct_answer"]),
                "source_correct": source_correct,
                "latency": round(end - start, 3)
            })

            # Time sleep to avoid api saturation
            time.sleep(sleep_time) 

    # Compute metrics
    experiment_metrics = compute_experiment_metrics(experiment_results)

    print("\n", "-"*10, "Experiment Completed", "-"*10, "\n")

    return experiment_results, experiment_metrics


def run_all_experiments(all_chunks, all_vstores, questions, num_runs, saving_dir, retrieval_methods=RETRIEVAL_METHODS, top_k=TOP_K, sleep_time=SLEEP_TIME):
    """
    Runs all experiments for this project, combining each chunking method with each retrieval method.
    Takes the chunks (dict(list(Documents))) and vstores (dict(Croma Obj)) containing the chunks obtained from
    each chunking method.
    Takes the questions (dict) and number of runs (int) to process.
    Takes the experiments saving directory.
    Takes the retrieval methods (list(str)) to experiment with.

    Returns:
        all_results: (list(dict)) containing the results for every experiment ran. 
        all_metrics: (list(dict)) containing the metrics for every combination of chunking and retrieval.
    """
    
    # Initialize LLM
    llm = initialize_llm()
    
    # Run Experiments

    print("="*10 + " Running All Experiments ... " + "="*10 + "\n")

    all_results = []
    all_metrics = []

    # Iterate through all combinations
    for method, chunks in all_chunks.items():

        vstore = all_vstores[method]

        for ret_method in retrieval_methods:

            # Retriever for this combination
            retriever = initialize_retriever(chunks, vstore, ret_method, top_k)
  
            # Run RAG
            results, metrics = run_experiment(llm, retriever, questions, num_runs, sleep_time)

            # Keep results and metrics
            all_results.extend(results)
            all_metrics.append(metrics)

    # Save all results and metrics
    save_all_experiments(all_results, all_metrics, saving_dir)
        
    print("="*10 + "Experiments Completed" + "="*10)

    return all_results, all_metrics
