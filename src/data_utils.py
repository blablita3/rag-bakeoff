import json
import PyPDF2
import shutil

def load_pdf_text(pdf_path):
    """Loads pdf text (str) from a pdf path (str / Path)"""

    text = ""
    with open(pdf_path, "rb") as f:
        reader = PyPDF2.PdfReader(f)
        for page in reader.pages:
            page_text = page.extract_text()
            if page_text:
                text += page_text + "\n"
    return text


def load_html(html_path):
    """Loads html as str form a path (str / Path)"""

    with open(html_path, "r", encoding="utf-8") as f:
        text = f.read()
    return text


def load_json(path):
    """Loads a dict from a json file path (str / Path)"""

    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def publish_directory(input_dir, output_dir):
    """
    Saves the content inside the input dir (path) into the ouput dir (path), overwritting it.
    """

    # Remove previous output dir
    if output_dir.exists():
        shutil.rmtree(output_dir)

    # Copy input into output
    shutil.copytree(input_dir, output_dir)

    print(f"Saved content of {input_dir} directory into {output_dir} directory")

    return None


def check_source_manually(question_id, running_results, documents):
    """Prints info to manually check is a source is valid (testing)"""
    
    print(f"Question: \n{running_results[question_id - 1]["question"]}\n")
    print(f"Paper Reference: \n{running_results[question_id - 1]["paper_reference"]}\n")
    print(f"LLM cited sources:")
    for source in running_results[question_id - 1]["llm_source"]:
        print(source)
        print(next((d.page_content for d in documents if d.metadata["id"] == source), ""))
        print("\n")
    return None