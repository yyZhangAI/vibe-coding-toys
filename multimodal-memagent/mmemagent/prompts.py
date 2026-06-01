NO_MEMORY = "No previous memory"


PAGE_MEMORY_PROMPT = """You are a visual long-document memory agent for MMLongBench-DOC.

Your task is to inspect exactly one document page image and update a compact memory that will help answer the question later.
Keep useful details from the previous memory. Add only evidence from the current page that may be relevant.
If the page is irrelevant, preserve the existing memory and briefly note that no relevant evidence was found.
Do not answer the final question yet.

<question>
{question}
</question>

<page>
Page {page_number} of {page_count}
</page>

<previous_memory>
{memory}
</previous_memory>

Return only the updated memory in plain text."""


FINAL_ANSWER_PROMPT = """You are answering an MMLongBench-DOC question from the memory collected across document pages.

Use only the memory below. If the memory does not contain enough evidence to answer, answer "Not answerable".
Give a brief rationale, then put the concise final answer inside <answer>...</answer>.

<question>
{question}
</question>

<memory>
{memory}
</memory>"""


def build_page_prompt(question: str, memory: str, page_number: int, page_count: int) -> str:
    return PAGE_MEMORY_PROMPT.format(
        question=question,
        memory=memory or NO_MEMORY,
        page_number=page_number,
        page_count=page_count,
    )


def build_final_prompt(question: str, memory: str) -> str:
    return FINAL_ANSWER_PROMPT.format(question=question, memory=memory or NO_MEMORY)

