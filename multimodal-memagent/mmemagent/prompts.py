NO_MEMORY = "No previous memory"


PAGE_MEMORY_PROMPT = """You are presented with a problem, an image of a document page that may contain the answer to the problem, and a previous memory. The previous memory is the information you have accumulated from earlier document pages. Please inspect the provided document page image carefully and update the memory with the new information that helps to answer the problem. Be sure to retain all relevant details from the previous memory while adding any new, useful information.

<problem>
{question}
</problem>

<memory>
{memory}
</memory>

<document_page>
Page {page_number} of {page_count}
The corresponding document page image is provided in the image input.
</document_page>

Updated memory:
"""


FINAL_ANSWER_PROMPT = """You are presented with a problem and a previous memory. The previous memory is the information you have accumulated from the document pages you inspected. Please answer the problem based on the previous memory and put the answer in \\boxed{{}}.

<problem>
{question}
</problem>

<memory>
{memory}
</memory>

Your answer:
"""


def build_page_prompt(question: str, memory: str, page_number: int, page_count: int) -> str:
    return PAGE_MEMORY_PROMPT.format(
        question=question,
        memory=memory or NO_MEMORY,
        page_number=page_number,
        page_count=page_count,
    )


def build_final_prompt(question: str, memory: str) -> str:
    return FINAL_ANSWER_PROMPT.format(question=question, memory=memory or NO_MEMORY)
