GEN_TEXT_METADATA_PROMPT = """
You receive the plain text content of a single document section:\n\n
Output requirements:\n
- Follow the provided schema exactly.\n
Document section:\n\n
{content}\n
"""


GEN_IMG_METADATA_PROMPT = """
You receive an image, extarct the Data for the provided schema.\n\n
Output requirements:\n
- Follow the provided schema exactly.\n
"""


GEN_TABLE_METADATA_PROMPT = """
You receive an Table, extarct the Data for the provided schema.\n\n
Output requirements:\n
- Follow the provided schema exactly.\n
Table:\n\n
{content}\n
"""
