from langchain_core.documents import Document

SPACE = " "
BLANK = ""
SZ_LIMIT = 6000


def cleanup(docs: list[Document]):
    for doc in docs:
        text = doc.page_content
        text = text.replace("&amp;apos;", "'")
        text = text.replace("&amp;", SPACE)
        text = text.replace("amp;", SPACE)
        text = text.replace("apos;", SPACE)
        text = text.replace("**", BLANK)
        text = text.replace("__", BLANK)
        text = text.replace("\t", BLANK)
        text = text.replace("  ", SPACE)
        doc.page_content = text
    return docs

def truncate_oversized(docs: list[Document]):
    for d in docs:
        if len(d.page_content) > SZ_LIMIT:
            d.page_content = d.page_content[:SZ_LIMIT]
    return docs
