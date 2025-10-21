from langchain_core.documents import Document

SPACE = " "
BLANK = ""
SZ_LIMIT = 6000

def extract_meta_and_propagate(docs: list[Document]):
    for doc in docs:
        text = doc.page_content
        title = extract_field("Title: ", text)
        image = extract_field("Image_Name: ", text)
        ingredients = extract_field("Cleaned_Ingredients: ", text)
        doc.metadata['title'] = title
        doc.metadata['image'] = image
        doc.metadata['ingredients'] = ingredients
        doc.id = doc.metadata['row']

def extract_field(tag: str, text: str) -> str:
    # Находим позицию маркера
    start_pos = text.find(tag) + len(tag)  # Смещаемся после маркера
    end_pos = text.find("\n", start_pos)  # Ищем перенос строки после маркера
    if start_pos> 0 and end_pos == -1: # скорее всего последний тэг
        end_pos = len(text)
    if start_pos != -1 and end_pos != -1:
        result = text[start_pos:end_pos]
        return result
    else:
        return ""

def cleanup(docs: list[Document]):
    # for d in docs:
    #     text = d.page_content
    #     # trim first ": number\n"
    #     start_pos = text.find("\n")
    #     d.page_content = text[start_pos+1:]
    return docs

def truncate_oversized(docs: list[Document]):
    for d in docs:
        if len(d.page_content) > SZ_LIMIT:
            d.page_content = d.page_content[:SZ_LIMIT]
    return docs
