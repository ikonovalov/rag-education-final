from langchain_gigachat import GigaChat, GigaChatEmbeddings
from ragas import EvaluationDataset

from src.utils import GIGACHAT_API_KEY, GIGACHAT_API_SCOPE

evaluator_llm = GigaChat(
    credentials=GIGACHAT_API_KEY,
    scope=GIGACHAT_API_SCOPE,
    model='GigaChat-2-Max',
    verify_ssl_certs=False,
    temperature=0.3,
)

evaluator_embedding = GigaChatEmbeddings(
    credentials=GIGACHAT_API_KEY,
    scope=GIGACHAT_API_SCOPE,
    verify_ssl_certs=False,
)

def create_ragas_dataset(state: dict, reference: str):
    dataset = [
        {
            "user_input": state["question"],
            "retrieved_contexts": [rdoc.page_content for rdoc in state["retrieved"]],
            "response": state["answer"],
            "reference": reference,
        }
    ]
    evaluation_dataset = EvaluationDataset.from_dict(dataset)
    return evaluation_dataset