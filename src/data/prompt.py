
def render(schema_ddl: str, question: str) -> str:
    return f"### Schema:\n{schema_ddl}\n\n### Prompt:\n{question}\n\n### Response:\n"
