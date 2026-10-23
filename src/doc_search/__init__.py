def main() -> None:
    import uvicorn

    uvicorn.run("doc_search.api:app", host="127.0.0.1", port=8000)
