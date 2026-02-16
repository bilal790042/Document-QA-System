from services.qa_service import QAService

def main():
    qa = QAService()

    while True:
        question = input("\nAsk a question (or type 'exit'): ")

        if question.lower() == "exit":
            break

        answer, docs = qa.ask(question)

        print("\nAnswer:\n", answer)

        print("\nSources:")
        for d in docs:
            print("-", d.metadata.get("source", "unknown"))

if __name__ == "__main__":
    main()