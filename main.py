from tutoringSystem import MisconceptionTutoringSystem
#from ontology import OntologyService

def main():
    #ontology = OntologyService(graph=None)  # replace with RDFLib graph
    tutor = MisconceptionTutoringSystem()

    session_id, greeting = tutor.start_conversation()
    print(greeting)

    while True:
        user_input = input("> ")
        if user_input.lower() in {"quit", "exit"}:
            break
        response = tutor.process_user_input(session_id, user_input)
        print(response)

if __name__ == "__main__":
    main()