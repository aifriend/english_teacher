from english_improvement_agent.EnglishImprovementAgent import EnglishImprovementAgent
from english_improvement_agent.commonsLib import loggerElk

logger = loggerElk(__name__)


def main():
    # Get the user input
    input_text = input("Enter your text: ")

    # Call the appropriate function based on the user's choice
    function = input("Choose a function ([1] write_properly, [2] write_the_same_grammar_fixed, [3] summarization): ")
    if function == "1":
        improved_text = english_teacher.write_properly(input_text)
        answer = f"Grammar correction:\n{improved_text}"
    elif function == "2":
        improved_text = english_teacher.write_the_same_grammar_fixed(input_text)
        answer = f"Rewrite style:\n{improved_text}"
    elif function == "3":
        improved_text = english_teacher.summarize(input_text)
        answer = f"{improved_text}"
    else:
        answer = "Invalid function. Please choose one of the available functions."

    return function, answer


if __name__ == '__main__':
    english_teacher = EnglishImprovementAgent()

    user_active = True
    while user_active:
        user_input, user_answer = main()
        if user_input == "quit":
            user_active = False
        print(user_answer)
