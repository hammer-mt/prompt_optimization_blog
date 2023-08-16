# run n completions with a prompt and tags
from langchain.chat_models import ChatOpenAI
from langchain.callbacks.manager import tracing_v2_enabled
from langchain.evaluation import load_evaluator
# Set up the first prompt template
from langchain.prompts import (
    ChatPromptTemplate,
    SystemMessagePromptTemplate,
    HumanMessagePromptTemplate,
)
# from langchain.evaluation.criteria import CriteriaEvalChain

import hashlib
from dotenv import load_dotenv

load_dotenv()

# create the llm object
chat = ChatOpenAI(model="gpt-4", temperature=1) # for testing
# llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0) # for evals

# # count the frequency of words in a string
# def word_frequency(string):
#     frequency = {}
#     for word in string.split():
#         if word in frequency:
#             frequency[word] += 1
#         else:
#             frequency[word] = 1
#     return frequency

# # count the difference in word frequency between the reference and the prediction
# def word_frequency_difference(reference, prediction):
#     # count the frequency of each word in reference and prediction
#     reference_frequency = word_frequency(reference)
#     prediction_frequency = word_frequency(prediction)
    
#     # calculate the difference between the two
#     difference = {}
#     for word in reference_frequency:
#         if word in prediction_frequency:
#             difference[word] = abs(reference_frequency[word] - prediction_frequency[word])
#         else:
#             difference[word] = reference_frequency[word]
#     for word in prediction_frequency:
#         if word not in reference_frequency:
#             difference[word] = prediction_frequency[word]
    
#     # sum up the difference
#     total_difference = sum(difference.values())

# compile the prompt
def compile_prompt(prompt_template, test_case):
    # create a prompt template for a System role
    system_message_prompt_template = SystemMessagePromptTemplate.from_template(
        prompt_template)
    # create a string template for a Human role
    human_template = "{input_text}"
    # create a prompt template for a Human role
    human_message_prompt_template = HumanMessagePromptTemplate.from_template(human_template)
    # create chat prompt template out of one or several message prompt templates
    chat_prompt_template = ChatPromptTemplate.from_messages(
        [system_message_prompt_template, human_message_prompt_template])
    # generate a final prompt by passing the variables
    final_prompt_template = chat_prompt_template.format_prompt(input_text=test_case)
    final_prompt = final_prompt_template.to_messages()

    return final_prompt

def hash_id(string):
    # hash to get a unique id that will be the same if passed the same string
    hash = hashlib.md5(string.encode()).hexdigest()
    hash_id = hash[:8]
    return hash_id

def gen_ex(prompt_templates, test_cases, test_runs=10, project_name="unassigned"):
    # check whether prompt templates is a list of strings, and test cases is a list of objects
    if not isinstance(prompt_templates, list) or not isinstance(test_cases, list):
        raise TypeError("prompt_templates and test_cases must be lists")
    if not all(isinstance(item, str) for item in prompt_templates):
        raise TypeError("prompt_templates must be a list of strings")
    if not all(isinstance(item, object) for item in test_cases):
        raise TypeError("test_cases must be a list of objects")

    # counter to keep track of the number of runs
    counter = 0
    # number of test cases multiplied by number of test runs multiplied by number of prompt templates
    total = len(test_cases) * test_runs * len(prompt_templates)

    # load the evaluators
    distance_evaluator = load_evaluator("embedding_distance")

    # correctness_evaluator = load_evaluator("labeled_criteria", criteria="correctness")

    # create an object to store the results
    results = []
    for template in prompt_templates:
        # hash to get a unique id that will be the same if passed the same template
        pid = f"pid_{hash_id(template)}"

        # loop through the test cases
        for case in test_cases:
            # get a unique id for the case
            cid = f"cid_{hash_id(case['input_text'])}"

            # compile the prompt
            final_prompt = compile_prompt(template, case)
            
            # run n completions with the prompt and tags
            for _ in range(test_runs):
                with tracing_v2_enabled(project_name=project_name, tags=[pid, cid]):
                    counter += 1
                    print(f"Running {pid} with {cid} ({counter}/{total})")
                    result = chat(final_prompt)
                    content = result.content
                    embedding_distance = distance_evaluator.evaluate_strings(prediction=content, reference=case["reference"])['score']
                    
                    # setting up similarity here because need to pass in reference dynamically
                    # criteria = {"style-similarity": f"How similar is this writing style to the reference text out of 100?\nReference: {case['reference']}"}
                    # similarity_evaluator = CriteriaEvalChain.from_llm(llm=llm, criteria=criteria)
                    # similarity_score = similarity_evaluator.evaluate_strings(prediction=content, input=final_prompt)
                    # print(similarity_score)
                    # print(f"Similarity: {similarity_score['score']:.2f}")
                    # correctness = correctness_evaluator.evaluate_strings(
                    #     input=case["input_text"],
                    #     prediction=content,
                    #     reference=case["reference"],
                    # )
                    # print(f"Correctness: {correctness}")

                    # calculate the word frequency difference
                    # frequency_difference = word_frequency_difference(case["reference"], content)

                    data = {
                        "prompt": template,
                        "case": case["input_text"],
                        "result": content,
                        "pid": pid,
                        "cid": cid,
                        "run": counter,
                        "project": project_name,
                        "reference": case["reference"],
                        "embedding_distance": embedding_distance,
                        # "similarity_score": similarity_score['score'],
                        # "frequency_difference": frequency_difference,
                        # "correctness": correctness["score"],
                    }
                    results.append(data)

    return results

