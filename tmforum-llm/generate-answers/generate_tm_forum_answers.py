import os
import json
import time
from datetime import datetime
from typing import List, Dict
import google.generativeai as genai


def setup_gemini_api(api_key: str) -> None:
    """
    Configure the Gemini API with the provided API key.
    """
    genai.configure(api_key=api_key)


def generate_answers_batch(
        system_prompt: str,
        questions: List[str],
        batch_size: int = 5,
        output_file: str = "optimized_question_answer_pairs.json"
) -> List[Dict[str, str]]:
    """
    Generate answers for a list of questions using the Gemini model in batches.

    Args:
        system_prompt: The system prompt describing the answer generation requirements.
        questions: A list of questions to answer.
        batch_size: Number of questions to process in each batch.
        output_file: File to save the question-answer pairs.

    Returns:
        A list of question-answer pairs.
    """
    # Prepare to collect all question-answer pairs
    question_answer_pairs = []

    # Load existing pairs if the file exists
    if os.path.exists(output_file):
        try:
            with open(output_file, 'r') as f:
                question_answer_pairs = json.load(f)
            print(f"Loaded {len(question_answer_pairs)} existing question-answer pairs from {output_file}")
        except json.JSONDecodeError:
            print(f"Error loading existing data, starting fresh.")

    # Set up the Gemini model configuration
    generation_config = {
        "temperature": 0.9,
        "top_p": 0.95,
        "top_k": 40,
        "max_output_tokens": 8192,
    }

    # Default Gemini model name

    model = genai.GenerativeModel(
        model_name="gemini-2.0-flash",
        generation_config=generation_config
    )


    # Process questions in batches
    for i in range(0, len(questions), batch_size):
        batch = questions[i:i + batch_size]
        batch_questions = batch # Extract questions from the batch

        # Create a prompt that lists all questions in the batch
        batch_prompt = (
                f"{system_prompt}\n\n"
                "Here is a list of questions:\n"
                + "\n".join([f"{idx + 1}. {q}" for idx, q in enumerate(batch_questions)])
                + "\n\n"
                  "Provide the answers in an array format, maintaining the same order as the questions."
                  "Sample format:"
                    "```json"
                    "["
                    "   {"
                    "       \"answer\":\"text\""
                    "   },"
                    "   {"
                    "       \"answer\":\"text\""
                    "   },"
                    "]"
        )

        print(f"Processing batch {i // batch_size + 1} with {len(batch_questions)} questions...")
        print(batch_prompt)
        api_start_time = datetime.now()
        try:
            # Generate answers for the batch
            response = model.generate_content(batch_prompt)
            response_text = response.text
            print(response_text)
            # Parse the response to extract the array of answers
            try:
                start_idx = response_text.find('[')
                end_idx = response_text.rfind(']') + 1

                if start_idx != -1 and end_idx != -1:
                    answers = json.loads(response_text[start_idx:end_idx])

                    # Validate if the number of answers matches the number of questions
                    if len(answers) != len(batch_questions):
                        print(f"Warning: Mismatch in question-answer count for batch {i // batch_size + 1}.")
                        continue

                    # Map answers back to their corresponding questions
                    for question, answer in zip(batch_questions, answers):
                        question_answer_pairs.append({
                            "question": question,
                            "answer": answer["answer"]
                        })

                    # Save pairs after each batch to prevent data loss
                    with open(output_file, 'w') as f:
                        json.dump(question_answer_pairs, f, indent=2)

                    print(f"Batch {i // batch_size + 1} processed. Total pairs: {len(question_answer_pairs)}")
                else:
                    print(f"Failed to extract answers array for batch {i // batch_size + 1}. Response: {response_text[:500]}")
            except json.JSONDecodeError as e:
                print(f"Error parsing JSON from response for batch {i // batch_size + 1}: {e}")
                print(f"Response: {response_text[:500]}")

        except Exception as e:
            print(f"An error occurred in batch {i // batch_size + 1}: {e}")

        # Add a delay to respect rate limits
        time_difference_seconds = int((datetime.now() - api_start_time).total_seconds())
        buffer_time = max(0, 240 - time_difference_seconds)
        time.sleep(buffer_time)

    return question_answer_pairs


def main():
    # System prompt for generating answers
    system_prompt = """
    You are a Principal Telecom Solutions Architect with unparalleled, expert-level mastery across the entire TM Forum portfolio. This includes, but is not limited to:
    - All TM Forum Open APIs: Deep understanding of their specifications (including data models derived from SID), user guides, intended scopes, interactions, event models, and common extension points.
    - ODA Components: Comprehensive knowledge of the ODA Functional Framework, component definitions, capabilities, relationships, exposed/required APIs, and their role within the ODA Canvas.
    - TM Forum SID (Shared Information/Data Model): Intricate knowledge of the SID framework, domains, Aggregate Business Entities (ABEs), relationships, and attribute levels.
    - eTOM (enhanced Telecom Operations Map): Detailed understanding of business process flows, levels, decompositions, and their relationship to ODA components and APIs.
    - TM Forum Best Practices & Guides: Familiarity with guides on topics like API Governance, ODA Implementation, AI Maturity, Data Governance, etc.
    
    Your Task:
    You will be provided with one complex, scenario-based question related to applying TM Forum standards in the telecom domain. Your task is to provide a detailed, accurate, and expert-level answer to this question.
    
    Answer Requirements:
    
    1.  Depth and Synthesis: Your answer must demonstrate a deep understanding by synthesizing information from various relevant TM Forum assets (APIs, ODA, SID, eTOM). Do not provide superficial answers; explain the -how- and -why-.
    2.  Conciseness: Aim for a comprehensive yet concise answer, typically around 200-300 words. Focus on delivering the most critical information effectively within this range.
    3.  Specificity (Based on Question Type): Tailor the details in your answer based on the nature of the question:
        a. If the question is primarily TECHNICAL:
            - Clearly identify the specific TM Forum Open API(s) involved by name and number (e.g., "TMF622 Product Order Management API").
            - Mention the key API endpoints relevant to the scenario (e.g., `POST /productOrder`, `GET /productOrder/{id}`).
            - Reference important schemas, data structures, or key SID attributes used within the API payloads (e.g., `ProductOrder`, `ProductOrderItem`, `productOffering.id`).
            - Briefly mention the relevant API specification version if critical, otherwise assume latest relevant version.
        b. If the question is primarily ARCHITECTURAL:
            - Name the core ODA Components involved (e.g., 'Service Order Management', 'Resource Inventory Management', 'API Gateway').
            - Describe their interrelationships, interaction patterns (e.g., synchronous calls, event-based communication via TMF688), and the flow of information.
            - Identify the key TM Forum APIs used for communication -between- these components.
            - Discuss relevant SID domains/entities crucial for data consistency across the architecture.
            - Highlight key design considerations or patterns applied.
        c. If the question is primarily BUSINESS:
            - Map the scenario to relevant eTOM business processes (mentioning levels if possible, e.g., Level 2/3 in Operations).
            - Identify the key ODA functional blocks or components supporting the business capability.
            - Reference the primary TM Forum APIs and SID concepts that underpin the business process implementation.
            - Focus on how TM Forum standards enable the described business outcome.
    4.  Clarity and Structure: Present the answer in a clear, logical, and well-structured manner.
    5.  Practicality: Provide information that is useful for informing the design and development of solutions using TM Forum standards.
    
    Your Goal: Act as a trusted advisor, providing a robust and insightful answer that leverages the full breadth of your TM Forum expertise to solve the presented challenge.
    """

    # Load questions from a file
    input_file = "../generate-questions/tm_forum_questions.json"  # Replace with the file containing your questions
    with open(input_file, 'r') as f:
        questions = json.load(f)

    questions_list = [q["question"] for q in questions][601:]

    # Replace with your actual API key or set as environment variable
    api_key = os.environ.get("GOOGLE_API_KEY", "")
    if not api_key:
        api_key = input("Please enter your Google API key: ")

    # Setup the API
    setup_gemini_api(api_key)

    # Generate answers
    output_file = "optimized_question_answer_pairs.json"
    question_answer_pairs = generate_answers_batch(
        system_prompt=system_prompt,
        questions=questions_list,
        batch_size=5,  # Customize batch size as needed
        output_file=output_file
    )

    print(f"Successfully generated answers for {len(question_answer_pairs)} questions.")
    print("\nSample question-answer pairs:")
    for pair in question_answer_pairs[:3]:
        print(f"Q: {pair['question']}\nA: {pair['answer']}\n")


if __name__ == "__main__":
    main()