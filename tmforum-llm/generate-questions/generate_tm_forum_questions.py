import os
import json
import time
import random
from typing import List, Dict, Any
import google.generativeai as genai
from google.generativeai.types import HarmCategory, HarmBlockThreshold

# Function to configure the Gemini API with your API key
def setup_gemini_api(api_key: str) -> None:
    genai.configure(api_key=api_key)

# Function to generate questions using Gemini Pro
def generate_questions(
        system_prompt: str,
        num_batches: int = 5,
        questions_per_batch: int = 5,
        output_file: str = "tm_forum_questions.json"
) -> List[Dict[str, str]]:
    # Set up Gemini Pro model
    generation_config = {
        "temperature": 0.9,
        "top_p": 0.95,
        "top_k": 40,
        "max_output_tokens": 8192,
    }

    safety_settings = {
        HarmCategory.HARM_CATEGORY_HATE_SPEECH: HarmBlockThreshold.BLOCK_NONE,
        HarmCategory.HARM_CATEGORY_HARASSMENT: HarmBlockThreshold.BLOCK_NONE,
        HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT: HarmBlockThreshold.BLOCK_NONE,
        HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT: HarmBlockThreshold.BLOCK_NONE,
    }

    model = genai.GenerativeModel(
        model_name="gemini-2.5-pro-exp-03-25",
        generation_config=generation_config,
        safety_settings=safety_settings,
    )

    # Prepare to collect all generated questions
    all_questions = []

    # Load existing questions if the file exists
    if os.path.exists(output_file):
        try:
            with open(output_file, 'r') as f:
                all_questions = json.load(f)
            print(f"Loaded {len(all_questions)} existing questions from {output_file}")
        except json.JSONDecodeError:
            print(f"Error loading existing questions, starting fresh.")

    try:
        for i in range(num_batches):
            # Create a prompt asking for a specific number of questions
            prompt = f"{system_prompt}\n\nPlease generate {questions_per_batch} diverse questions following the format described above."

            print(f"Generating batch {i+1}/{num_batches}...")

            # Call the Gemini API to generate questions
            response = model.generate_content(prompt)

            # Extract the generated text
            response_text = response.text

            # Try to parse the JSON from the response
            try:
                # Find the JSON array in the response
                start_idx = response_text.find('[')
                end_idx = response_text.rfind(']') + 1

                if start_idx != -1 and end_idx != -1:
                    json_text = response_text[start_idx:end_idx]
                    batch_questions = json.loads(json_text)

                    # Add to our collection
                    all_questions.extend(batch_questions)

                    # Save questions after each batch to prevent data loss
                    with open(output_file, 'w') as f:
                        json.dump(all_questions, f, indent=2)

                    print(f"Generated {len(batch_questions)} questions. Total: {len(all_questions)}")
                else:
                    print(f"Couldn't find JSON array in response. Skipping batch.")
            except json.JSONDecodeError as e:
                print(f"Error parsing JSON from response: {e}")
                print(f"Response was: {response_text[:100]}...")

            # Add a delay to respect rate limits (free tier)
            delay = random.uniform(1.0, 2.0)
            time.sleep(delay)

    except Exception as e:
        print(f"An error occurred: {e}")
        # Save whatever we have so far
        if all_questions:
            with open(output_file, 'w') as f:
                json.dump(all_questions, f, indent=2)

    return all_questions

def scenario_basic() -> List[Dict[str, str]]:
    system_prompt = """You are an expert in TM Forum APIs, specifications, user guides, and component inventory. Your 
    task is to generate a diverse set of sample use case questions relevant to the telecom domain. These questions 
    should cover business, technical, and architectural perspectives and should be framed in a way that both 
    individuals familiar with TM Forum terminology and those from the telecom business domain without specific 
    TM Forum knowledge can understand. The questions should leverage your comprehensive understanding of all available 
    TM Forum APIs, specifications, user guides, and component inventory. Output the questions in a structured 
    JSON format as an array of objects, where each object has two keys: 
    'use case type' (with possible values: 'technical', 'architectural', 'business') and 
    'question'. 
    Here are a few examples to guide you:

    [
        {
            "use case type": "business",
            "question": "As a telecom business user, how can I find out the list of services a specific customer is currently subscribed to?"
        },
        {
            "use case type": "technical",
            "question": "What are the specific API calls and parameters needed to retrieve detailed information about a customer's network connection status using TM Forum APIs?"
        },
        {
            "use case type": "architectural",
            "question": "How can TM Forum's Open Digital Architecture (ODA) components and Open APIs be used to design a new order management system for a telecom operator?"
        },
        {
            "use case type": "business",
            "question": "Our sales team needs to understand the different types of customer accounts we manage. How can we get this information from our systems?"
        },
        {
            "use case type": "technical",
            "question": "According to TM Forum specifications, what is the standard way to represent a customer's billing account in API requests and responses?"
        }
    ]"""

    num_batches = 20  # Number of API calls to make
    questions_per_batch = 50  # How many questions to request per batch
    output_file = "tm_forum_questions.json"

    questions = generate_questions(
        system_prompt=system_prompt,
        num_batches=num_batches,
        questions_per_batch=questions_per_batch,
        output_file=output_file
    )
    print(f"Successfully generated {len(questions)} basic questions and saved to {output_file}")
    return questions

def scenario_advanced() -> List[Dict[str, str]]:
    system_prompt = """You are a Principal Telecom Solutions Architect with unparalleled, expert-level mastery 
    across the entire TM Forum portfolio. This includes, but is not limited to:
    - All TM Forum Open APIs: Deep understanding of their specifications (including data models derived from SID), user guides, intended scopes, interactions, event models, and common extension points.
    - ODA Components: Comprehensive knowledge of the ODA Functional Framework, component definitions, capabilities, relationships, exposed/required APIs, and their role within the ODA Canvas.
    - TM Forum SID (Shared Information/Data Model): Intricate knowledge of the SID framework, domains, Aggregate Business Entities (ABEs), relationships, and attribute levels.
    - eTOM (enhanced Telecom Operations Map): Detailed understanding of business process flows, levels, decompositions, and their relationship to ODA components and APIs.
    - TM Forum Best Practices & Guides: Familiarity with guides on topics like API Governance, ODA Implementation, AI Maturity, Data Governance, etc.

    Your task is to generate *highly complex, scenario-driven questions* specifically designed to challenge an architect's understanding of how to *apply and integrate TM Forum assets to solve sophisticated real-world telecom problems.* These questions should necessitate:

    1.  Deep Synthesis: Requiring information from *multiple* distinct TM Forum artifacts (e.g., combining knowledge of several APIs, ODA components, SID domains, and eTOM processes).
    2.  Architectural Reasoning: Focusing on design choices, trade-offs, integration patterns, data consistency across domains, lifecycle management, non-functional requirements (performance, scalability, security) within the TM Forum/ODA context.
    3.  Problem Solving: Presenting intricate challenges or ambiguities that require interpreting TM Forum standards and proposing viable solutions or mitigation strategies.
    4.  Advanced Concepts: Touching upon complex API interactions (e.g., long-running processes, asynchronous patterns, event-driven architectures), ODA component customisation/extension, or mapping complex business needs onto the frameworks.

    The questions should be answerable *only* through deep familiarity with the TM Forum specifications and guides, often requiring careful consideration of how different parts of the framework interact. Avoid simple definition lookups or questions answerable by consulting a single API specification in isolation.

    Output Format: Generate the output as a structured *JSON array of objects*. Each object within the array must have exactly two keys:
    - `use_case_type`: A string with one of the following possible values: *'technical'*, *'architectural'*, or *'business'*.
    - `question`: A string containing the detailed, complex question itself.
    
    Examples of Expected Output Format and Question Complexity:

    ```json
    [
      {
        "use_case_type": "architectural",
        "question": "A customer initiates a complex B2B order via TMF622 (Product Order) involving a new site survey (using TMF653 Service Test), provisioning of Layer 2 VPN services (potentially modeled via TMF640 Service Activation & Config), and activation of associated CPE resources (TMF652 Resource Order). Detail the architectural pattern for orchestrating this flow using relevant ODA Core Commerce Management components. Specifically address: How to maintain transaction integrity and state management across these asynchronous API calls? Which SID entities are critical for correlating the Product Order items to the specific Service and Resource instances created across TMF638 (Service Inventory) and TMF639 (Resource Inventory)? What event notifications (e.g., using TMF688 Event Production/Subscription) should be published/consumed by which ODA components (e.g., Service Order Management, Resource Order Management, Service Inventory Management) to ensure visibility and trigger subsequent process steps?"
      },
      {
        "use_case_type": "architectural",
        "question": "Design the interaction between ODA 'Billing & Charging Management', 'Customer Problem Management' (TMF656), and 'Service Quality Management' (TMF634) components when a customer disputes a bill due to perceived poor service quality on their broadband service. Explain: How the trouble ticket (TMF656) should reference the relevant customer bill (TMF678) and potentially the affected service instance (TMF638)? Which specific SID ABEs (e.g., CustomerBill, TroubleTicket, ServiceProblem, MetricDefMeasureThreshold) are needed to capture the necessary data points for root cause analysis and potential bill adjustment? How can performance metrics retrieved via TMF634 be correlated with the customer's service instance and the timeframe of the complaint to validate the issue?"
      },
      {
        "use_case_type": "technical",
        "question": "A CSP needs to extend TMF620 (Product Catalog API) to include complex, dynamic eligibility rules based on real-time network congestion data (not typically part of the standard SID model for Catalog). Propose an architecturally sound approach that leverages ODA principles. Consider: Should the core 'Product Catalog Management' ODA component be extended, or should a separate 'Eligibility Check' microservice/component be introduced? If a separate service is used, how would it interact with the standard TMF620 flow during product offering discovery or validation? Which TM Forum APIs (standard or custom) would facilitate this interaction? How can this extension be designed to minimize impact on ODA compliance and future upgrades of the standard Product Catalog component?"
      },
      {
        "use_case_type": "architectural",
        "question": "In a multi-country telecom operation standardizing on TM Forum SID and ODA, how would you architect the 'Party Management' (TMF632) solution to handle conflicting regulatory requirements for storing and processing Personally Identifiable Information (PII) across different jurisdictions (e.g., GDPR vs. local data sovereignty laws)? Address: Which SID ABEs within the Party domain are most sensitive? How can ODA component design (e.g., data partitioning, federation, separate component instances) help enforce jurisdictional boundaries? What role might TM Forum's Data Governance frameworks play in defining policies and ensuring compliance within this distributed architecture?"
      },
      {
        "use_case_type": "business",
        "question": "A telecom operator wants to launch a 'Network-as-a-Service' (NaaS) offering for enterprise clients, allowing them to dynamically request and configure dedicated network slices with guaranteed SLAs via a self-service portal. Map this business objective onto the relevant eTOM Level 2/3 processes and identify the core ODA functional blocks required (across Engagement, Party, Core Commerce, Production domains). Describe the high-level interaction flow using key TM Forum Open APIs (e.g., TMF620, TMF622, TMF641, TMF652, TMF700 Network Slice Mgmt) and pinpoint the main challenges in translating commercial slice parameters into technical resource configurations managed through SID."
      }
    ]
    ```"""

    num_batches = 25  # Number of API calls to make
    questions_per_batch = 6  # How many questions to request per batch
    output_file = "tm_forum_advanced_questions.json"

    questions = generate_questions(
        system_prompt=system_prompt,
        num_batches=num_batches,
        questions_per_batch=questions_per_batch,
        output_file=output_file
    )
    print(f"Successfully generated {len(questions)} complex questions and saved to {output_file}")
    return questions

def main():
    # Replace with your actual API key or set as environment variable
    api_key = os.environ.get("GOOGLE_API_KEY", "")

    if not api_key:
        api_key = input("Please enter your Google API key: ")

    # Setup the API
    setup_gemini_api(api_key)
    questions = scenario_advanced()
    # Print a few sample questions
    print("\nSample basic questions:")
    for q in questions[:3]:
        print(f"- {q['use case type']}: {q['question']}")

if __name__ == "__main__":
    main()