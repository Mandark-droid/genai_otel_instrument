"""CrewAI Multi-Agent Framework Instrumentation Example.

This example demonstrates automatic OpenTelemetry instrumentation for
CrewAI, a popular multi-agent collaboration framework with role-based
agents, tasks, and crews.

CrewAI has 30,000+ GitHub stars and 1M+ monthly downloads, making it
one of the most popular agent frameworks.

Requirements:
    pip install genai-otel-instrument
    pip install crewai crewai-tools
    export OPENAI_API_KEY=your_api_key  # Or other LLM provider
    export OTEL_EXPORTER_OTLP_ENDPOINT=http://localhost:4318
"""

import os

import genai_otel

# Initialize instrumentation - CrewAI is enabled automatically
genai_otel.instrument(
    service_name="crewai-example",
    endpoint="http://localhost:4318",
)

print("\n" + "=" * 80)
print("CrewAI Multi-Agent Framework OpenTelemetry Instrumentation Example")
print("=" * 80 + "\n")

# Import CrewAI
try:
    from crewai import Agent, Crew, Process, Task
except ImportError:
    print("ERROR: CrewAI not installed. Install with:")
    print("  pip install crewai")
    exit(1)

# Check for API key (CrewAI typically uses OpenAI by default)
api_key = os.getenv("OPENAI_API_KEY")
if not api_key:
    print("ERROR: OPENAI_API_KEY environment variable not set")
    print("CrewAI uses OpenAI models by default. Get your key from:")
    print("  https://platform.openai.com/api-keys")
    exit(1)

print("1. Simple Sequential Crew...")
print("-" * 80)

# Define agents with roles, goals, and backstories
researcher = Agent(
    role="Senior Research Analyst",
    goal="Uncover cutting-edge developments in AI and data science",
    backstory=(
        "You are a senior research analyst with expertise in AI and "
        "data science. You have a keen eye for emerging trends and "
        "breakthrough technologies."
    ),
    verbose=True,
)

writer = Agent(
    role="Tech Content Writer",
    goal="Write compelling and informative tech articles",
    backstory=(
        "You are a skilled tech writer with the ability to explain "
        "complex technical concepts in simple, engaging language."
    ),
    verbose=True,
)

# Define tasks
research_task = Task(
    description=(
        "Research the latest developments in large language models (LLMs) "
        "from the past 6 months. Focus on breakthrough capabilities and "
        "real-world applications."
    ),
    expected_output=(
        "A comprehensive report on recent LLM developments, including "
        "key breakthroughs, new capabilities, and notable applications."
    ),
    agent=researcher,
)

writing_task = Task(
    description=(
        "Using the research findings, write a 300-word article about "
        "the latest LLM developments that would be suitable for a "
        "tech blog audience."
    ),
    expected_output=(
        "A well-written 300-word article suitable for publication "
        "on a tech blog, with an engaging headline and clear structure."
    ),
    agent=writer,
)

# Create a crew with sequential process
crew = Crew(
    agents=[researcher, writer],
    tasks=[research_task, writing_task],
    process=Process.sequential,  # Tasks execute in order
    verbose=True,
)

# Execute the crew (this is what gets instrumented)
print("Executing crew...")
result = crew.kickoff()

print("\n" + "-" * 80)
print("CREW RESULT:")
print("-" * 80)
print(result)
print()

print("2. Hierarchical Crew with Manager...")
print("-" * 80)

# Create agents for hierarchical process
data_analyst = Agent(
    role="Data Analyst",
    goal="Analyze data and provide insights",
    backstory="You are an expert data analyst with strong analytical skills.",
    verbose=True,
)

visualization_expert = Agent(
    role="Data Visualization Specialist",
    goal="Create clear and informative data visualizations",
    backstory="You excel at transforming data into visual stories.",
    verbose=True,
)

# Create tasks
analysis_task = Task(
    description="Analyze sales data trends for Q4 2024",
    expected_output="Detailed analysis of sales trends with key insights",
    agent=data_analyst,
)

visualization_task = Task(
    description="Create visualization recommendations based on the analysis",
    expected_output="Recommendations for charts and graphs to visualize the data",
    agent=visualization_expert,
)

# Hierarchical crew with a manager agent
hierarchical_crew = Crew(
    agents=[data_analyst, visualization_expert],
    tasks=[analysis_task, visualization_task],
    process=Process.hierarchical,  # Manager coordinates agents
    verbose=True,
)

print("Executing hierarchical crew...")
result = hierarchical_crew.kickoff()

print("\n" + "-" * 80)
print("HIERARCHICAL CREW RESULT:")
print("-" * 80)
print(result)
print()

print("3. Crew with Custom Inputs...")
print("-" * 80)

# Create a crew that accepts custom inputs
topic_researcher = Agent(
    role="Research Specialist",
    goal="Research any given topic thoroughly",
    backstory="You are a versatile researcher who can quickly understand and research any topic.",
    verbose=True,
)

topic_task = Task(
    description="Research the topic: {topic} and provide a summary",
    expected_output="A comprehensive summary of the research findings",
    agent=topic_researcher,
)

research_crew = Crew(
    agents=[topic_researcher],
    tasks=[topic_task],
    process=Process.sequential,
    verbose=True,
)

# Execute with custom inputs
inputs = {"topic": "Quantum Computing Applications"}
print(f"Researching topic: {inputs['topic']}")
result = research_crew.kickoff(inputs=inputs)

print("\n" + "-" * 80)
print("RESEARCH RESULT:")
print("-" * 80)
print(result)
print()

print("=" * 80)
print("Telemetry Data Collected:")
print("=" * 80)
print(
    """
For each crew execution, the following data is automatically collected:

TRACES (Spans):
- Span name: crewai.crew.execution
- Attributes:
  - gen_ai.system: "crewai"
  - gen_ai.operation.name: "crew.execution"
  - crewai.crew.id: Unique crew identifier
  - crewai.crew.name: Crew name
  - crewai.process.type: "sequential" or "hierarchical"
  - crewai.agent_count: Number of agents in the crew
  - crewai.agent.roles: List of agent roles
  - crewai.agent.goals: List of agent goals
  - crewai.task_count: Number of tasks
  - crewai.task.descriptions: List of task descriptions
  - crewai.tools: List of tools used by agents
  - crewai.tool_count: Total number of tools
  - crewai.inputs.*: Custom input parameters
  - crewai.output: Crew execution result
  - crewai.tasks_completed: Number of completed tasks

METRICS:
- genai.requests: Crew execution count
- genai.tokens: Aggregated token usage across all agents
- genai.latency: Crew execution duration
- genai.cost: Total cost for all LLM calls in the crew

View these metrics in your observability platform (Grafana, Jaeger, etc.)
"""
)

print("=" * 80)
print("CrewAI Framework Features:")
print("=" * 80)
print(
    """
Key Features Instrumented:
- Crew Execution: Tracks crew.kickoff() with full visibility
- Agent Collaboration: Monitors role-based agent interactions
- Task Execution: Tracks task assignment and completion
- Process Types: Sequential and hierarchical coordination
- Manager Agent: Tracks manager delegation in hierarchical mode
- Custom Inputs: Captures parameterized crew executions
- Multi-Agent Workflows: Complete observability of agent teams

Crew Modes:
- Sequential: Agents work in order, output feeds next task
- Hierarchical: Manager agent coordinates and delegates tasks

Benefits:
- Popular framework with 30K+ stars and 1M+ downloads
- Role-based agent design for clear responsibility
- Task-oriented workflow with expected outputs
- Built-in collaboration between specialized agents
- Zero-code instrumentation with full observability
"""
)

print("=" * 80)
print("Example complete! Check your OTLP collector/Grafana for traces and metrics.")
print("=" * 80 + "\n")
