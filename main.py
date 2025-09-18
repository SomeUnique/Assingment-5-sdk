import asyncio
from pydantic import BaseModel
from agents import (
    Agent, 
    OutputGuardrailTripwireTriggered, 
    Runner,
    input_guardrail,
    output_guardrail,
    GuardrailFunctionOutput,
    InputGuardrailTripwireTriggered
)
import rich
from connection import config

# ==================== Exercise 1: Class Timings Guardrail ====================
class ClassTimingOutput(BaseModel):
    response: str
    isTimingChangeRequested: bool

class_timing_guard = Agent(
    name="Class Timing Guard",
    instructions="""
        Your task is to check if the student is requesting to change class timings.
        If the request contains any plea to change timings (especially with crying emojis),
        mark it as a timing change request.
    """,
    output_type=ClassTimingOutput
)

@input_guardrail
async def timing_guardrail(ctx, agent, input):
    result = await Runner.run(class_timing_guard, input, run_config=config)
    rich.print(result.final_output)
    return GuardrailFunctionOutput(
        output_info=result.final_output.response,
        tripwire_triggered=result.final_output.isTimingChangeRequested
    )

student_agent = Agent(
    name='Student',
    instructions="You are a student agent",
    input_guardrails=[timing_guardrail]
)

async def ex1_main():
    try:
        result = await Runner.run(student_agent, 
                                 'I want to change my class timings 😭😭', 
                                 run_config=config)
        print("Request processed")
    except InputGuardrailTripwireTriggered:
        print('Class timing change request denied!')

# ==================== Exercise 2: Father Temperature Guardrail ====================
class TemperatureOutput(BaseModel):
    response: str
    isTooCold: bool

father_agent = Agent(
    name="Father Agent",
    instructions="""
        You are a father monitoring your child's room temperature settings.
        If the child tries to set temperature below 26°C, stop them.
        Always be protective but gentle in your response.
    """,
    output_type=TemperatureOutput
)

@input_guardrail
async def temperature_guardrail(ctx, agent, input):
    result = await Runner.run(father_agent, input, run_config=config)
    return GuardrailFunctionOutput(
        output_info=result.final_output.response,
        tripwire_triggered=result.final_output.isTooCold
    )

child_agent = Agent(
    name="Child",
    instructions="You are a child who wants to adjust room temperature",
    input_guardrails=[temperature_guardrail]
)

async def ex2_main():
    try:
        result = await Runner.run(child_agent, 
                                "Can we set AC to 22°C? It's too hot!", 
                                run_config=config)
        print(result.final_output)
    except InputGuardrailTripwireTriggered:
        print("Father says: Temperature cannot be set below 26°C!")

# ==================== Exercise 3: Gatekeeper School Guardrail ====================
class StudentVerificationOutput(BaseModel):
    response: str
    isFromOtherSchool: bool

gatekeeper_agent = Agent(
    name="Gatekeeper Agent",
    instructions="""
        You are a school gatekeeper. Verify if the student belongs to your school.
        Check for school ID, uniform, or any other identification.
        If the student is from another school, deny entry.
    """,
    output_type=StudentVerificationOutput
)

@input_guardrail
async def gatekeeper_guardrail(ctx, agent, input):
    result = await Runner.run(gatekeeper_agent, input, run_config=config)
    return GuardrailFunctionOutput(
        output_info=result.final_output.response,
        tripwire_triggered=result.final_output.isFromOtherSchool
    )

school_student_agent = Agent(
    name="Student",
    instructions="You are a student trying to enter the school",
    input_guardrails=[gatekeeper_guardrail]
)

async def ex3_main():
    try:
        result = await Runner.run(school_student_agent, 
                                "I'm from XYZ School, can I come in?", 
                                run_config=config)
        print(result.final_output)
    except InputGuardrailTripwireTriggered:
        print("Gatekeeper says: Only our school students are allowed!")

# ==================== Original Code from Class ====================
class PassengerOutput(BaseModel):
    response: str
    isWeightExceed: bool
     
airport_security_guard = Agent(
    name="Airport Security Guard",
    instructions=""" 
        Your task is to check the passenger luggage.
        If passenger's luggage is more then 25KGs, gracefully stop them
    """,
    output_type=PassengerOutput
)

@input_guardrail
async def security_guardrail(ctx, agent, input):
    result = await Runner.run(airport_security_guard, 
                            input, 
                            run_config=config)
    rich.print(result.final_output)
    return GuardrailFunctionOutput(
        output_info=result.final_output.response,
        tripwire_triggered=result.final_output.isWeightExceed
    )

passenger_agent = Agent(
    name='Passenger',
    instructions="You are a passenger agent",
    input_guardrails=[security_guardrail]
)

async def original_main():
    try:
        result = await Runner.run(passenger_agent, 
                                'My luggage weight is 20kgs', 
                                run_config=config)
        print("Passenger is onboarded")
    except InputGuardrailTripwireTriggered:
        print('Passenger cannot check-in')

class MessageOutput(BaseModel):
    response: str

class PHDOutput(BaseModel):
    isPHDLevelResponse: bool

phd_guardrail_agent = Agent(
    name="PHD Guardrail Agent",
    instructions="""
        You are a PHD Guardrail Agent that evaluates if text is too complex for 8th grade students. 
        If the response is very hard to read for an eight grade student, deny the agent response
    """,
    output_type=PHDOutput
)

@output_guardrail
async def PHD_guardrail(ctx, agent: Agent, output) -> GuardrailFunctionOutput:
    result = await Runner.run(phd_guardrail_agent, 
                            output.response,  
                            run_config=config)
    return GuardrailFunctionOutput(
        output_info=result.final_output,
        tripwire_triggered=result.final_output.isPHDLevelResponse
    )

eigth_grade_std = Agent(
    name="Eight grade student",
    instructions="""
        1. You are an agent that answer query to a eight standard student. Keep your vocabulary simple and easy. 
        2. If asked to give answers in most difficult level use the most hardest english terms
    """,
    output_type=MessageOutput,
    output_guardrails=[PHD_guardrail]
)

async def original_og_main():
    query = "What are trees? Explain using the most complex scientific terminology possible"
    # query = "What are trees? Explain in simple words"
    try:
        result = await Runner.run(eigth_grade_std, query, run_config=config)
        print(result.final_output)
    except OutputGuardrailTripwireTriggered:
        print('Agent output is not according to the expectations')

# ==================== Main Execution ====================
async def main():
    print("\n=== Running Exercise 1 ===")
    await ex1_main()
    
    print("\n=== Running Exercise 2 ===")
    await ex2_main()
    
    print("\n=== Running Exercise 3 ===")
    await ex3_main()
    
    print("\n=== Running Classo7 ===")
    await original_main()
    await original_og_main()

if __name__ == "__main__":
    asyncio.run(main())