#standard library imports
from os import getenv
from types import SimpleNamespace

#third party imports
from openai import OpenAI
from dotenv import load_dotenv

#local app imports
from code_snippets.tools import TOOLS,run_tool
from Utilities.utilities import log

load_dotenv() #by default looks for .env 

client = OpenAI(
    base_url= getenv("BASE_URL"),
    api_key=getenv("API_KEY")
)

messages = []

MAX_STEPS = 5 # max no. of steps a chained for loop can run 

messages.append({
      "role" : "system",
      "content" : (
            "You are a helpful, humble assistant for Nimbus Technologies employees. "
            "Answer general questions (programming, science, advice) from your own knowledge. "
            "Use get_current_time, get_city_weather and get_user_city only for the current time, the weather, or the user's city. "
            "For ANY question about Nimbus Technologies policies, benefits, leave, medical cover, HR, hiring, tech processes, incident reports or travel expenses, "
            "you MUST first call search_knowledge_base. "
            "Answer company questions ONLY from the text that search_knowledge_base returns. Do not use outside knowledge for company facts. "
            "After the answer, cite the source like (leave_policy.txt #3). "
            "If the returned text does not contain the answer, say exactly: I could not find this in the knowledge base. "
            "Keep answers under 100 words."
      ),
}) # system prompt given at the start

def stream_chat(history):
     stream = client.chat.completions.create(
          model=getenv("MODEL"),
          messages=history,
          tools=TOOLS,
          stream=True
     )

     full_response = ""
     calls = {}
     label_printed = False

     try:
       for chunk in stream:
          if not chunk.choices:
               continue
          delta = chunk.choices[0].delta

          # check do we content or tool calls

          # content streaming
          if delta.content:
               if not label_printed:
                    print("Assistant : ", end="", flush=True)
                    label_printed = True
               print(delta.content, end="", flush=True)
               full_response += delta.content

          # tool calls
          for tc in (delta.tool_calls or []):
               obj = calls.setdefault(
                    tc.index,
                    SimpleNamespace(id=None,function=SimpleNamespace(name="",arguments=""))
               )
               if tc.id:
                    obj.id = tc.id

               if tc.function and tc.function.name:
                    obj.function.name += tc.function.name

               if tc.function and tc.function.arguments:
                    obj.function.arguments += tc.function.arguments 

     except KeyboardInterrupt as e:
          print(f" \n Stopped by you🤔")
          calls={}

     if label_printed:
          print()

     calls_list = []
     for key in sorted(calls):
          calls_list.append(calls[key])

     return full_response,calls_list



def build_tool_calls_payload(tool_calls):
     """
     Converts our collected tool calls into dictionary format that the api excepts inside the assitant message
     """
     payload = []
     for tool_call in tool_calls:
          tool_call_dic = {
               "id" : tool_call.id,
               "type" : "function",
               "function" : {
                    "name" : tool_call.function.name,
                    "arguments" : tool_call.function.arguments,
               }
          }
          payload.append(tool_call_dic)

     return payload


while True:

        user_input=input("\n You : ")

        if user_input.lower() in ["exit","bye","quit"]:
            break

        messages.append({
              "role" : "user",
              "content" : user_input
        })


        for steps in range(MAX_STEPS):


            full_response, tool_calls = stream_chat(messages)
            #stream here and collect tool calls 

            if not tool_calls:
                 break

            log(f"Step : {steps + 1} : model requested {len(tool_calls)} tool calls")

            messages.append({
                 "role" : "assistant",
                 "content" : full_response,
                 "tool_calls" : build_tool_calls_payload(tool_calls)
            })

          # chained tool calling
            for tool_call in tool_calls:
                result = run_tool(tool_call) 
                messages.append({
                      "role" : "tool",
                      "tool_call_id" : tool_call.id,
                      "content" : str(result)
                 })

          # if maxsteps exceed
        else:
             messages.append({
                  "role" : "assistant",
                  "content" : "exhausted max steps we ran out of steps the models coudlnt give the answer sorry"
             })
             print("Assistant : exhausted max steps we ran out of steps the models coudlnt give the answer sorry" )
             continue

        messages.append({
             "role" : "assistant",
             "content" : full_response
        })


