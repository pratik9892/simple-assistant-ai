from datetime import datetime
import requests
import json
from Utilities.utilities import log
from rag import search
def get_curr_time():
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def get_city_weather(city):
    url = f"https://wttr.in/{city}?format=3"
    response = requests.get(url,timeout=10)
    response.raise_for_status()
    return response.text

def get_user_city():
    return "Bangalore"


def search_knowledge_base(query):
   results = search(query,3)

   log(f"Knowledge base search : {query}")
   for r in results:
      log(f" {r['score']:.2f} - {r['source']} #{r['chunk_id']}")

   blocks = []
   for r in results:
      sourceFile = r['source']
      chunkId = r['chunk_id']
      chunkText = r['text']

      header = f"[{sourceFile} #{chunkId}]"
      block = f"{header}\n{chunkText}"
      blocks.append(block)
      
   blocks[0]
   return "\n\n".join(blocks)
 
   

TOOLS = [
    {
        "type" : "function",
        "function" : {
                "name" : "get_curr_time",
                "description" : "Returns the current date and time as a formatted string",
                "parameters" : {"type" : "object" , "properties" : {}}
            }
    },
    {
        "type" : "function",
        "function" : {
            "name" : "get_city_weather",
            "description" : "Returns the live weather of the current city",
            "parameters" : {
                "type" : "object",
                "properties":{
                    "city" : {
                        "type" : "string",
                        "description" : "city to get the weather for"
                    }
                },
                "required" : ["city"]
            }
        }
    },
    {
        "type" : "function",
        "function" : {
            "name" : "get_user_city",
            "description" : "Returns the city where the user lives",
            "parameters" : {"type" : "object" , "properties" : {}}
        }
    },
    {
       "type" : "function",
       "function" : {
          "name" : "search_knowledge_base",
          "description" : "Searches the nimbus technologies company knowledge base: hiring process, hr docs, leave policy, medical policy,"
                          "rca docs, tech docs, travel expense. Use it for any questions related to company policy or document. Returns most relevant chunk",
          "parameters" : {
             "type":"object",
             "properties" : {
                "query" : {
                   "type" : "string",
                   "description" : "The users question written as a short, self contained search query"
                }
             },
             "required" : ["query"]
          }
       }
    }
]

FUNCTION_NAME_TO_FUNCTION_MAP = {
    "get_curr_time" : get_curr_time,
    "get_city_weather" : get_city_weather,
    "get_user_city" : get_user_city,
    "search_knowledge_base" : search_knowledge_base
}

def run_tool(tool_call):
    func_name = tool_call.function.name 
    func_args = {}
    try:
      func_args = json.loads(tool_call.function.arguments or "{}")
      log(f"function name : {func_name} , function args : {func_args}")
      func = FUNCTION_NAME_TO_FUNCTION_MAP[func_name]
      result = func(**func_args)
    except Exception as e:
      result = f"Error while running {func_name} : {e}"
      log(f"   -> {result}") 

    return str(result)  