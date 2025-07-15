import requests
import json
import re
def validate_response(response: str):
  # try:
      # print("response", response)
  pattern = r"<(\w+)>(.*?)</\1>"
  matches = re.findall(pattern, response, re.DOTALL)

  #NOTE(sjh) 字段名为键，字段值为值
  found_fields = {match[0]: match[1].strip() for match in matches}
  # print("found_fields", found_fields)

  # math_answer是必需的，code是可选的
  # if "math_answer" not in found_fields or not found_fields["math_answer"]:
      # return False, {"error": "Missing math_answer field"}

  return found_fields
  # except Exception:
      # return False, None

str = "<math_reasoning>1</math_reasoning>agjalgjaljgal<python_code>2</python_code><final_synthesis>3</final_synthesis><final_answer>4</final_answer>"
res = validate_response(str)
if res:
  print(res)
else:
  print("error")