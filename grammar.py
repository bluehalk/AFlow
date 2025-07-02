import re

response = "<modification>add a new node</modification><graph>add a new node</graph><prompt>add a new node</prompt>"
pattern = r"<(\w+)>(.*?)</\1>"
matches = re.findall(pattern, response, re.DOTALL)

#NOTE(sjh) 字段名为键，字段值为值
found_fields = {match[0]: match[1].strip() for match in matches}
print(found_fields)