import google.generativeai as genai
from google.generativeai import GenerativeModel
genai.configure(api_key='AIzaSyCIf7E2PdGo357stA23ClZkoWuQeMxAPXs')
model = GenerativeModel('gemini-pro')
response = model.generate_content('The opposite of hot is')
print(response.text)


