import requests

# Define the repository details
owner = 'Deadsg'
repo = 'B.A.T.M.A.N'
path = 'F:\Python github'

# Make a request to the GitHub API to get the contents of the file
url = f'https://api.github.com/repos/{owner}/{repo}/contents/{path}'
response = requests.get(url)

# Check if the request was successful (status code 200)
if response.status_code == 200:
    # The response contains information about the file, including its content
    file_info = response.json()
    content = file_info['content']
    
    # Decode the content from base64 (GitHub API encodes content in base64)
    content = content.decode('base64')
    
    print(content)
else:
    print(f"Error fetching file. Status code: {response.status_code}")