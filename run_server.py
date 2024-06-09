import uvicorn
from server.py import app  # Replace 'your_module_name' with the actual name of your module

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)

