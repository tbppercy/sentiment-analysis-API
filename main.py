import os
from fastapi import FastAPI, HTTPException, responses
from pydantic import BaseModel
from openai import AsyncOpenAI
from dotenv import load_dotenv
import instructor


load_dotenv()
token = os.getenv("TOKEN")
print(token)
endpoint = "https://models.inference.ai.azure.com"

client = instructor.from_openai(
    AsyncOpenAI(base_url=endpoint,api_key=token)
    )
settings = {
    "model": "gpt-4o-mini",
    "temperature": 0,
}

app = FastAPI()


class Request(BaseModel):
    text: str

class SentimentResponse(BaseModel):
    text: str
    sentiment: str
    score: float
    analysis: str


@app.get("/")
async def root():
    return responses.RedirectResponse('/docs')


@app.post("/check_sentiment")
async def analyze(request: Request):
    try:
        response = (
        await client.chat.completions.create(
            messages=[
                {
                    "role": "system",
                    "content": """Analyze the sentiment of the following text. Please categorize it as one of the following.
                         Positive: if the tone is optimistic, happy, or approving.
                         Negative: if the tone is critical, sad, or disapproving.
                         Neutral: if the tone is objective or lacks strong emotion.
                        """,
                },
                {"role": "user", "content": request.text},
            ],
            **settings,
            response_model=SentimentResponse
        )
        )
  
        return response
       

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
