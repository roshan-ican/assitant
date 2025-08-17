# main.py - Fixed
from fastapi import FastAPI, HTTPException
from brain.learner import JarvisLearner
from brain.suggester import JarvisSuggester
from data.storage import TaskStorage
from integrations.notion import NotionIntegration
import uvicorn
import logging

logging.basicConfig(level=logging.INFO)


app = FastAPI(title="Jarvis Todo Assistant")

# Initialize components once
storage = TaskStorage()
learner = JarvisLearner()
suggester = JarvisSuggester(learner, storage)
notion = NotionIntegration()


@app.post("/task")
@app.post("/task")
@app.post("/task")
async def add_task(data: dict):
    task_text = data.get('text')
    user_id = data.get('user_id')

    if not task_text or not user_id:
        raise HTTPException(status_code=400, detail="text and user_id required")

    # Predict category using ML
    predicted_category = learner.predict_category(task_text, user_id)
    logging.info(f"Predicted category: {predicted_category}")

    # Create in Notion with category
    notion_page = notion.create_task(user_id, task_text, predicted_category)

    # Save for ML and learn patterns
    storage.save_ml_data(user_id, task_text, notion_page['id'])
    learner.learn_from_task(task_text, user_id)

    return {
        "notion_id": notion_page['id'],
        "predicted_category": predicted_category
    }


@app.post("/tasks/bulk")
async def create_bulk_tasks(data: dict):
    """Create multiple tasks at once"""
    tasks = data.get('tasks', [])
    user_id = data.get('user_id')

    if not tasks or not user_id:
        raise HTTPException(status_code=400, detail="tasks array and user_id required")

    created_tasks = []

    for task_text in tasks:
        if task_text.strip():  # Skip empty tasks
            # Predict category
            predicted_category = learner.predict_category(task_text, user_id)

            # Create in Notion
            notion_page = notion.create_task(user_id, task_text, predicted_category)

            # Save for ML
            storage.save_ml_data(user_id, task_text, notion_page['id'])
            learner.learn_from_task(task_text, user_id)

            created_tasks.append({
                "notion_id": notion_page['id'],
                "text": task_text,
                "category": predicted_category
            })

    return {"created_tasks": created_tasks, "count": len(created_tasks)}

if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)