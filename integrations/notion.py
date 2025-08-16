# integrations/notion.py
import os
from notion_client import Client
from dotenv import load_dotenv

load_dotenv()


class NotionIntegration:
    def __init__(self):
        self.notion = Client(auth=os.getenv('NOTION_TOKEN'))
        self.database_id = None

    def ensure_database(self):
        """Create or find the GTD database"""
        if self.database_id:
            return self.database_id

        # Search for existing database
        try:
            search_response = self.notion.search(
                query='Daily Tasks',
                filter={'value': 'database', 'property': 'object'}
            )

            if search_response['results']:
                self.database_id = search_response['results'][0]['id']
                return self.database_id
        except:
            pass

        # Create new database (needs a parent page)
        parent_page_id = os.getenv('NOTION_PARENT_PAGE_ID')  # Add this to .env

        new_db = self.notion.databases.create(
            parent={'type': 'page_id', 'page_id': parent_page_id},
            title=[{'type': 'text', 'text': {'content': 'Daily Tasks'}}],
            properties={
                'Task': {'title': {}},
                'Status': {
                    'select': {
                        'options': [
                            {'name': 'Todo', 'color': 'yellow'},
                            {'name': 'Done', 'color': 'green'}
                        ]
                    }
                },
                'User': {'rich_text': {}}
            }
        )

        self.database_id = new_db['id']
        return self.database_id

    # integrations/notion.py
    def create_task(self, user_id, task_text, category=None):
        db_id = self.ensure_database()

        properties = {
            'Task': {'title': [{'text': {'content': task_text}}]},
            'Status': {'select': {'name': 'Todo'}}
        }

        # Add category if predicted and database has Category property
        if category and category != "unknown":
            properties['Category'] = {'select': {'name': category.title()}}

        return self.notion.pages.create(
            parent={'database_id': db_id},
            properties=properties
        )