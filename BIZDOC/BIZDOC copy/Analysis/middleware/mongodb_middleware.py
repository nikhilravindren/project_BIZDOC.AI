# myapp/middleware/mongodb_middleware.py

from django.db import connections

class MongoDBConnectionMiddleware:
    def __init__(self, get_response):
        self.get_response = get_response

    def __call__(self, request):
        response = self.get_response(request)
        self.close_connections()
        return response

    def close_connections(self):
        # Close all database connections
        try:
            for alias in connections:
                conn = connections[alias]
                if conn.connection:
                    conn.close()
        except Exception as e:
            # Log or handle the exception if needed
            print(f"Error closing database connection: {e}")
