import sys
from database import db_manager

"""
This script is used to view and interact with the database.

Usage:
  python db_actions.py tables
  python db_actions.py view <table>
  python db_actions.py query "<SQL>"
  python db_actions.py view_raw
  python db_actions.py view_scores
  python db_actions.py [view] (legacy)
"""

def print_all_models():
    models = db_manager.load_models_from_db()
    if not models:
        print("No models found in the database.")
        return
    print(f"Found {len(models)} models:")
    for model in models:
        print(f"- Name: {model['name']}")
        print(f"  Model ID: {model['model_id']}")
        print(f"  Scores:")
        for category, scores in model['scores'].items():
            print(f"    {category}: accuracy={scores['accuracy']}, speed={scores['speed']}, cost={scores['cost']}")
        print()

def print_all_models_raw():
    models = db_manager.load_all_models_raw()
    if not models:
        print("No models found in the models table.")
        return
    print(f"Found {len(models)} models in the models table:")
    for model in models:
        print(f"- Name: {model.get('name')}")
        print(f"  Model ID: {model.get('model_id')}")
        print(f"  is_huggingface: {model.get('is_huggingface')}")
        print(f"  huggingface_url: {model.get('huggingface_url')}")
        print()

def print_all_model_scores():
    conn = db_manager.get_connection()
    if not conn:
        print("Could not connect to database.")
        return
    try:
        cursor = conn.cursor()
        cursor.execute("SELECT model_id, category, accuracy, speed, cost FROM model_scores ORDER BY model_id, category;")
        rows = cursor.fetchall()
        if not rows:
            print("No model scores found in the model_scores table.")
            return
        print(f"Found {len(rows)} model score entries:")
        for row in rows:
            print(f"- Model ID: {row[0]}")
            print(f"  Category: {row[1]}")
            print(f"  Accuracy: {row[2]}")
            print(f"  Speed: {row[3]}")
            print(f"  Cost: {row[4]}")
            print()
    except Exception as e:
        print(f"Error loading model scores: {e}")
    finally:
        if cursor:
            cursor.close()

def list_tables():
    conn = db_manager.get_connection()
    if not conn:
        print("Could not connect to database.")
        return
    try:
        cursor = conn.cursor()
        cursor.execute("SELECT table_name FROM information_schema.tables WHERE table_schema='public' ORDER BY table_name;")
        rows = cursor.fetchall()
        if not rows:
            print("No tables found.")
            return
        print("Tables:")
        for row in rows:
            print(f"- {row[0]}")
    except Exception as e:
        print(f"Error listing tables: {e}")
    finally:
        if cursor:
            cursor.close()

def view_table(table_name):
    conn = db_manager.get_connection()
    if not conn:
        print("Could not connect to database.")
        return
    try:
        cursor = conn.cursor()
        cursor.execute(f"SELECT * FROM {table_name} LIMIT 100;")
        rows = cursor.fetchall()
        colnames = [desc[0] for desc in cursor.description]
        if not rows:
            print(f"No rows found in {table_name}.")
            return
        print(f"Rows in {table_name} (up to 100):")
        print(colnames)
        for row in rows:
            print(row)
    except Exception as e:
        print(f"Error viewing table {table_name}: {e}")
    finally:
        if cursor:
            cursor.close()

def run_query(sql):
    print("WARNING: You are running a raw SQL query. Be careful!")
    conn = db_manager.get_connection()
    if not conn:
        print("Could not connect to database.")
        return
    try:
        cursor = conn.cursor()
        cursor.execute(sql)
        if cursor.description:
            colnames = [desc[0] for desc in cursor.description]
            rows = cursor.fetchall()
            print(colnames)
            for row in rows:
                print(row)
        else:
            conn.commit()
            print("Query executed successfully.")
    except Exception as e:
        print(f"Error running query: {e}")
    finally:
        if cursor:
            cursor.close()

def main():
    if len(sys.argv) < 2:
        print("Usage: python db_actions.py [tables|view <table>|query \"<SQL>\"|view_raw|view_scores|view]")
        sys.exit(1)
    action = sys.argv[1]
    if action == "tables":
        list_tables()
    elif action == "view" and len(sys.argv) == 3:
        view_table(sys.argv[2])
    elif action == "query" and len(sys.argv) == 3:
        run_query(sys.argv[2])
    elif action == "view":
        print_all_models()
    elif action == "view_raw":
        print_all_models_raw()
    elif action == "view_scores":
        print_all_model_scores()
    else:
        print(f"Unknown or incomplete action: {sys.argv[1:]}")
        print("Usage: python db_actions.py [tables|view <table>|query \"<SQL>\"|view_raw|view_scores|view]")
        sys.exit(1)

if __name__ == "__main__":
    main() 