import sqlite3
import logging
from datetime import datetime

DB_PATH = "self_healing.db"

def initialize_db():
    """Creates the database and the log table if they don't exist."""
    try:
        with sqlite3.connect(DB_PATH) as conn:
            cursor = conn.cursor()
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS healing_log (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp TEXT NOT NULL,
                    event TEXT NOT NULL,
                    details TEXT
                )
            """)
            # Create a new table for learned data if it doesn't exist
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS learned_data (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp TEXT NOT NULL,
                    data TEXT NOT NULL
                )
            """)
            conn.commit()
            logging.info("Database initialized successfully, including learned_data table.")
    except sqlite3.Error as e:
        logging.error(f"Database error during initialization: {e}")

def log_healing_event(event, details=""):
    """Logs a self-healing event to the database."""
    try:
        with sqlite3.connect(DB_PATH) as conn:
            cursor = conn.cursor()
            timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            cursor.execute(
                "INSERT INTO healing_log (timestamp, event, details) VALUES (?, ?, ?)",
                (timestamp, event, details)
            )
            conn.commit()
            logging.info(f"Logged healing event: {event}")
    except sqlite3.Error as e:
        logging.error(f"Failed to log healing event: {e}")

def save_learned_data(data_list):
    """Saves a list of learned data strings to the database."""
    try:
        with sqlite3.connect(DB_PATH) as conn:
            cursor = conn.cursor()
            # Clear existing learned data before saving the new list
            cursor.execute("DELETE FROM learned_data")
            timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            for data_item in data_list:
                cursor.execute(
                    "INSERT INTO learned_data (timestamp, data) VALUES (?, ?)",
                    (timestamp, data_item)
                )
            conn.commit()
            logging.info(f"Saved {len(data_list)} learned data items to the database.")
    except sqlite3.Error as e:
        logging.error(f"Failed to save learned data: {e}")

def load_learned_data():
    """Loads all learned data strings from the database."""
    try:
        with sqlite3.connect(DB_PATH) as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT data FROM learned_data ORDER BY timestamp")
            learned_data = [row[0] for row in cursor.fetchall()]
            logging.info(f"Loaded {len(learned_data)} learned data items from the database.")
            return learned_data
    except sqlite3.Error as e:
        logging.error(f"Failed to load learned data: {e}")
        return [] # Return empty list on error

if __name__ == '__main__':
    initialize_db()
    log_healing_event("Test Event", "This is a test of the self-healing log.")
    # Example of saving and loading learned data
    # save_learned_data(["fact 1", "fact 2"])
    # loaded_data = load_learned_data()
    # print(f"Loaded data: {loaded_data}")
    print("Database initialized and test event logged.")
