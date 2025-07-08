import re
import os
from datetime import datetime

def parse_log_entry(log_line: str) -> dict | None:
    """
    Parses a single log line into a dictionary of its components.
    Assumes log format: YYYY-MM-DD HH:MM:SS,ms:LEVEL:NAME: MESSAGE
    """
    match = re.match(r'(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2},\d{3}):([A-Z]+):([^:]+): (.*)', log_line)
    if match:
        timestamp_str, level, name, message = match.groups()
        try:
            dt_object = datetime.strptime(timestamp_str.split(',')[0], "%Y-%m-%d %H:%M:%S")
        except ValueError:
            dt_object = None
        
        return {
            "timestamp": timestamp_str,
            "datetime": dt_object,
            "level": level,
            "name": name,
            "message": message.strip()
        }
    return None

def parse_log_file(file_path: str) -> list[dict]:
    """
    Reads a log file and parses each line into a list of dictionaries.
    """
    parsed_data = []
    if not os.path.exists(file_path):
        return []
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            for line in f:
                entry = parse_log_entry(line)
                if entry:
                    parsed_data.append(entry)
    except Exception as e:
        print(f"Error reading or parsing log file {file_path}: {e}")
    return parsed_data

def analyze_logs(base_dir: str) -> str:
    """
    Analyzes log files in a given directory and returns a formatted summary string.
    """
    log_files = {
        "bot_activity.log": os.path.join(base_dir, "bot_activity.log"),
        "cleaner.log": os.path.join(base_dir, "cleaner.log")
    }

    analysis_output = "--- Log Analysis Summary ---\n"
    found_issues = False

    for log_name, file_path in log_files.items():
        analysis_output += f"\n--- Parsing {log_name} ---\n"
        logs = parse_log_file(file_path)
        if logs:
            log_issues = []
            for entry in logs:
                if entry['level'] in ['ERROR', 'WARNING'] or "error" in entry['message'].lower() or "fail" in entry['message'].lower():
                    log_issues.append(f"[{entry['level']}] {entry['timestamp']} - {entry['message']}")
            
            if log_issues:
                found_issues = True
                analysis_output += "\n".join(log_issues) + "\n"
            else:
                analysis_output += "No significant issues found.\n"
        else:
            analysis_output += "No log entries found or file does not exist.\n"
            
    if not found_issues:
        return "✅ All logs analyzed. No significant issues found."
        
    return analysis_output

if __name__ == "__main__":
    # Example of running as a standalone script
    # Uses the current directory as the base directory
    current_directory = os.path.dirname(os.path.abspath(__file__))
    summary = analyze_logs(current_directory)
    print(summary)

