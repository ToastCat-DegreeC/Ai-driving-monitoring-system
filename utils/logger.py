import csv
import os
from datetime import datetime

class SessionLogger:
    def __init__(self, log_dir="logs"):
        self.log_dir = log_dir
        if not os.path.exists(self.log_dir):
            os.makedirs(self.log_dir)
        
        self.filename = f"session_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
        self.file_path = os.path.join(self.log_dir, self.filename)
        self._write_header()

    def _write_header(self):
        try:
            header = [
                "Timestamp", "HeartRate", "SpO2", "HRV", "FatigueProb", 
                "Status", "Distraction", "IsValid"
            ]
            with open(self.file_path, mode='w', newline='') as f:
                writer = csv.writer(f)
                writer.writerow(header)
        except Exception as e:
            print(f"Logging error: {e}")

    def log(self, data):
        """
        Logs a dictionary of monitoring results to the CSV file.
        """
        try:
            row = [
                datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                f"{data.get('hr', 0):.1f}",
                f"{data.get('spo2', 0):.1f}",
                f"{data.get('hrv', 0):.1f}",
                f"{data.get('fatigue_prob', 0):.2f}",
                data.get('status', "Unknown"),
                data.get('distraction', "Unknown"),
                "1" if data.get('valid', False) else "0"
            ]
            with open(self.file_path, mode='a', newline='') as f:
                writer = csv.writer(f)
                writer.writerow(row)
        except Exception as e:
            print(f"Logging error: {e}")
